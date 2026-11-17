"""Emoji-aware tokenizer: byte fallback + atomic emoji tokens + BPE text.

Vocab layout (4096):
  0..255   raw UTF-8 bytes (lossless fallback)
  256..    specials <bos> <eos> <pad> <decode> <encode> <rebus> <sep> <alt> <hint> <emojify>
  then     modifiers VS16 / ZWJ / keycap / skin tones (unknown ZWJ seqs decompose)
  then     emoji inventory (single codepoints + frequent multi-codepoint seqs)
  then     BPE text tokens (space written as U+2581)
  padding  <unusedN> up to size
"""

import json
from pathlib import Path
import re
from collections import Counter

SPECIALS = [
    "<bos>",
    "<eos>",
    "<pad>",
    "<decode>",
    "<encode>",
    "<rebus>",
    "<sep>",
    "<alt>",
    "<hint>",
    "<emojify>",
]
MODIFIERS = ["\ufe0f", "\u200d", "\u20e3"] + [chr(c) for c in range(0x1F3FB, 0x1F400)]
KEYCAPS = [c + "\ufe0f\u20e3" for c in "0123456789#*"]
SP = "\u2581"

COMMON_EMOJI = (
    "\U0001f600 \U0001f602 \U0001f60d \U0001f60e \U0001f62d \U0001f621 \U0001f64f \U0001f914 \U0001f923 \U0001f970 "
    "\U0001f44d \U0001f44e \U0001f44f \U0001f64c \U0001f4aa \U0001f91d \U0001f440 \U0001f44b \U0001f926 \U0001f937 "
    "\u2764\ufe0f \U0001f494 \U0001f495 \U0001f4af \U0001f525 \u2728 \U0001f389 \U0001f381 \U0001f3c6 \U0001f3af "
    "\U0001f436 \U0001f431 \U0001f981 \U0001f42e \U0001f437 \U0001f414 \U0001f427 \U0001f986 \U0001f41d \U0001f41b "
    "\U0001f98b \U0001f40d \U0001f410 \U0001f411 \U0001f400 \U0001f980 \U0001f419 \U0001f422 \U0001f988 \U0001f433 "
    "\U0001f355 \U0001f354 \U0001f32e \U0001f35f \U0001f35e \U0001f9c0 \U0001f357 \U0001f957 \U0001f34e \U0001f34c "
    "\U0001f353 \U0001f351 \U0001f34b \U0001f95c \U0001f330 \U0001f33d \U0001f955 \U0001f9c4 \U0001f9c5 \U0001f345 "
    "\u2615 \U0001f375 \U0001fad6 \U0001f37a \U0001f377 \U0001f9ca \U0001f36f \U0001f9c2 \U0001f9c8 \U0001f95a "
    "\U0001f967 \U0001f370 \U0001f369 \U0001f36a \U0001f36b \U0001f37f \U0001f964 \U0001f9cb "
    "\U0001f333 \U0001f335 \U0001f337 \U0001f339 \U0001f33b \U0001f343 \U0001f340 \U0001f331 \U0001f33f \U0001f344 "
    "\u2600\ufe0f \U0001f319 \u2b50 \U0001f308 \u26a1 \u2744\ufe0f \U0001f327\ufe0f \U0001f30a \U0001f3d4\ufe0f \U0001f3dd\ufe0f "
    "\U0001f697 \U0001f680 \u2708\ufe0f \U0001f6a2 \U0001f3e0 \U0001f3eb \U0001f6aa \U0001f573\ufe0f \U0001f511 \U0001f3b9 \U0001f3a4 "
    "\U0001f4f1 \U0001f4bb \U0001f4da \U0001f4b0 \U0001f451 \U0001f9e2 \U0001f480 \U0001f47b \U0001f916 \U0001f47d "
    "\u26bd \U0001f3c0 \U0001f3b1 \u26f3 \U0001f3b3 \U0001f3f9 \U0001f94a \U0001f3ae \U0001f3b2 "
    "\U0001f466 \U0001f467 \U0001f468 \U0001f469 \U0001f476 \U0001f474 \U0001f475 \U0001f46e \U0001f477 \U0001f9d1\u200d\U0001f33e "
    "\U0001f441\ufe0f \U0001f441\ufe0f\u200d\U0001f5e8\ufe0f \U0001f9e0 \U0001f9b7 \U0001f445 \U0001f442 \U0001f9b6 "
    "\U0001f1fa\U0001f1f8 \U0001f1ec\U0001f1e7 \U0001f1f5\U0001f1f1 \U0001f1ef\U0001f1f5 \U0001f1e9\U0001f1ea \U0001f1eb\U0001f1f7 "
    "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466 \U0001f469\u200d\u2695\ufe0f \U0001f44d\U0001f3fd \u2764\ufe0f\u200d\U0001f525 "
    "\U0001f9d1\u200d\U0001f4bb \U0001f3f3\ufe0f\u200d\U0001f308 \U0001f468\u200d\U0001f373"
).split()

_RANGES = [
    (0xA9, 0xA9),
    (0xAE, 0xAE),
    (0x203C, 0x203C),
    (0x2049, 0x2049),
    (0x2122, 0x2122),
    (0x2139, 0x2139),
    (0x2194, 0x21AA),
    (0x231A, 0x231B),
    (0x2328, 0x2328),
    (0x23CF, 0x23CF),
    (0x23E9, 0x23FA),
    (0x24C2, 0x24C2),
    (0x25AA, 0x25FE),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3299),
    (0x1F000, 0x1FAFF),
    (0x1FC00, 0x1FFFD),
]
_MODSET = set(MODIFIERS)


def _is_base(ch):
    cp = ord(ch)
    if cp < 0xA9 or ch in _MODSET or 0xE0020 <= cp <= 0xE007F:
        return False
    return any(lo <= cp <= hi for lo, hi in _RANGES)


def _is_ri(ch):
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def emoji_at(s, i):
    """Rule-based extraction of one emoji sequence at i, or None."""
    n = len(s)
    c = s[i]
    if c in "#*0123456789":
        if s[i + 1 : i + 3] == "\ufe0f\u20e3":
            return s[i : i + 3]
        if s[i + 1 : i + 2] == "\u20e3":
            return s[i : i + 2]
        return None
    if _is_ri(c):
        return s[i : i + 2] if i + 1 < n and _is_ri(s[i + 1]) else None
    if not _is_base(c):
        return None
    j = i + 1
    if j < n and s[j] in "\ufe0e\ufe0f":
        j += 1
    if j < n and 0x1F3FB <= ord(s[j]) <= 0x1F3FF:
        j += 1
    if c == "\U0001f3f4" and j < n and 0xE0020 <= ord(s[j]) <= 0xE007F:
        while j < n and 0xE0020 <= ord(s[j]) <= 0xE007F:
            j += 1
        return s[i:j]
    while j + 1 < n and s[j] == "\u200d" and _is_base(s[j + 1]):
        j += 2
        if j < n and s[j] in "\ufe0e\ufe0f":
            j += 1
        if j < n and 0x1F3FB <= ord(s[j]) <= 0x1F3FF:
            j += 1
    return s[i:j]


_PIECE = re.compile(r" ?(?:[A-Za-z]+|[0-9]|[^A-Za-z0-9\s▁])|.", re.DOTALL)


def text_pieces(run):
    for match in _PIECE.finditer(run):
        piece = match.group()
        yield SP + piece[1:] if piece.startswith(" ") else piece


def train_bpe(texts, n_merges, min_freq=2):
    freq = Counter()
    for t in texts:
        for piece in text_pieces(t.replace(SP, " ")):
            freq[piece] += 1
    base = sorted({SP} | {ch for p in freq for ch in p})  # all chars stay representable
    words = {
        tuple(p): f for p, f in freq.items() if f >= min_freq
    }  # prune hapax for speed
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, f in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best, bf = pairs.most_common(1)[0]
        if bf < min_freq:
            break
        merges.append(best)
        nw = {}
        for w, f in words.items():
            if best[0] not in w:
                nw[w] = nw.get(w, 0) + f
                continue  # skip words without the pair
            out, k = [], 0
            while k < len(w):
                if k + 1 < len(w) and (w[k], w[k + 1]) == best:
                    out.append(w[k] + w[k + 1])
                    k += 2
                else:
                    out.append(w[k])
                    k += 1
            key = tuple(out)
            nw[key] = nw.get(key, 0) + f
        words = nw
    return merges, base + [a + b for a, b in merges]


class Tokenizer:
    def __init__(self, emoji, merges, text_tokens, size=4096):
        self.size = size
        self.specials = list(SPECIALS)
        self.modifiers = list(MODIFIERS)
        self.emoji = list(dict.fromkeys(emoji))
        self.merges = [tuple(m) for m in merges]
        self.text_tokens = list(text_tokens)
        toks = self.specials + self.modifiers + self.emoji + self.text_tokens
        if 256 + len(toks) > size:
            raise ValueError(f"vocab overflow {256 + len(toks)}>{size}")
        self.used = 256 + len(toks)
        toks += [f"<unused{i}>" for i in range(size - 256 - len(toks))]
        self.itos = [f"<byte{i}>" for i in range(256)] + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.by_first = {}
        for e in sorted(self.emoji + self.modifiers, key=len, reverse=True):
            self.by_first.setdefault(e[0], []).append(e)
        self.sid = {s: self.stoi[s] for s in SPECIALS}

    @classmethod
    def train(cls, texts, emoji, size=4096):
        clean = []
        for t in texts:
            out, i = [], 0
            while i < len(t):
                seq = emoji_at(t, i)
                if seq is not None:
                    out.append(" ")
                    i += len(seq)
                else:
                    out.append(t[i])
                    i += 1
            clean.append("".join(out))
        emoji = list(dict.fromkeys(list(emoji) + COMMON_EMOJI + KEYCAPS))
        budget = size - 256 - len(SPECIALS) - len(MODIFIERS) - len(emoji)
        if budget < 1:
            raise ValueError("Vocabulary is too small for the emoji inventory")
        merges, tt = train_bpe(clean, budget)
        tt = list(dict.fromkeys([SP, *tt]))[:budget]
        keep = set(tt)
        merges = [m for m in merges if m[0] + m[1] in keep]
        return cls(emoji, merges, tt, size)

    def _bpe(self, piece):
        syms = list(piece)
        while len(syms) > 1:
            best, bi = None, -1
            for k in range(len(syms) - 1):
                r = self.rank.get((syms[k], syms[k + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, k
            if best is None:
                break
            syms[bi : bi + 2] = [syms[bi] + syms[bi + 1]]
        ids = []
        for s in syms:
            if s in self.stoi:
                ids.append(self.stoi[s])
            else:
                ids.extend(s.encode("utf-8"))
        return ids

    def _emoji_ids(self, seq):
        if seq in self.stoi:
            return [self.stoi[seq]]
        ids = []
        for k, part in enumerate(seq.split("\u200d")):
            if k:
                ids.append(self.stoi["\u200d"])
            if part in self.stoi:
                ids.append(self.stoi[part])
                continue
            for ch in part:
                if ch in self.stoi:
                    ids.append(self.stoi[ch])
                else:
                    ids.extend(ch.encode("utf-8"))
        return ids

    def encode(self, text):
        ids, buf, i, n = [], [], 0, len(text)

        def flush():
            for index, part in enumerate("".join(buf).split(SP)):
                if index:
                    ids.extend(SP.encode("utf-8"))
                for piece in text_pieces(part):
                    ids.extend(self._bpe(piece))
            buf.clear()

        while i < n:
            match = None
            for cand in self.by_first.get(text[i], ()):
                if text.startswith(cand, i):
                    match = cand
                    break
            if match is None:
                seq = emoji_at(text, i)
                if seq is not None:
                    match = seq
            if match is not None:
                flush()
                ids.extend(self._emoji_ids(match))
                i += len(match)
            else:
                buf.append(text[i])
                i += 1
        flush()
        return ids

    def decode(self, ids):
        out, buf = [], []
        for token in ids:
            if token < 256:
                buf.append(token)
                continue
            if buf:
                out.append(bytes(buf).decode("utf-8", "replace"))
                buf.clear()
            if token == self.sid["<alt>"]:
                out.append(" / ")
            elif 256 + len(self.specials) <= token < self.used:
                out.append(self.itos[token].replace(SP, " "))
        if buf:
            out.append(bytes(buf).decode("utf-8", "replace"))
        return "".join(out)

    def build(self, task, src, tgt=None, hint=None, *, target_parts=None):
        ids = [self.sid["<bos>"], self.sid["<" + task + ">"]] + self.encode(src)
        if hint:
            ids += [self.sid["<hint>"]] + self.encode(hint)
        ids += [self.sid["<sep>"]]
        mask = [0] * len(ids)
        if target_parts is not None and "".join(target_parts) != tgt:
            raise ValueError("Target parts must concatenate to the target text")
        if tgt is not None:
            target = []
            if target_parts is not None:
                for part in target_parts:
                    target.extend(self.encode(part))
            else:
                alternatives = tgt.split(" <alt> ") if task == "rebus" else [tgt]
                for index, alternative in enumerate(alternatives):
                    if index:
                        target.append(self.sid["<alt>"])
                    target.extend(self.encode(alternative))
            target.append(self.sid["<eos>"])
            ids.extend(target)
            mask.extend([1] * len(target))
        return ids, mask

    def to_dict(self):
        return {
            "format": "emojilm-tok-2",
            "size": self.size,
            "emoji": self.emoji,
            "merges": [list(merge) for merge in self.merges],
            "text_tokens": self.text_tokens,
        }

    @classmethod
    def from_dict(cls, spec):
        if spec.get("format") != "emojilm-tok-2":
            raise ValueError(
                "Unsupported tokenizer format. Rebuild data with this version."
            )
        return cls(spec["emoji"], spec["merges"], spec["text_tokens"], spec["size"])

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
