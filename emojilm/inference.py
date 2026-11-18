"""Load a bundled checkpoint and generate answers with PyTorch."""

import math
import re

import torch

from .composition import Composer, bracket_words
from .config import Config
from .corpora import canonical
from .edits import (
    EDIT_ALIASES,
    EDIT_TASK,
    KEEP,
    apply_edits,
    contextual_label,
    edit_label,
    keep_seal_verb,
)
from .model import EmojiLM
from .tokenizer import Tokenizer


def load(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "emojilm-1":
        raise ValueError("Expected an EmojiLM checkpoint from this project")
    tok = Tokenizer.from_dict(checkpoint["tokenizer"])
    cfg = Config.from_dict(checkpoint["cfg"])
    if cfg.vocab != tok.size:
        raise ValueError("Checkpoint model and tokenizer vocabulary sizes differ")
    model = EmojiLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model.edit_candidates = checkpoint.get("edit_candidates")
    model.decode_readings = checkpoint.get("decode_readings")
    model.sense_heads = {
        label: (head["choices"], head["weight"].to(device), head["bias"].to(device))
        for label, head in checkpoint.get("sense_heads", {}).items()
    }
    model.noun_verb_gate = checkpoint.get("noun_verb_gate")
    if model.noun_verb_gate:
        for key in ("weight", "bias"):
            model.noun_verb_gate[key] = model.noun_verb_gate[key].to(device)
    model.to(device).eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, task, source, hint=None, max_new=64, *, require_eos=False):
    prompt, _ = tok.build(task, source, hint=hint)
    if len(prompt) > model.cfg.ctx:
        raise ValueError(
            f"Prompt needs {len(prompt)} tokens but the model context is {model.cfg.ctx}"
        )
    device = model.tok_emb.weight.device
    tokens = torch.empty((1, len(prompt) + max_new), dtype=torch.long, device=device)
    tokens[0, : len(prompt)] = torch.tensor(prompt, device=device)
    eos, alt = tok.sid["<eos>"], tok.sid["<alt>"]
    forbidden = [
        token
        for token in tok.sid.values()
        if token != eos and (task != "rebus" or token != alt)
    ]
    answers, current = [], []
    prediction_count = min(max_new, model.cfg.ctx - len(prompt) + 1)
    for pos in range(len(prompt), len(prompt) + prediction_count):
        logits = model(tokens[:, :pos])[0, -1]
        logits[forbidden] = -torch.inf
        logits[tok.used :] = -torch.inf
        token = int(logits.argmax())
        if token == eos:
            break
        if token == alt:
            answers.append(tok.decode(current))
            current = []
        else:
            current.append(token)
        tokens[0, pos] = token
    else:
        if require_eos:
            return []
    answers.append(tok.decode(current))
    return list(dict.fromkeys(answer for answer in answers if answer))


_CONTEXT_WORDS = (
    frozenset(
        "a an the this that these those my your his her its our their each every another some".split()
    ),
    frozenset("i you he she it we they".split()),
    frozenset("can could will would may might must shall should do does did".split()),
    frozenset({"to"}),
)


@torch.inference_mode()
def span_features(model, tok, focused):
    source, marker = focused.rsplit("\n", 1)
    index = int(marker.split(" ", 1)[0]) - 1
    start, end, _ = list(bracket_words(source))[index]
    # Capture the left context, the marked span, and nearby words to its right.
    # Each occurrence gets its own representation, including repeated labels.
    middle = tok.encode(f"[{edit_label(focused)}]")
    available = model.cfg.ctx - len(middle) - 1
    if available < 0:
        raise ValueError("Marked word exceeds classifier context")
    left_context = tok.encode(source[:start].rstrip())
    left_count = min(len(left_context), 24, available)
    left = [tok.sid["<bos>"], *left_context[len(left_context) - left_count :]]
    right = tok.encode(source[end:])[: min(12, available - left_count)]
    ids = torch.tensor([left + middle + right], device=model.tok_emb.weight.device)
    h = model.hidden(ids)[0]
    before = re.findall(r"\w+|[^\w\s]", source[:start].casefold())
    after = re.search(r"\w+|[^\w\s]", source[end:].casefold())
    neighbors = (["", ""] + before)[-2:] + [after[0] if after else ""]
    # Scale the binary grammatical cues to the norm of one hidden vector.
    cues = h.new_tensor(
        [word in group for word in neighbors for group in _CONTEXT_WORDS]
    )
    return torch.cat(
        (
            h[len(left) - 1],
            h[len(left) + len(middle) - 1],
            h[-1],
            cues * math.sqrt(model.cfg.dim),
        )
    )


@torch.inference_mode()
def choose_edit(model, tok, focused, choices, max_new):
    gate = getattr(model, "noun_verb_gate", None)
    if gate and edit_label(focused) in gate["labels"]:
        if (
            gate["weight"] @ span_features(model, tok, focused) + gate["bias"]
        ).item() <= 0:
            return KEEP
        choices = [choice for choice in choices if choice != KEEP]
    if len(choices) <= 1:
        return next(iter(choices), None)

    prefix, _ = tok.build(EDIT_TASK, focused)
    limit = min(max_new, model.cfg.ctx - len(prefix) + 1)
    eos = tok.sid["<eos>"]
    device = model.tok_emb.weight.device
    head = getattr(model, "sense_heads", {}).get(edit_label(focused))
    if head is not None:
        labels, weight, bias = head
        if list(choices) == labels and all(
            len(tok.encode(v)) + 1 <= limit for v in labels
        ):
            features = model.hidden(torch.tensor([prefix], device=device))[0, -1]
            return labels[int((weight @ features + bias).argmax())]
    best_value, best_score = None, -math.inf
    for value in choices:
        target = [*tok.encode(value), eos]
        if len(target) > limit:
            continue
        inputs = torch.tensor([prefix + target[:-1]], device=device)
        target_ids = torch.tensor(target, device=device).unsqueeze(-1)
        logits = model(inputs)[0, len(prefix) - 1 :]
        score = float(logits.float().log_softmax(-1).gather(-1, target_ids).sum())
        if score > best_score:
            best_value, best_score = value, score
    return best_value


def generate_edits(model, tok, source, max_new=16):
    if not tok.encode_edits:
        raise ValueError("Use a checkpoint trained with the encode edits dataset")

    candidates = getattr(model, "edit_candidates", None)
    limit = min(max_new, 16)

    def predict(focused):
        if candidates is not None:
            label = edit_label(focused)
            if label in candidates:
                return choose_edit(model, tok, focused, candidates[label], limit)
            alias = EDIT_ALIASES.get(label)
            return alias[0] if alias else None
        prompt, _ = tok.build(EDIT_TASK, focused)
        if len(prompt) > model.cfg.ctx:
            return None
        answers = generate(
            model, tok, EDIT_TASK, focused, max_new=limit, require_eos=True
        )
        return answers[0] if answers else None

    if candidates is None:
        return apply_edits(source, predict)

    animal_only_seal = set(candidates.get("seal", ())) - {KEEP} == {"🦭"}

    def label_for(start, end, word):
        if len(candidates.get(canonical(word), ())) > 1:
            return word
        if animal_only_seal and keep_seal_verb(source, start, end, word):
            return None
        return contextual_label(source, start, end, word)

    return apply_edits(source, predict, label_for=label_for)


def edit_decoder(model, tok, max_new=64):
    readings = getattr(model, "decode_readings", None)
    if readings is not None:
        aliases = {
            canonical(glyph): [reading] for glyph, reading in EDIT_ALIASES.values()
        }
        return Composer(aliases | readings, prefer_known_runs=True)
    return NeuralComposer(model, tok, max_new)


class NeuralComposer(Composer):
    """Use the checkpoint's individual rebus readings in both directions."""

    def __init__(self, model, tok, max_new=64):
        super().__init__({}, partial_encode=True)
        self.model = model
        self.tok = tok
        self.max_new = max_new
        self.device = model.tok_emb.weight.device
        self.glyphs = [emoji for emoji in tok.emoji if len(tok.encode(emoji)) == 1]
        self.emoji_variants = {}
        for emoji in sorted(self.glyphs, key=lambda glyph: "\ufe0f" not in glyph):
            self.emoji_variants.setdefault(canonical(emoji), emoji)
        self.prompts = None

    def component_readings(self, emoji):
        emoji = self.emoji_variants.get(canonical(emoji), emoji)
        if emoji not in self.readings:
            prompt, _ = self.tok.build("rebus", emoji)
            answers = (
                generate(self.model, self.tok, "rebus", emoji, max_new=self.max_new)
                if len(prompt) <= self.model.cfg.ctx
                else []
            )
            words = [answer.strip() for answer in answers]
            self.readings[emoji] = [word for word in words if word.isprintable()]
        return self.readings[emoji]

    def read_component(self, emoji):
        words = self.component_readings(emoji)
        return words[0] if words else None

    def write_component(self, word):
        key = " ".join(word.split())
        if key not in self.symbols:
            ranking = self.rank_glyphs(key)
            glyph = None
            for rank, (candidate, score) in enumerate(ranking):
                # A greedy reading may miss a well-supported complete label.
                decisive = (
                    rank == 0
                    and len(ranking) > 1
                    and score >= math.log(0.05)
                    and score - ranking[1][1] >= math.log(100)
                )
                if decisive or any(
                    reading.casefold() == key.casefold()
                    for reading in self.component_readings(candidate)
                ):
                    glyph = candidate
                    break
            if glyph is None and key.casefold() != key:
                glyph = self.write_component(key.casefold())
            self.symbols[key] = glyph
        return self.symbols[key]

    @torch.inference_mode()
    def rank_glyphs(self, word):
        # A bracketed word inverts the learned single-emoji rebus task.
        labels = self.tok.encode(word)
        width = len(labels) + 4
        if not labels or not self.glyphs or width > self.model.cfg.ctx:
            return []
        if self.prompts is None:
            self.prompts = torch.tensor(
                [self.tok.build("rebus", emoji)[0] for emoji in self.glyphs],
                dtype=torch.long,
                device=self.device,
            )
        label_ids = torch.tensor(labels, dtype=torch.long, device=self.device)
        stops = [self.tok.sid["<eos>"], self.tok.sid["<alt>"]]
        batch_size = min(128, max(1, 2048 // width))
        scores_by_batch = []
        for offset in range(0, len(self.glyphs), batch_size):
            prompts = self.prompts[offset : offset + batch_size]
            inputs = torch.cat((prompts, label_ids.expand(len(prompts), -1)), dim=1)
            logits = self.model(inputs)[:, prompts.size(1) - 1 :]
            log_probs = logits.log_softmax(dim=-1)
            targets = label_ids.view(1, -1, 1).expand(len(prompts), -1, -1)
            words = log_probs[:, :-1].gather(-1, targets).squeeze(-1).sum(-1)
            scores = words + log_probs[:, -1, stops].logsumexp(-1)
            scores_by_batch.append(scores)
        scores = torch.cat(scores_by_batch)
        best = scores.argsort(descending=True, stable=True)[:4]
        return list(
            zip(
                (self.glyphs[i] for i in best.tolist()),
                scores[best].tolist(),
            )
        )
