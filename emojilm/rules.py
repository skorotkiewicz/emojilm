"""Deterministic translations from the GPT project's curated rebuses."""

import json
from pathlib import Path
import re

from .data import emojis_in

LEXICON = Path(__file__).resolve().parents[1] / "data" / "rebus.json"
SELECTOR = r"[\ufe0e\ufe0f]?"


def load_lexicon(path=LEXICON):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emoji_pattern(text):
    # The lexicon contains both compact and spaced spellings. Accept either
    # presentation selector on these glyphs, including glyphs stored bare.
    result = []
    start = 0
    for symbol in emojis_in(text):
        index = text.index(symbol, start)
        result.append(re.escape(text[start:index]))
        pattern = re.escape(symbol).replace("\ufe0f", SELECTOR)
        result.append(pattern if "\ufe0f" in symbol else pattern + SELECTOR)
        start = index + len(symbol)
    result.append(re.escape(text[start:]))
    return "".join(result)


def apply_rules(task, source, entries):
    """Return a translation when a rule matches, or None."""
    brackets, plain, emojis = {}, {}, {}
    if task == "encode":
        for entry in entries:
            brackets.setdefault(re.escape(entry["bracket"]), entry["emoji_spaced"])
            brackets.setdefault(re.escape(f"[{entry['text']}]"), entry["emoji"])
            phrase = re.escape(entry["text"])
            plain.setdefault(r"(?<!\w)" + phrase + r"(?!\w)", entry["emoji"])
        replacements = sorted(brackets, key=len, reverse=True)
        replacements.append(r"\[[^\[\]]*\]")
        replacements.extend(sorted(plain, key=len, reverse=True))
    elif task in {"decode", "rebus"}:
        for entry in entries:
            for key in (entry["emoji"], entry["emoji_spaced"]):
                emojis.setdefault(emoji_pattern(key), entry)
        replacements = sorted(emojis, key=len, reverse=True)
    else:
        return None
    if not replacements:
        return None

    names = {f"rule{i}": pattern for i, pattern in enumerate(replacements)}
    regex = re.compile(
        "|".join(f"(?P<{name}>{pattern})" for name, pattern in names.items()),
        re.IGNORECASE if task == "encode" else 0,
    )
    matched = []

    def replace(match):
        pattern = names[match.lastgroup]
        if task == "encode":
            return brackets.get(pattern, plain.get(pattern, match[0]))
        entry = emojis[pattern]
        matched.append(entry["explanation"])
        return entry["text"]

    result = regex.sub(replace, source)
    if result == source:
        return None
    if task == "rebus":
        result += " (" + " / ".join(dict.fromkeys(matched)) + ")"
    return result


def decode_preserves_text(source, candidate, symbols):
    """Check that the model replaced only runs of emojis."""
    if not symbols:
        return candidate == source
    emoji = (
        "(?:"
        + "|".join(re.escape(e) for e in sorted(set(symbols), key=len, reverse=True))
        + ")"
    )
    literals = re.split(rf"{emoji}(?:\s*{emoji})*", source)
    if not candidate.startswith(literals[0]) or not candidate.endswith(literals[-1]):
        return False
    start, end = len(literals[0]), len(candidate) - len(literals[-1])
    for literal in literals[1:-1]:
        # Each emoji run must have a non-empty replacement before the next anchor.
        first = start
        while first < end and candidate[first].isspace():
            first += 1
        start = candidate.find(literal, first + 1, end)
        if start == -1:
            return False
        start += len(literal)
    return bool(candidate[start:end].strip())
