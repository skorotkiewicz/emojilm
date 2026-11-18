"""Translate explicit emoji and bracketed word components in place."""

from .corpora import canonical, hint_dictionary
from .data import REBUSES
from .rebus_bank import BANK
from .tokenizer import emoji_at


def bracket_words(text):
    depth = 0
    for i, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = i
            depth += 1
        elif char == "]" and depth:
            depth -= 1
            if depth == 0:
                part = text[start + 1 : i]
                if part and not any(mark in part for mark in "[]\r\n"):
                    yield start, i + 1, part


def bracket_runs(text):
    parts = []
    run_start = run_end = 0
    for start, end, part in bracket_words(text):
        if parts and text[run_end:start].strip(" \t"):
            yield run_start, run_end, parts
            parts = []
        if not parts:
            run_start = start
        run_end = end
        parts.append(part)
    if parts:
        yield run_start, run_end, parts


def compact_emoji_runs(text):
    i = 0
    while i < len(text):
        emoji = emoji_at(text, i)
        if emoji is None:
            i += 1
            continue
        start = i
        parts = []
        while emoji is not None:
            parts.append(emoji)
            i += len(emoji)
            emoji = emoji_at(text, i) if i < len(text) else None
        yield start, i, parts


class Composer:
    def __init__(
        self,
        readings,
        known_combinations=(),
        *,
        partial_encode=False,
        prefer_known_runs=False,
    ):
        self.readings = {}
        self.symbols = {}
        self.partial_encode = partial_encode
        self.prefer_known_runs = prefer_known_runs
        self.known_combinations = {canonical(emoji) for emoji in known_combinations}
        priorities = {}
        for emoji, words in readings.items():
            if words:
                self.readings.setdefault(canonical(emoji), words[0])
            for rank, word in enumerate(words):
                key = canonical(word)
                if key not in priorities or rank < priorities[key]:
                    self.symbols[key] = emoji
                    priorities[key] = rank

    def read_component(self, emoji):
        return self.readings.get(canonical(emoji))

    def write_component(self, word):
        return self.symbols.get(canonical(word))

    def read_run(self, parts):
        if self.prefer_known_runs and len(parts) > 1:
            known = self.read_component("".join(parts))
            if known:
                return known
        replacements = [self.read_component(emoji) for emoji in parts]
        if not all(replacements):
            return None
        separator = "" if all(word.isalpha() for word in replacements) else " "
        return separator.join(replacements)

    def render_encode_run(self, source, replacements):
        if all(replacements):
            return "".join(replacements)
        if not self.partial_encode or not any(replacements):
            return source
        pieces = []
        offset = 0
        previous = None
        for (start, end, _), glyph in zip(bracket_words(source), replacements):
            if not (previous and glyph):
                pieces.append(source[offset:start])
            pieces.append(glyph or source[start:end])
            offset = end
            previous = glyph
        return "".join(pieces)

    def translate(self, task, source, *, automatic=False):
        if task == "encode":
            runs = bracket_runs(source)
        elif task in {"decode", "rebus"}:
            runs = compact_emoji_runs(source)
        else:
            return None

        pieces, offset = [], 0
        for start, end, parts in runs:
            original = source[start:end]
            if task == "encode":
                replacements = [self.write_component(word) for word in parts]
                if automatic and not all(replacements):
                    return None
                replacement = self.render_encode_run(original, replacements)
            else:
                if automatic and (
                    len(parts) < 2 or canonical(original) in self.known_combinations
                ):
                    return None
                # Adjacent symbols form a compound. Explicit separators
                # stay in the source, and multiword readings need spaces.
                replacement = self.read_run(parts)
                if automatic and replacement is None:
                    return None
            pieces.append(source[offset:start])
            pieces.append(replacement if replacement is not None else original)
            offset = end
        pieces.append(source[offset:])
        candidate = "".join(pieces)
        return candidate if candidate != source else None


def default_composer(*, partial_encode=False):
    known = [entry["emoji"] for entry in REBUSES] + [entry[0] for entry in BANK]
    return Composer(hint_dictionary(), known, partial_encode=partial_encode)
