"""Prepare and apply replacements for individual bracketed spans."""

from functools import cache

from .composition import bracket_words
from .corpora import canonical
from .data import emojis_in
from .tokenizer import emoji_at

EDIT_TASK = "encode-edit"
KEEP = "KEEP"


def edit_prompt(source, index, word):
    return f"{source}\n{index + 1} [{word}]"


def edit_label(prompt):
    marker = prompt.rsplit("\n", 1)[-1].partition(" ")[2]
    return canonical(marker[1:-1])


def is_emoji_replacement(value):
    return bool(value and "".join(emojis_in(value)) == value)


def aligned_replacements(source, target, spans, *, allow_joined=False):
    @cache
    def align(index, pos):
        left = spans[index - 1][1] if index else 0
        if index == len(spans):
            return [()] if target[pos:] == source[left:] else []

        start, end, _ = spans[index]
        literal = source[left:start]
        if target.startswith(literal, pos):
            pos += len(literal)
        elif not (allow_joined and index and literal.isspace()):
            return []
        original = source[start:end]
        candidates = [(len(original), KEEP)] if target.startswith(original, pos) else []
        cursor = pos
        while cursor < len(target) and (emoji := emoji_at(target, cursor)):
            cursor += len(emoji)
            candidates.append((cursor - pos, target[pos:cursor]))

        options = []
        for consumed, value in candidates:
            for rest in align(index + 1, pos + consumed):
                options.append((value, *rest))
                if len(options) == 2:
                    return options
        return options

    choices = align(0, 0)
    return choices[0] if len(choices) == 1 else None


def edit_rows(row):
    if row["task"] != "encode":
        return
    source = row["src"]
    spans = list(bracket_words(source))
    replacements = aligned_replacements(source, row["tgt"], spans, allow_joined=True)
    if replacements is None:
        return
    for index, ((_, _, word), replacement) in enumerate(zip(spans, replacements)):
        yield {
            "task": EDIT_TASK,
            "src": edit_prompt(source, index, word),
            "tgt": replacement,
            "tag": "encode-edits",
        }


def apply_edits(source, predict):
    pieces = []
    offset = 0
    for index, (start, end, word) in enumerate(bracket_words(source)):
        value = predict(edit_prompt(source, index, word))
        pieces.append(source[offset:start])
        if is_emoji_replacement(value):
            pieces.append(value)
        else:
            pieces.append(source[start:end])
        offset = end
    pieces.append(source[offset:])
    return "".join(pieces)


def edit_reference(row):
    if row["task"] != "encode":
        return row
    source = row["src"]
    spans = list(bracket_words(source))
    replacements = aligned_replacements(source, row["tgt"], spans, allow_joined=True)
    if replacements is None:
        return row
    values = iter(replacements)
    return {**row, "tgt": apply_edits(source, lambda _: next(values))}
