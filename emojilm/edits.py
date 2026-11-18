"""Prepare and apply replacements for individual bracketed spans."""

from functools import cache
import re

from .composition import bracket_words
from .corpora import canonical
from .data import emojis_in
from .tokenizer import emoji_at

EDIT_TASK = "encode-edit"
KEEP = "KEEP"
# Short forms for CLDR readings used in bracketed text.
EDIT_ALIASES = {
    "eat": ("🍽️", "fork and knife with plate"),
    "think": ("🤔", "thinking face"),
}
SEAL_OBJECT = re.compile(
    r"[ \t]+(?:up[ \t]+)?"
    r"(?:(?:the|an?|my|your|his|her|its|our|their|this|that|these|those)[ \t]+"
    r"(?:[A-Za-z-]+[ \t]+){0,2})?"
    r"(?:envelopes?|jars?|bottles?|bags?|packages?|boxes?|containers?|lids?|"
    r"doors?|windows?|gaps?|cracks?|it|them)\b",
    re.IGNORECASE,
)
NOUN_DETERMINER = re.compile(
    r"\b(?:an?|the|this|that|these|those)[ \t]*$", re.IGNORECASE
)
MOUSE_DEVICE_BEFORE = re.compile(
    r"\b(?:(?:wireless|usb|bluetooth|optical|gaming|computer)[ \t]+"
    r"|(?:click(?:s|ed|ing)?|scroll(?:s|ed|ing)?)[ \t]+"
    r"(?:the[ \t]+(?:(?:left|right|middle)[ \t]+)?button[ \t]+)?"
    r"(?:on|with)[ \t]+(?:(?:the|my|your|his|her|our|their)[ \t]+)?)$",
    re.IGNORECASE,
)
MOUSE_DEVICE_AFTER = re.compile(
    r"[ \t]+(?:button|wheel|cursor|pointer|to[ \t]+(?:click|scroll))\b", re.IGNORECASE
)
SPORTS_BAT_BEFORE = re.compile(
    r"\b(?:(?P<sport>baseball|cricket)[ \t]+|(?:swung|swing(?:s|ing)?)[ \t]+"
    r"(?:(?:the|an?|my|your|his|her|its|our|their|this|that)[ \t]+"
    r"(?:[A-Za-z-]+[ \t]+){0,2})?)$",
    re.IGNORECASE,
)


def edit_prompt(source, index, word):
    return f"{source}\n{index + 1} [{word}]"


def edit_label(prompt):
    marker = prompt.rsplit("\n", 1)[-1].partition(" ")[2]
    return canonical(marker[1:-1])


def is_emoji_replacement(value):
    return bool(value and "".join(emojis_in(value)) == value)


def keep_seal_verb(source, start, end, word):
    return (
        canonical(word) == "seal"
        and SEAL_OBJECT.match(source, end) is not None
        and NOUN_DETERMINER.search(source, 0, start) is None
    )


def contextual_label(source, start, end, word):
    label = canonical(word)
    if label == "mouse" and (
        MOUSE_DEVICE_BEFORE.search(source, 0, start)
        or MOUSE_DEVICE_AFTER.match(source, end)
    ):
        return "computer mouse"
    if label == "bat" and (match := SPORTS_BAT_BEFORE.search(source, 0, start)):
        return f"{(match['sport'] or 'baseball').casefold()} bat"
    return word


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


def apply_edits(source, predict, *, label_for=None):
    pieces = []
    offset = 0
    for index, (start, end, word) in enumerate(bracket_words(source)):
        label = label_for(start, end, word) if label_for else word
        value = (
            predict(edit_prompt(source, index, label)) if label is not None else None
        )
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
