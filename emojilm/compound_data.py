"""Component supervision for the separate direct-generation experiment."""

from collections import Counter
import json
from pathlib import Path
import random

PAIR_FILE = Path(__file__).resolve().parents[1] / "data" / "compound_pairs.json"
TRAIN_FRAMES = (
    "I found a {} by the window.",
    "Do you see the {}?",
    "They packed the {} in a hurry.",
    "Look for the {} on your left.",
    'The label said "{}" in big letters.',
    "  Entry: {}.  ",
    "Before dusk, we passed the {} again.",
    "She drew a {} on the back of the card.",
)
DEV_FRAMES = (
    "Look closely at the {} near the edge.",
    "Outside, I noticed a {} beside the path.",
    "Is that the {}?",
    "At first I missed the {}!",
    'It was labeled "{}".',
    "Under the shelf: {}.",
    "The {} surprised me.",
    "What happened to the {} after lunch?",
)


def split_pairs(pairs, seed, dev_size=12):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    counts = Counter(part for pair in pairs for part in pair)
    train, dev = [], []
    for pair in pairs:
        if len(dev) < dev_size and all(counts[part] > 1 for part in pair):
            dev.append(pair)
            counts.subtract(pair)
        else:
            train.append(pair)
    if len(dev) != dev_size:
        raise ValueError("The compound bank has too few pairs with shared components")
    return train, dev


def pair_examples(pair, frames, *, training):
    (left_emoji, left), (right_emoji, right) = pair
    emoji = left_emoji + right_emoji
    word = left + right
    brackets = f"[{left}] [{right}]"

    def row(task, source, target, target_parts=None):
        result = {"task": task, "src": source, "tgt": target, "tag": "composition"}
        if training and target_parts:
            result["target_parts"] = target_parts
        return result

    yield row("rebus", emoji, word, [left, right])
    for frame in frames:
        prefix, suffix = frame.split("{}")
        yield row(
            "decode",
            frame.format(emoji),
            frame.format(word),
            [prefix, left, right, suffix],
        )
        yield row("encode", frame.format(brackets), frame.format(emoji))
    if training:
        separated_emoji = f"{left_emoji} {right_emoji}"
        separated_words = f"{left} {right}"
        yield row("rebus", separated_emoji, separated_words, [left, " ", right])
        for frame in frames[:3]:
            prefix, suffix = frame.split("{}")
            yield row(
                "decode",
                frame.format(separated_emoji),
                frame.format(separated_words),
                [prefix, left, " ", right, suffix],
            )
        yield row("encode", f"[{word}]", emoji)


def composition_examples(seed, reserved):
    raw_pairs = json.loads(PAIR_FILE.read_text(encoding="utf-8"))
    pairs = [tuple((emoji, word) for emoji, word in pair) for pair in raw_pairs]
    eligible = [
        pair
        for pair in pairs
        if not any(
            reserved(row) for row in pair_examples(pair, TRAIN_FRAMES, training=True)
        )
    ]
    training_pairs, dev_pairs = split_pairs(eligible, seed)
    train = [
        row
        for pair in training_pairs
        for row in pair_examples(pair, TRAIN_FRAMES, training=True)
    ]
    # The development combinations are unseen. Their individual readings remain learnable.
    for emoji, word in sorted({part for pair in eligible for part in pair}):
        for task, source, target in (
            ("rebus", emoji, word),
            ("decode", f"The symbol is {emoji}.", f"The symbol is {word}."),
            ("encode", f"[{word}]", emoji),
        ):
            train.append(
                {"task": task, "src": source, "tgt": target, "tag": "composition"}
            )
    dev = [
        row
        for i, pair in enumerate(dev_pairs)
        for row in pair_examples(
            pair,
            (
                DEV_FRAMES[i % len(DEV_FRAMES)],
                DEV_FRAMES[(i + 3) % len(DEV_FRAMES)],
            ),
            training=False,
        )
    ]
    return train, dev
