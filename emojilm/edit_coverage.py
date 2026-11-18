"""Supplement bracket-edit training with readings from the training split."""

import argparse
from collections import Counter, defaultdict
from itertools import chain
import json
from pathlib import Path
import shutil
import tempfile

from .composition import bracket_words
from .config import Config
from .corpora import canonical, group_key, reserved_filter
from .data import load_jsonl
from .edits import EDIT_TASK, KEEP, aligned_replacements, edit_label, edit_rows
from .story_data import known_mappings
from .tokenizer import Tokenizer

READING_EXAMPLES = 6
VOCABULARY_FRAMES = (
    "Draw {} here.",
    "The next card shows {}.",
    "In my notes, I put {} below the heading.",
    "Use {} in the caption, GPT.",
    "She wrote {} beside the title and left the rest of the page blank.",
    "On the corner of the card, someone marked {} in blue ink.",
    "They circled {} before handing the page back to the teacher.",
    "Please put {} on this line, then leave a space for the next label.",
)
PAIR_FRAMES = (
    "The picture had {} near the edge.",
    "At the top of the page, Mia drew {}.",
    "Someone added {} below the title in the final draft.",
    "I copied {} onto the card before bringing it to class.",
    "The next scene showed {} beside the road.",
    "On the way home, I made a sketch of {} in my notebook.",
)
EVALUATIONS = (
    "holdout",
    "composition-val",
    "context-val",
    "long-val",
    "edit-val",
    "edit-holdout",
)


def contextual_pairs(rows, mappings):
    observed = defaultdict(set)
    for row in rows:
        if row["task"] != "encode":
            continue
        source = row["src"]
        spans = list(bracket_words(source))
        if len(spans) < 2:
            continue
        replacements = aligned_replacements(
            source, row["tgt"], spans, allow_joined=True
        )
        if replacements is None:
            continue
        for index, (left, right) in enumerate(zip(spans, spans[1:])):
            words = canonical(left[2]), canonical(right[2])
            glyphs = replacements[index : index + 2]
            if source[left[1] : right[0]].strip() or KEEP in glyphs:
                continue
            if all(word in mappings for word in words):
                observed[words].add(glyphs)
    for words, choices in sorted(observed.items()):
        if len(choices) != 1:
            continue
        glyphs = next(iter(choices))
        standalone = tuple(mappings[word][0] for word in words)
        if tuple(map(canonical, glyphs)) != tuple(map(canonical, standalone)):
            yield words, glyphs, standalone


def coverage_rows(rows, edits, excluded, reserved):
    mappings = known_mappings(rows)
    counts = Counter()
    known_labels, kept_labels = set(), set()
    seen = {group_key(edit["src"]) for edit in edits}
    for edit in edits:
        label = edit_label(edit["src"])
        counts[label, canonical(edit["tgt"])] += 1
        (kept_labels if edit["tgt"] == KEEP else known_labels).add(label)

    def focused(source, target, tag):
        example = {"task": "encode", "src": source, "tgt": target}
        if (
            group_key(source) in excluded
            or group_key(target) in excluded
            or reserved(example)
        ):
            return
        for edit in edit_rows(example):
            key = group_key(edit["src"])
            if key in seen or key in excluded:
                continue
            seen.add(key)
            yield {**edit, "tag": tag}

    for word, (glyph, _) in sorted(mappings.items()):
        missing = READING_EXAMPLES - counts[word, canonical(glyph)]
        for frame in VOCABULARY_FRAMES:
            if missing <= 0:
                break
            generated = list(
                focused(
                    frame.format(f"[{word}]"),
                    frame.format(glyph),
                    "encode-edit-vocabulary",
                )
            )
            missing -= len(generated)
            yield from generated

    # Give KEEP examples the same style of context as the new readings.
    for index, word in enumerate(sorted(kept_labels - known_labels - mappings.keys())):
        source = VOCABULARY_FRAMES[index % len(VOCABULARY_FRAMES)].format(f"[{word}]")
        yield from focused(source, source, "encode-edit-vocabulary")

    for words, glyphs, standalone in contextual_pairs(rows, mappings):
        markers = [f"[{word}]" for word in words]
        for frame in PAIR_FRAMES:
            for separator, labels in ((" ", glyphs), (" and ", standalone)):
                yield from focused(
                    frame.format(separator.join(markers)),
                    frame.format(separator.join(labels)),
                    "encode-edit-context",
                )


def build_coverage(base, output):
    if base.resolve() == output.resolve():
        raise ValueError("Choose an output directory separate from the base dataset")
    if not (base / "train.jsonl").is_file():
        raise ValueError(
            f"Missing {base / 'train.jsonl'}. Run just data to prepare the default dataset."
        )
    rows = load_jsonl(base / "train.jsonl")
    edits = [row for row in rows if row["task"] == EDIT_TASK]
    tokenizer = Tokenizer.load(base / "tokenizer.json")
    if not edits or not tokenizer.encode_edits:
        raise ValueError(
            f"{base} needs bracket-edit training examples and their tokenizer"
        )
    evaluation = {
        name: load_jsonl(base / f"{name}.jsonl") for name in (*EVALUATIONS, "val")
    }
    reserved = reserved_filter(evaluation["holdout"] + evaluation["composition-val"])
    excluded = {
        group_key(row[field])
        for references in evaluation.values()
        for row in references
        for field in ("src", "tgt")
    }
    ctx = Config().ctx
    additions = [
        row
        for row in coverage_rows(rows, edits, excluded, reserved)
        if len(tokenizer.build(EDIT_TASK, row["src"], row["tgt"])[0]) - 1 <= ctx
    ]

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as directory:
        staged = Path(directory)
        for filename in (
            "tokenizer.json",
            *(f"{name}.jsonl" for name in (*EVALUATIONS, "val")),
        ):
            shutil.copyfile(base / filename, staged / filename)
        with (staged / "train.jsonl").open("w", encoding="utf-8") as stream:
            for row in chain(rows, additions):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for path in staged.iterdir():
            path.replace(output / path.name)
    return Counter(row["tag"] for row in additions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/encode-edits"))
    parser.add_argument("--out", type=Path, default=Path("data/encode-edits-expanded"))
    args = parser.parse_args()
    try:
        counts = build_coverage(args.base, args.out)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Added encode edits: {dict(counts)} -> {args.out}")


if __name__ == "__main__":
    main()
