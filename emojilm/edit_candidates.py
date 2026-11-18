"""Bundle supported bracket replacements into an existing model-edit checkpoint."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import tempfile

import torch

from .data import load_jsonl
from .corpora import canonical
from .edits import EDIT_ALIASES, EDIT_TASK, KEEP, edit_label, is_emoji_replacement
from .tokenizer import Tokenizer


def edit_bundle_metadata(rows):
    readings = defaultdict(set)
    rebuses = defaultdict(Counter)
    for row in rows:
        if row["task"] == "rebus" and is_emoji_replacement(row["src"]):
            primary = row["tgt"].split(" <alt> ", 1)[0].strip()
            if primary:
                rebuses[canonical(row["src"])][primary] += 1
        if row["task"] != EDIT_TASK:
            continue
        label = edit_label(row["src"])
        target = row["tgt"]
        if label and (target == KEEP or is_emoji_replacement(target)):
            readings[label].add(target)
    if not readings:
        raise ValueError("The training split contains no bracket replacements")
    for label, (glyph, reading) in EDIT_ALIASES.items():
        readings.setdefault(label, {glyph})
        rebuses.setdefault(canonical(glyph), Counter({reading: 1}))

    return {
        "edit_candidates": {
            label: sorted(choices) for label, choices in sorted(readings.items())
        },
        "decode_readings": {
            glyph: [counts.most_common(1)[0][0]]
            for glyph, counts in sorted(rebuses.items())
        },
    }


def bundle_checkpoint(source, data, output):
    if source.resolve() == output.resolve():
        raise ValueError("Choose a separate path for the bundled checkpoint")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format") != "emojilm-1"
        or not Tokenizer.from_dict(checkpoint["tokenizer"]).encode_edits
    ):
        raise ValueError("Use a checkpoint trained with the encode edits dataset")

    checkpoint.update(edit_bundle_metadata(load_jsonl(data)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        snapshot = Path(directory) / output.name
        torch.save(checkpoint, snapshot)
        snapshot.replace(output)
    return len(checkpoint["edit_candidates"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=Path, default=Path("ckpt/encode-edits-expanded/model.pt")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/encode-edits-expanded/train.jsonl")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("ckpt/encode-edits-bundled/model.pt")
    )
    args = parser.parse_args()
    try:
        count = bundle_checkpoint(args.ckpt, args.data, args.out)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Bundled replacements for {count} labels -> {args.out}")


if __name__ == "__main__":
    main()
