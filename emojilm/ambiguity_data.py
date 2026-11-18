"""Build an isolated bracket-edit experiment with contrasting word senses."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from .config import Config
from .corpora import group_key
from .data import load_jsonl
from .edit_coverage import EVALUATIONS
from .edits import EDIT_TASK, KEEP, edit_label, edit_rows
from .tokenizer import Tokenizer


def build_ambiguity(base, output):
    if base.resolve() == output.resolve():
        raise ValueError("Choose an output directory separate from the base dataset")
    if not (base / "train.jsonl").is_file():
        raise ValueError(f"Missing {base / 'train.jsonl'}. Run just data first.")

    examples = json.loads(
        Path(__file__).with_name("ambiguity_examples.json").read_text(encoding="utf-8")
    )
    tokenizer = Tokenizer.load(base / "tokenizer.json")
    if not tokenizer.encode_edits:
        raise ValueError(f"{base} needs a bracket-edit tokenizer")
    ctx = Config().ctx

    train = load_jsonl(base / "train.jsonl")
    seen = set()
    for name in ("train", "val", *EVALUATIONS):
        rows = train if name == "train" else load_jsonl(base / f"{name}.jsonl")
        for row in rows:
            source = row["src"]
            if row["task"] == EDIT_TASK:
                source = source.rsplit("\n", 1)[0]
            seen.add(group_key(source))

    splits = {"train": [], "dev": [], "holdout": []}
    for label, senses in examples.items():
        marker = f"[{label}]"
        for replacement, sentences in senses.items():
            for split, sources in sentences.items():
                for source in sources:
                    key = group_key(source)
                    if key in seen:
                        raise ValueError(f"Duplicate ambiguity prompt: {source}")
                    seen.add(key)
                    if source.count(marker) != 1:
                        raise ValueError(f"Expected one {marker} marker: {source}")
                    target = (
                        source
                        if replacement == KEEP
                        else source.replace(marker, replacement)
                    )
                    row = {
                        "task": "encode",
                        "src": source,
                        "tgt": target,
                        "tag": "ambiguity",
                    }
                    edits = list(edit_rows(row))
                    if len(edits) != 1:
                        raise ValueError(
                            f"Expected one aligned bracket replacement: {source}"
                        )
                    edit = edits[0]
                    ids, _ = tokenizer.build(EDIT_TASK, edit["src"], edit["tgt"])
                    if len(ids) - 1 > ctx:
                        raise ValueError(
                            f"Ambiguity prompt exceeds the training context: {source}"
                        )
                    splits[split].append(
                        {**edit, "tag": "ambiguity"} if split == "train" else row
                    )

    # Replace the animal-only edit rows for these labels with balanced examples.
    train = [
        row
        for row in train
        if row["task"] != EDIT_TASK or edit_label(row["src"]) not in examples
    ]
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as directory:
        staged = Path(directory)
        for filename in ("tokenizer.json", "val.jsonl"):
            shutil.copyfile(base / filename, staged / filename)
        for filename, rows in (
            ("train.jsonl", train + splits["train"]),
            ("ambiguity-dev.jsonl", splits["dev"]),
            ("ambiguity-holdout.jsonl", splits["holdout"]),
        ):
            with (staged / filename).open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for path in staged.iterdir():
            path.replace(output / path.name)
    return {split: len(rows) for split, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/encode-edits-expanded"))
    parser.add_argument("--out", type=Path, default=Path("data/ambiguity"))
    args = parser.parse_args()
    try:
        counts = build_ambiguity(args.base, args.out)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Ambiguity examples {counts} -> {args.out}")


if __name__ == "__main__":
    main()
