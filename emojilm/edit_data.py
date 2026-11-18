"""Add replacement-only encode examples to a prepared dataset."""

import argparse
from itertools import chain
import json
from pathlib import Path
import shutil
import tempfile

from .config import Config
from .data import load_jsonl
from .edits import edit_reference, edit_rows
from .tokenizer import Tokenizer


def build_edit_data(base, output):
    if base.resolve() == output.resolve():
        raise ValueError("Choose an output directory separate from the base dataset")

    tokenizer = Tokenizer.load(base / "tokenizer.json")
    tokenizer.encode_edits = True
    ctx = Config().ctx
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    with tempfile.TemporaryDirectory(dir=output) as directory:
        staged = Path(directory)
        tokenizer.save(staged / "tokenizer.json")
        for name in ("train", "val"):
            rows = load_jsonl(base / f"{name}.jsonl")
            additions, seen = [], set()
            for row in rows:
                for edit in edit_rows(row):
                    key = edit["src"], edit["tgt"]
                    if (
                        key in seen
                        or len(tokenizer.build(edit["task"], *key)[0]) - 1 > ctx
                    ):
                        continue
                    seen.add(key)
                    additions.append(edit)
            if not additions:
                raise ValueError(f"No bracket replacements found in the {name} split")
            counts[name] = len(additions)
            with (staged / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for row in chain(rows, additions):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for name in ("holdout", "composition-val", "context-val", "long-val"):
            shutil.copyfile(base / f"{name}.jsonl", staged / f"{name}.jsonl")
        for source, name in (
            ("composition-val", "edit-val"),
            ("holdout", "edit-holdout"),
        ):
            with (staged / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for row in load_jsonl(base / f"{source}.jsonl"):
                    stream.write(
                        json.dumps(edit_reference(row), ensure_ascii=False) + "\n"
                    )
        for path in staged.iterdir():
            path.replace(output / path.name)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/tinystories-long"))
    parser.add_argument("--out", type=Path, default=Path("data/encode-edits"))
    args = parser.parse_args()
    try:
        counts = build_edit_data(args.base, args.out)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Encode edits train={counts['train']} val={counts['val']} -> {args.out}")


if __name__ == "__main__":
    main()
