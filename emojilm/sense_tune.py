"""Fit word-sense classifiers on frozen EmojiLM features for bundled encode edits."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from .corpora import group_key
from .data import load_jsonl, matches_reference
from .edits import EDIT_TASK, edit_label, edit_prompt
from .inference import generate_edits, load


@torch.inference_mode()
def fit(source, data, output):
    if output.resolve() == source.resolve():
        raise ValueError("Choose a separate output checkpoint")
    model, tok = load(source)
    if not model.edit_candidates or model.sense_heads:
        raise ValueError("Use a bundled checkpoint without fitted sense classifiers")
    train = defaultdict(list)
    for row in load_jsonl(data / "train.jsonl"):
        if row["task"] == EDIT_TASK and row.get("tag") == "ambiguity":
            train[edit_label(row["src"])].append((row["src"], row["tgt"]))
    pairs = json.loads(Path(__file__).with_name("sense_pairs.json").read_text())
    dev = load_jsonl(data / "ambiguity-dev.jsonl")
    reserved = {
        group_key(row["src"])
        for row in dev + load_jsonl(data / "ambiguity-holdout.jsonl")
    }
    seen = set()
    heads = {}
    for label, senses in pairs.items():
        choices = model.edit_candidates.get(label, [])
        if set(senses) != set(choices) or not train[label]:
            raise ValueError(
                f"Missing training examples or bundled candidates for {label}"
            )
        for value, sentences in senses.items():
            for sentence in sentences:
                if sentence.count(f"[{label}]") != 1:
                    raise ValueError(f"Expected one [{label}] in {sentence}")
                train[label].append((edit_prompt(sentence, 0, label), value))

        features, targets = [], []
        for prompt, value in train[label]:
            key = group_key(prompt.rsplit("\n", 1)[0])
            if key in reserved or key in seen:
                raise ValueError(f"Overlapping sense example: {prompt}")
            seen.add(key)
            ids, _ = tok.build(EDIT_TASK, prompt)
            if len(ids) > model.cfg.ctx:
                raise ValueError("Sense example exceeds the model context")
            features.append(model.hidden(torch.tensor([ids]))[0, -1])
            targets.append(choices.index(value))

        x = torch.stack(features)
        center = x.mean(0)
        x = F.normalize(x - center)
        y = F.one_hot(torch.tensor(targets), len(choices)).float()
        # Dual ridge regression is cheaper than a feature-sized solve on this small dataset.
        dual = torch.linalg.solve(x @ x.T + 0.5 * torch.eye(len(x)), y)
        weight = (x.T @ dual).T.contiguous()
        bias = -(weight @ center)
        rows = [row for row in dev if f"[{label}]" in row["src"]]

        def exact():
            return sum(
                matches_reference(row, [generate_edits(model, tok, row["src"])])
                for row in rows
            )

        baseline = exact()
        model.sense_heads[label] = (choices, weight, bias)
        fitted = exact()
        print(
            f"{label}: train={len(x)} dev={baseline}/{len(rows)} -> {fitted}/{len(rows)}"
        )
        if fitted > baseline:
            heads[label] = {"choices": choices, "weight": weight, "bias": bias}
        else:
            del model.sense_heads[label]

    if not heads:
        raise ValueError("No sense classifier improved on development examples")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["sense_heads"] = heads
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / output.name
        torch.save(checkpoint, staged)
        staged.replace(output)
    print(f"Saved fitted classifiers for {', '.join(heads)} to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt/ambiguity/model.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/ambiguity"))
    parser.add_argument("--out", type=Path, default=Path("ckpt/sense/model.pt"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    fit(args.ckpt, args.data, args.out)


if __name__ == "__main__":
    main()
