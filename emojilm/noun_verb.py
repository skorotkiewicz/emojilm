"""Fit and evaluate a shared noun/verb classifier for marked book, watch and train."""

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from .corpora import group_key
from .edits import KEEP, edit_prompt
from .inference import generate_edits, load, span_features

GLYPHS = {"book": "📖", "watch": "⌚", "train": "🚆"}


def load_pairs():
    pairs = json.loads(Path(__file__).with_name("noun_verb_pairs.json").read_text())
    seen = set()
    for label, splits in pairs.items():
        for rows in splits.values():
            for noun, verb in rows:
                for source in (noun, verb):
                    key = group_key(source)
                    if key in seen or source.count(f"[{label}]") != 1:
                        raise ValueError(
                            f"Invalid or duplicate noun/verb prompt: {source}"
                        )
                    seen.add(key)
    return pairs


def evaluate(model, tok, pairs, split):
    results = {}
    for label, splits in pairs.items():
        counts = Counter()
        for noun, verb in splits[split]:
            expected = noun.replace(f"[{label}]", GLYPHS[label])
            noun_ok = generate_edits(model, tok, noun) == expected
            verb_ok = generate_edits(model, tok, verb) == verb
            counts.update(
                noun=noun_ok, verb=verb_ok, pairs=noun_ok and verb_ok, total=1
            )
        results[label] = counts
        print(
            f"{split} {label}: {counts['pairs']}/{counts['total']} pairs, "
            f"noun={counts['noun']} verb={counts['verb']}"
        )
    return results


@torch.inference_mode()
def fit(source, output, pairs):
    if source.resolve() == output.resolve():
        raise ValueError("Choose a separate output checkpoint")
    model, tok = load(source)
    if not model.edit_candidates or model.noun_verb_gate:
        raise ValueError("Expected a bundled checkpoint without a noun/verb classifier")
    baseline = evaluate(model, tok, pairs, "dev")

    features, targets = [], []
    for label, splits in pairs.items():
        for noun, verb in splits["train"]:
            for sentence, target in ((noun, 1), (verb, -1)):
                features.append(
                    span_features(model, tok, edit_prompt(sentence, 0, label))
                )
                targets.append(target)

    x = torch.stack(features)
    center = x.mean(0)
    x = F.normalize(x - center)
    dual = torch.linalg.solve(
        x @ x.T + 0.5 * torch.eye(len(x)), torch.tensor(targets).float()
    )
    weight = x.T @ dual
    gate = {"labels": list(pairs), "weight": weight, "bias": -(weight @ center)}
    model.noun_verb_gate = gate
    for label in pairs:
        model.edit_candidates[label] = sorted(
            set(model.edit_candidates.get(label, ())) | {KEEP, GLYPHS[label]}
        )
    fitted = evaluate(model, tok, pairs, "dev")
    gate["labels"] = [k for k in pairs if fitted[k]["pairs"] > baseline[k]["pairs"]]
    if not gate["labels"]:
        raise ValueError("No word improved on development pairs")

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["noun_verb_gate"] = gate
    for label in gate["labels"]:
        checkpoint["edit_candidates"][label] = model.edit_candidates[label]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / output.name
        torch.save(checkpoint, staged)
        staged.replace(output)
    print(
        f"Fitted {len(targets)} sentences; saved {', '.join(gate['labels'])} to {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt/sense/model.pt"))
    parser.add_argument("--out", type=Path, default=Path("ckpt/noun-verb/model.pt"))
    parser.add_argument(
        "--eval-only", action="store_true", help="Score --ckpt on held-out pairs"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    pairs = load_pairs()
    if args.eval_only:
        model, tok = load(args.ckpt)
        evaluate(model, tok, pairs, "holdout")
    else:
        fit(args.ckpt, args.out, pairs)


if __name__ == "__main__":
    main()
