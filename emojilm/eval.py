"""Measure exact matches on the compositional holdout with neural inference."""

import argparse
from collections import Counter
from pathlib import Path

import torch

from .corpora import hint_dictionary, hint_for
from .inference import generate, load
from .train import load_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt/model.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/holdout.jsonl"))
    parser.add_argument("--hints", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 0:
        parser.error("max-new-tokens must be positive and threads non-negative")
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        rows = load_jsonl(args.data)
        model, tok = load(args.ckpt, args.device)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not rows:
        parser.error("Evaluation file is empty")

    lexicon = hint_dictionary() if args.hints else {}
    correct, total, skipped = Counter(), Counter(), Counter()
    for row in rows:
        task, source, target = row["task"], row["src"], row["tgt"]
        hint = hint_for(source, lexicon) if args.hints else None
        total[task] += 1
        prompt, _ = tok.build(task, source, hint=hint)
        if len(prompt) > model.cfg.ctx:
            skipped[task] += 1
            continue
        answers = generate(model, tok, task, source, hint, args.max_new_tokens)
        expected = target.split(" <alt> ") if task == "rebus" else [target]
        correct[task] += bool(
            {answer.strip() for answer in answers}
            & {answer.strip() for answer in expected}
        )
    print(f"engine=model hints={args.hints}")
    for task in sorted(total):
        print(
            f"{task} exact {correct[task]}/{total[task]} over_context={skipped[task]}"
        )


if __name__ == "__main__":
    main()
