"""Measure exact matches on the compositional holdout."""

import argparse
from collections import Counter
from pathlib import Path

from .composition import default_composer
from .corpora import hint_dictionary, hint_for
from .data import load_jsonl


def report(rows, predict, engine, *, hints=False):
    correct, total, skipped = Counter(), Counter(), Counter()
    for row in rows:
        task, target = row["task"], row["tgt"]
        total[task] += 1
        answers = predict(row)
        if answers is None:
            skipped[task] += 1
            continue
        expected = target.split(" <alt> ") if task == "rebus" else [target]
        correct[task] += bool(
            {answer.strip() for answer in answers}
            & {answer.strip() for answer in expected}
        )
    print(f"engine={engine} hints={hints}")
    for task in sorted(total):
        print(
            f"{task} exact {correct[task]}/{total[task]} over_context={skipped[task]}"
        )
    print(f"total exact {sum(correct.values())}/{sum(total.values())}")


def evaluate_model(rows, args):
    import torch

    from .inference import NeuralComposer, generate, load

    if args.threads:
        torch.set_num_threads(args.threads)
    model, tok = load(args.ckpt, args.device)
    lexicon = hint_dictionary() if args.hints else {}

    def predict(row):
        task, source = row["task"], row["src"]
        hint = hint_for(source, lexicon) if args.hints else None
        prompt, _ = tok.build(task, source, hint=hint)
        if len(prompt) > model.cfg.ctx:
            return None
        return generate(model, tok, task, source, hint, args.max_new_tokens)

    if args.engine != "model-compose":
        report(rows, predict, "model", hints=args.hints)
    if args.engine != "model":
        composer = NeuralComposer(model, tok, args.max_new_tokens)
        evaluate_composition(rows, composer, "model-compose")


def evaluate_composition(rows, composer, engine):
    supported = [row for row in rows if row["task"] in {"decode", "encode", "rebus"}]
    if not supported:
        print(f"engine={engine} no supported tasks in evaluation file")
        return

    def predict(row):
        answer = composer.translate(row["task"], row["src"])
        return [answer if answer is not None else row["src"]]

    report(supported, predict, engine)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt/model.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/holdout.jsonl"))
    parser.add_argument(
        "--engine", choices=["all", "model", "model-compose", "compose"], default="all"
    )
    parser.add_argument(
        "--hints", action="store_true", help="Supply hints to direct model generation"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 0:
        parser.error("max-new-tokens must be positive and threads non-negative")
    if args.hints and args.engine in {"compose", "model-compose"}:
        parser.error("--hints applies to direct model generation")
    try:
        rows = load_jsonl(args.data)
        if not rows:
            parser.error("Evaluation file is empty")
        if args.engine != "compose":
            evaluate_model(rows, args)
        if args.engine in {"all", "compose"}:
            evaluate_composition(rows, default_composer(), "compose")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
