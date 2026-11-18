"""Measure exact matches on the compositional holdout."""

import argparse
from collections import Counter
from pathlib import Path

from .composition import bracket_runs, compact_emoji_runs, default_composer
from .corpora import hint_dictionary, hint_for
from .data import load_jsonl, matches_reference
from .rules import preserves_literals


def preserves_surrounding_text(task, source, answer):
    spans = bracket_runs(source) if task == "encode" else compact_emoji_runs(source)
    literals = []
    offset = 0
    for start, end, _ in spans:
        literals.append(source[offset:start])
        offset = end
    literals.append(source[offset:])
    return preserves_literals(literals, answer)


def report(rows, predict, engine, *, hints=False, copy_metrics=False):
    correct, total, skipped, copied = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        task = row["task"]
        total[task] += 1
        answers = predict(row)
        if answers is None:
            skipped[task] += 1
            continue
        correct[task] += matches_reference(row, answers)
        if copy_metrics and task in {"encode", "decode"}:
            copied[task] += any(
                preserves_surrounding_text(task, row["src"], answer)
                for answer in answers
            )
    print(f"engine={engine} hints={hints}")
    for task in sorted(total):
        metrics = (
            f"{task} exact {correct[task]}/{total[task]} over_context={skipped[task]}"
        )
        if copy_metrics and task in {"encode", "decode"}:
            metrics += f" preserved_text={copied[task]}/{total[task]}"
        print(metrics)
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
        report(rows, predict, "model", hints=args.hints, copy_metrics=args.copy_metrics)
    if args.engine != "model":
        composer = NeuralComposer(model, tok, args.max_new_tokens)
        evaluate_composition(rows, composer, "model-compose", args.copy_metrics)


def evaluate_composition(rows, composer, engine, copy_metrics):
    supported = [row for row in rows if row["task"] in {"decode", "encode", "rebus"}]
    if not supported:
        print(f"engine={engine} no supported tasks in evaluation file")
        return

    def predict(row):
        answer = composer.translate(row["task"], row["src"])
        return [answer if answer is not None else row["src"]]

    report(supported, predict, engine, copy_metrics=copy_metrics)


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
    parser.add_argument(
        "--copy-metrics",
        action="store_true",
        help="Report preservation of surrounding text",
    )
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
            evaluate_composition(rows, default_composer(), "compose", args.copy_metrics)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
