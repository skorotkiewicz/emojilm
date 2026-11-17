"""Translate text with curated rules, a trained checkpoint, or automatic selection."""

import argparse
from pathlib import Path
import sys

from .composition import default_composer
from .data import emojis_in
from .rules import LEXICON, apply_rules, decode_preserves_text, load_lexicon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default="decode", choices=["decode", "encode", "rebus", "emojify"]
    )
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--engine",
        choices=["auto", "rules", "compose", "model", "model-compose"],
        default="auto",
    )
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt/model.pt"))
    parser.add_argument("--lexicon", type=Path, default=LEXICON)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--hints", nargs="?", const=True, help="Use the lexicon or supply a hint string"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.engine in {"rules", "compose", "model-compose"} and args.mode == "emojify":
        parser.error("emojify uses the model or auto engine")
    if args.engine == "model-compose" and args.hints is not None:
        parser.error("model-compose uses only the checkpoint's component readings")
    symbols = emojis_in(args.text) if args.mode == "decode" else []
    try:
        if args.engine in {"auto", "rules"} and args.mode != "emojify":
            result = apply_rules(args.mode, args.text, load_lexicon(args.lexicon))
            if result is not None or args.engine == "rules":
                engine = "rules"
                if (
                    args.engine == "auto"
                    and args.mode == "encode"
                    and result is not None
                ):
                    composed = default_composer().translate("encode", result)
                    if composed is not None:
                        result = composed
                        engine = "rules+compose"
                print(f"engine={engine}", file=sys.stderr)
                print(args.text if result is None else result)
                return
            if args.mode == "decode" and not symbols:
                print("engine=copy", file=sys.stderr)
                print(args.text)
                return
        if args.engine == "compose" or (
            args.engine == "auto"
            and args.mode in {"encode", "rebus"}
            and args.hints is None
        ):
            result = default_composer().translate(
                args.mode, args.text, automatic=args.engine == "auto"
            )
            if result is not None or args.engine == "compose":
                print("engine=compose", file=sys.stderr)
                print(args.text if result is None else result)
                return
        if args.engine == "auto" and args.mode != "emojify":
            if not args.ckpt.is_file():
                if args.mode in {"decode", "encode"}:
                    print("engine=rules (no match)", file=sys.stderr)
                    print(args.text)
                    return
                raise ValueError(
                    "No curated rebus match. Train a model to generate a reading."
                )
        if not args.ckpt.is_file():
            raise ValueError(
                f"Checkpoint not found at {args.ckpt}. Run training first."
            )

        from .inference import NeuralComposer, generate, load

        model, tok = load(args.ckpt, args.device)
        if args.engine == "model-compose":
            composer = NeuralComposer(model, tok, args.max_new_tokens)
            result = composer.translate(args.mode, args.text)
            print("engine=model-compose", file=sys.stderr)
            print(args.text if result is None else result)
            return

        hint = args.hints
        if hint is True:
            from .corpora import hint_dictionary, hint_for

            hint = hint_for(args.text, hint_dictionary())
        answers = generate(model, tok, args.mode, args.text, hint, args.max_new_tokens)
        if args.engine == "auto" and args.mode == "decode":
            answers = [
                answer
                for answer in answers
                if decode_preserves_text(args.text, answer, symbols)
            ]
            if not answers:
                print("engine=copy (model changed surrounding text)", file=sys.stderr)
                print(args.text)
                return
        print("engine=model", file=sys.stderr)
        print("\n".join(answers))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
