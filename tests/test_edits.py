import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


class EncodeEditTests(unittest.TestCase):
    def test_trained_alternatives_go_to_the_ranker_even_when_a_rule_matches(self):
        import torch

        from emojilm.config import Config
        from emojilm.inference import generate_edits
        from emojilm.model import EmojiLM
        from emojilm.tokenizer import Tokenizer

        source = "Please [seal] the envelope."
        tok = Tokenizer.train([source], ["🦭"])
        tok.encode_edits = True
        model = EmojiLM(
            Config(vocab=tok.size, ctx=64, layers=1, dim=16, heads=4, ffn=32)
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        # With uniform probabilities, the one-token emoji outscores the KEEP text.
        model.edit_candidates = {"seal": ["KEEP", "🦭"]}
        self.assertEqual(generate_edits(model, tok, source), "Please 🦭 the envelope.")

    def test_distinguishes_mouse_and_bat_from_their_animal_readings(self):
        from emojilm.inference import generate_edits

        model = SimpleNamespace(
            edit_candidates={
                "mouse": ["🐁"],
                "computer mouse": ["🖱"],
                "bat": ["🦇"],
            }
        )
        tokenizer = SimpleNamespace(encode_edits=True)
        for sentence, expected in (
            (
                "Click the left button on your [mouse].",
                "Click the left button on your 🖱.",
            ),
            (
                "Use a wireless [mouse], then draw a [mouse].",
                "Use a wireless 🖱, then draw a 🐁.",
            ),
            ("Click a photo of a [mouse].", "Click a photo of a 🐁."),
            (
                "She swung the [bat] and hit the ball.",
                "She swung the [bat] and hit the ball.",
            ),
            (
                "She swung at the [bat] with a broom.",
                "She swung at the 🦇 with a broom.",
            ),
            ("A [bat] flew out of the cave.", "A 🦇 flew out of the cave."),
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(generate_edits(model, tokenizer, sentence), expected)

    def test_keeps_seal_as_a_verb_and_encodes_the_animal(self):
        from emojilm.inference import generate_edits

        model = SimpleNamespace(edit_candidates={"seal": ["🦭"]})
        tokenizer = SimpleNamespace(encode_edits=True)
        for sentence, expected in (
            ("Please [seal] the envelope.", "Please [seal] the envelope."),
            (
                "We [seal] the padded envelope and saw a [seal] near the harbor.",
                "We [seal] the padded envelope and saw a 🦭 near the harbor.",
            ),
            ("They [SEAL] the jars tightly.", "They [SEAL] the jars tightly."),
            ("There was a [seal] on the beach.", "There was a 🦭 on the beach."),
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(generate_edits(model, tokenizer, sentence), expected)

    def test_bundles_training_readings_without_changing_model_weights(self):
        import torch

        from emojilm.config import Config
        from emojilm.edit_candidates import bundle_checkpoint
        from emojilm.edits import edit_rows
        from emojilm.inference import generate_edits, load
        from emojilm.model import EmojiLM
        from emojilm.tokenizer import Tokenizer

        row = {
            "task": "encode",
            "src": "Use [cat] and [friend].",
            "tgt": "Use 🐱 and [friend].",
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, bundled, data = (
                directory / name
                for name in ("original.pt", "ckpt/final/model.pt", "train.jsonl")
            )
            tok = Tokenizer.train([row["src"]], ["🐱"])
            tok.encode_edits = True
            cfg = Config(vocab=tok.size, ctx=32, layers=1, dim=16, heads=4, ffn=32)
            weights = EmojiLM(cfg).state_dict()
            torch.save(
                {
                    "format": "emojilm-1",
                    "cfg": cfg.to_dict(),
                    "tokenizer": tok.to_dict(),
                    "model": weights,
                },
                source,
            )
            readings = [
                {"task": "rebus", "src": "🐑", "tgt": "sheep <alt> ewe"},
                {"task": "rebus", "src": "🥧", "tgt": "pie"},
                {"task": "rebus", "src": "🐝🍃", "tgt": "believe"},
            ]
            data.write_text(
                "".join(
                    json.dumps(edit) + "\n" for edit in [*edit_rows(row), *readings]
                ),
                encoding="utf-8",
            )

            self.assertEqual(bundle_checkpoint(source, data, bundled), 4)
            self.assertNotIn("edit_candidates", torch.load(source, weights_only=True))
            model, tokenizer = load(bundled)
            self.assertTrue(
                all(
                    torch.equal(tensor, model.state_dict()[name])
                    for name, tensor in weights.items()
                )
            )
            self.assertEqual(
                generate_edits(
                    model, tokenizer, "  GPT [CAT]  [friend] and [unregistered].\n"
                ),
                "  GPT 🐱  [friend] and [unregistered].\n",
            )
            self.assertEqual(generate_edits(model, tokenizer, "I [eat]."), "I 🍽️.")
            decoded = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "emojilm.generate",
                    "--engine",
                    "model-edit",
                    "--ckpt",
                    str(bundled),
                    "--mode",
                    "decode",
                    "--text",
                    "  I 🐑 this 🥧. 🐝🍃! 🦄",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(decoded.returncode, 0, decoded.stderr)
            self.assertEqual(decoded.stdout, "  I sheep this pie. believe! 🦄\n")
            project = Path(__file__).resolve().parents[1]
            for mode, text, expected in (
                ("encode", "I [think] and [cat].", "I 🤔 and 🐱."),
                ("decode", "I 🐑 this 🥧.", "I sheep this pie."),
            ):
                with self.subTest(mode=mode):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "emojilm.generate",
                            "--mode",
                            mode,
                            "--text",
                            text,
                        ],
                        cwd=directory,
                        env={**os.environ, "PYTHONPATH": str(project)},
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, expected + "\n")

    def test_expands_rare_readings_and_uses_only_training_compounds(self):
        from emojilm.corpora import group_key, reserved_filter
        from emojilm.edit_coverage import coverage_rows

        rows = [
            {"task": "rebus", "src": "🍰", "tgt": "cake"},
            {"task": "rebus", "src": "🍓", "tgt": "strawberry"},
            {"task": "rebus", "src": "🌙", "tgt": "moon"},
            {"task": "rebus", "src": "🐧", "tgt": "penguin"},
            {
                "task": "encode",
                "src": "Bring [strawberry] [cake].",
                "tgt": "Bring 🍓🎂.",
            },
        ]
        reserved = reserved_filter(
            [
                {"task": "rebus", "src": "🌙🍰", "tgt": "mooncake"},
                {"task": "encode", "src": "Bring [moon] [cake].", "tgt": "Bring 🌙🍰."},
            ]
        )
        generated = list(
            coverage_rows(rows, [], {group_key("Draw [penguin] here.")}, reserved)
        )
        lexical = [r for r in generated if r["tag"] == "encode-edit-vocabulary"]
        self.assertEqual(sum(r["tgt"] == "🐧" for r in lexical), 6)
        self.assertFalse(
            any(r["src"].startswith("Draw [penguin] here.\n") for r in generated)
        )
        self.assertTrue(
            any(
                "[strawberry] [cake]" in r["src"] and r["tgt"] == "🎂"
                for r in generated
            )
        )
        self.assertTrue(
            any(
                "[strawberry] and [cake]" in r["src"] and r["tgt"] == "🍰"
                for r in generated
            )
        )
        self.assertFalse(any("[moon] [cake]" in r["src"] for r in generated))

    def test_copies_every_character_outside_individual_brackets(self):
        from emojilm.edits import apply_edits

        source = "  GPT [apple]  [pie]\t[unknown], [[nested]]!\n"
        answers = iter(["🍎", "🥧", "KEEP"])
        self.assertEqual(
            apply_edits(source, lambda _: next(answers)),
            "  GPT 🍎  🥧\t[unknown], [[nested]]!\n",
        )
        self.assertEqual(apply_edits(source, lambda _: "changed text 🥧"), source)

    def test_prepares_only_unambiguous_targets_that_preserve_literals(self):
        from emojilm.edits import edit_reference, edit_rows

        row = {
            "task": "encode",
            "src": "GPT [apple] [unknown].",
            "tgt": "GPT 🍎 [unknown].",
        }
        edits = list(edit_rows(row))
        self.assertEqual([edit["tgt"] for edit in edits], ["🍎", "KEEP"])
        self.assertIn("[unknown]", edits[1]["src"])
        self.assertFalse(list(edit_rows({**row, "tgt": "G 🍎 [unknown]."})))
        self.assertFalse(list(edit_rows({**row, "src": "[a][b]", "tgt": "🍎🥧🍒"})))
        compound = {**row, "src": "[apple]  [pie]", "tgt": "🍎🥧"}
        self.assertEqual([edit["tgt"] for edit in edit_rows(compound)], ["🍎", "🥧"])
        self.assertEqual(edit_reference(compound)["tgt"], "🍎  🥧")

    def test_edit_task_keeps_existing_token_ids(self):
        from emojilm.tokenizer import Tokenizer

        original = Tokenizer.train(["GPT [apple]"], ["🍎"])
        extended = Tokenizer.from_dict({**original.to_dict(), "encode_edits": True})
        source = "GPT [apple]"
        self.assertEqual(
            original.build("encode", source), extended.build("encode", source)
        )
        self.assertNotEqual(
            extended.build("encode", source), extended.build("encode-edit", source)
        )
        self.assertTrue(Tokenizer.from_dict(extended.to_dict()).encode_edits)
        with self.assertRaises(ValueError):
            original.build("encode-edit", source)
