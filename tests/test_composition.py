import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CompositionTests(unittest.TestCase):
    def test_composes_unlisted_pairs_and_preserves_literal_text(self):
        from emojilm.composition import Composer

        composer = Composer({"🏠": ["house"], "🐈": ["cat"], "🍃": ["leaf"]})
        self.assertEqual(composer.translate("rebus", "🏠🐈"), "housecat")
        self.assertEqual(
            composer.translate("decode", "  A 🏠🐈 prowled.\t🍃 🐈!  "),
            "  A housecat prowled.\tleaf cat!  ",
        )
        self.assertEqual(
            composer.translate("encode", "  A [house] [cat] prowled.\n"),
            "  A 🏠🐈 prowled.\n",
        )

    def test_unmapped_components_leave_their_entire_run_intact(self):
        from emojilm.composition import Composer

        composer = Composer({"🏠": ["house"], "🐈": ["cat"]})
        self.assertIsNone(composer.translate("decode", "A 🏠👻 prowled."))
        self.assertIsNone(composer.translate("encode", "Keep [ [cat] ] intact."))
        self.assertEqual(
            composer.translate("encode", "A [house] [ghost] hunted [cat]."),
            "A [house] [ghost] hunted 🐈.",
        )

    def test_auto_keeps_ordinary_emoji_lists_and_completes_mixed_bracket_input(self):
        commands = [
            (["--text", "I had 🍕🍺."], "I had 🍕🍺."),
            (
                [
                    "--mode",
                    "encode",
                    "--text",
                    "Eat [apple pie], then see a [house] [cat].",
                ],
                "Eat 🍎🥧, then see a 🏠🐱.",
            ),
        ]
        for options, expected in commands:
            with self.subTest(options=options):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "emojilm.generate",
                        "--ckpt",
                        "no-checkpoint.pt",
                        *options,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected + "\n")

    def test_composed_eval_is_independent_of_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"task": "rebus", "src": "🏠🐈", "tgt": "housecat"},
                        {
                            "task": "decode",
                            "src": "literal text",
                            "tgt": "literal text",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "emojilm.eval",
                    "--engine",
                    "compose",
                    "--ckpt",
                    "no-checkpoint.pt",
                    "--data",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("engine=compose", result.stdout)
        self.assertIn("rebus exact 1/1", result.stdout)
        self.assertIn("decode exact 1/1", result.stdout)
