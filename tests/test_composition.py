import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class CompositionTests(unittest.TestCase):
    def test_preserves_known_compounds_inside_a_longer_emoji_run(self):
        from emojilm.composition import Composer

        composer = Composer(
            {
                "🐝": ["bee"],
                "🍃": ["leaf"],
                "🍎": ["apple"],
                "🥧": ["pie"],
                "🐝🍃": ["believe"],
                "🍎🥧": ["apple pie"],
            },
            prefer_known_runs=True,
        )
        self.assertEqual(
            composer.translate("decode", "  The card says 🐝🍃🍎🥧.\t"),
            "  The card says believe apple pie.\t",
        )
        self.assertEqual(
            composer.translate("decode", "The card says 🐝🍃🛸."),
            "The card says believe 🛸.",
        )
        self.assertEqual(composer.translate("decode", "🍃🥧"), "leafpie")
        self.assertEqual(composer.translate("decode", "🐝 🍃"), "bee leaf")

    def test_model_encode_can_confirm_a_secondary_reading_on_a_lower_ranked_glyph(self):
        from emojilm.composition import Composer
        from emojilm.inference import NeuralComposer

        composer = NeuralComposer.__new__(NeuralComposer)
        Composer.__init__(composer, {}, partial_encode=True)
        with (
            patch.object(
                NeuralComposer,
                "rank_glyphs",
                return_value=[("🌡️", -5.46), ("🖌️", -6.40), ("❤️", -6.56)],
            ),
            patch.object(
                NeuralComposer,
                "component_readings",
                side_effect={
                    "🌡️": ["thermometer"],
                    "🖌️": ["paintbrush"],
                    "❤️": ["love", "heart"],
                }.__getitem__,
            ),
        ):
            self.assertEqual(composer.translate("encode", "GPT [heart]"), "GPT ❤️")

    def test_model_encode_uses_a_clear_ranking_and_preserves_unrecognized_brackets(
        self,
    ):
        from emojilm.inference import NeuralComposer
        from emojilm.composition import Composer

        composer = NeuralComposer.__new__(NeuralComposer)
        Composer.__init__(composer, {}, partial_encode=True)
        ranked = {
            "apple": [("🍎", math.log(0.09)), ("🐂", math.log(0.0002))],
            "pie": [("🥧", math.log(0.9))],
            "sweet": [("📞", math.log(0.000002)), ("🚋", math.log(0.000001))],
            "friend": [],
        }
        readings = {
            "🍎": ["appleglie"],
            "🥧": ["pie"],
            "📞": ["telephone"],
            "🚋": ["tram"],
        }
        with (
            patch.object(NeuralComposer, "rank_glyphs", side_effect=ranked.__getitem__),
            patch.object(
                NeuralComposer, "component_readings", side_effect=readings.__getitem__
            ),
        ):
            self.assertEqual(
                composer.translate(
                    "encode", "The [sweet]  [apple] [pie]\t[friend] arrived."
                ),
                "The [sweet]  🍎🥧\t[friend] arrived.",
            )

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
            (
                [
                    "--mode",
                    "encode",
                    "--text",
                    "I [love] you and [apple] with [sweet] [pie] [friend].",
                ],
                "I ❤️ you and 🍎 with [sweet] 🥧 [friend].",
            ),
            (
                [
                    "--mode",
                    "encode",
                    "--text",
                    "Keep apple pie and [friend] beside [apple].",
                ],
                "Keep apple pie and [friend] beside 🍎.",
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
