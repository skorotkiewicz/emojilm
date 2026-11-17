import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from emojilm import corpora
from emojilm.data import emojis_in
from emojilm.rules import apply_rules, decode_preserves_text, load_lexicon
from emojilm.tokenizer import Tokenizer


class PipelineTests(unittest.TestCase):
    def test_tokenizer_preserves_copy_text_and_encodes_rebus_alternatives(self):
        sample = "  hello\t🍑⛳\n▁ 🧑🏽‍🚀 "
        tok = Tokenizer.train(["hello  there", "look here"], ["🍑", "⛳"])
        self.assertEqual(tok.decode(tok.encode(sample)), sample)
        self.assertEqual(len(tok.encode("🍑⛳")), 2)
        ids, mask = tok.build("rebus", "🍑", "peach <alt> butt")
        self.assertEqual(ids.count(tok.sid["<alt>"]), 1)
        self.assertEqual(ids[-1], tok.sid["<eos>"])
        self.assertEqual(mask[-1], 1)

    def test_rules_accept_full_bracket_phrases_and_emoji_variants(self):
        entries = load_lexicon()
        self.assertEqual(
            apply_rules("encode", "I baked an [apple pie].", entries),
            "I baked an 🍎🥧.",
        )
        self.assertEqual(
            apply_rules("encode", "I baked an [apple] [pie].", entries),
            "I baked an 🍎 🥧.",
        )
        self.assertIsNone(apply_rules("encode", "Keep [warm apple pie].", entries))
        self.assertEqual(apply_rules("decode", "A 🍑⛳️!", entries), "A asshole!")

    def test_auto_decode_only_accepts_replacements_inside_emoji_runs(self):
        source = "  I saw 🌙 🚲, then 🐟!  "
        symbols = emojis_in(source)
        self.assertTrue(
            decode_preserves_text(source, "  I saw a moon bike, then fish!  ", symbols)
        )
        self.assertFalse(
            decode_preserves_text(source, "I saw a moon bike, then fish!", symbols)
        )
        self.assertFalse(
            decode_preserves_text(source, "  I own a moon bike, then fish!  ", symbols)
        )
        self.assertFalse(
            decode_preserves_text(source, "  I saw a moon bike, then !  ", symbols)
        )

    def test_holdout_blocks_selectors_and_hints_come_from_training_morphemes(self):
        holdout = [
            {"task": "rebus", "src": "🌧️🏹", "tgt": "rainbow"},
            {"task": "encode", "src": "A [rain] [bow] appeared."},
        ]
        reserved = corpora.reserved_filter(holdout)
        self.assertTrue(reserved({"src": "Look at 🌧︎ 🏹!", "tgt": "A colorful arch."}))
        self.assertNotIn("man", corpora.hint_dictionary().get("⛄", []))

    def test_decode_copies_plain_text_in_auto_mode_even_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.touch()
            text = "  Ordinary text.\tLeave this alone.  "
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "emojilm.generate",
                    "--text",
                    text,
                    "--ckpt",
                    str(checkpoint),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, text + "\n")

    def test_merge_reserves_holdout_and_splits_hinted_twins_together(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            external = data / "local.jsonl"
            external.write_text(
                json.dumps({"text": "a butterfly", "emoji": "🧈 🪰"})
                + "\n"
                + json.dumps({"text": "a pizza", "emoji": "🍕"})
                + "\n",
                encoding="utf-8",
            )
            corpora.build_mix(data, t2e_path=external, core_repeat=2)
            train, val, holdout = [
                [
                    json.loads(line)
                    for line in (data / f"{split}.jsonl").read_text().splitlines()
                ]
                for split in ("train", "val", "holdout")
            ]
            self.assertEqual(sum(row["task"] == "rebus" for row in holdout), 10)
            self.assertTrue(any(row["tag"] == "gpt" for row in train))
            self.assertTrue(any(row["task"] == "emojify" for row in train + val))
            self.assertFalse(
                any(
                    "butterfly" in row["src"].lower()
                    or "butterfly" in row["tgt"].lower()
                    for row in train + val
                )
            )
            self.assertFalse(
                {(r["task"], r["src"]) for r in train}
                & {(r["task"], r["src"]) for r in val}
            )
            self.assertFalse(
                any(
                    "[cat] [fish]" in row["src"] or "[snow] [man]" in row["src"]
                    for row in train + val
                )
            )
            targets = {
                row["tgt"]
                for row in train + val
                if row["task"] == "encode" and row["src"] == "I ordered [ice] [coffee]."
            }
            self.assertEqual(targets, {"I ordered 🧊☕."})
            self.assertEqual(
                len(val), len({json.dumps(r, sort_keys=True) for r in val})
            )


if __name__ == "__main__":
    unittest.main()
