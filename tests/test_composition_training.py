from pathlib import Path
import tempfile
import unittest

from emojilm.corpora import build_mix, reserved_filter
from emojilm.data import emojis_in, load_jsonl
from emojilm.tokenizer import Tokenizer


class CompositionTrainingTests(unittest.TestCase):
    def test_compound_target_decodes_with_the_boundary_between_components(self):
        tok = Tokenizer.train(
            ["moonlight moonlight moonlight", "moon light"], ["🌙", "💡"]
        )
        ids, mask = tok.build(
            "rebus", "🌙💡", "moonlight", target_parts=["moon", "light"]
        )
        target = [token for token, learn in zip(ids, mask) if learn]
        self.assertEqual(
            target, tok.encode("moon") + tok.encode("light") + [tok.sid["<eos>"]]
        )
        self.assertEqual(tok.decode(target), "moonlight")
        with self.assertRaises(ValueError):
            tok.build("rebus", "🌙💡", "moonlight", target_parts=["moon", "fish"])

    def test_composition_build_keeps_development_pairs_and_final_holdout_out_of_train(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_mix(root, composition=True, bpe_sample=1000)
            train = load_jsonl(root / "train.jsonl")
            val = load_jsonl(root / "val.jsonl")
            dev = load_jsonl(root / "composition-val.jsonl")
            holdout = load_jsonl(root / "holdout.jsonl")
            self.assertEqual(len(holdout), 120)
            self.assertGreaterEqual(len(dev), 30)
            self.assertEqual({row["tag"] for row in dev}, {"composition"})
            self.assertGreater(sum(row["tag"] == "composition" for row in train), 500)
            self.assertTrue(any("target_parts" in row for row in train))
            self.assertFalse(
                any(row["tag"] == "composition" and "hint" in row for row in train)
            )
            forbidden = reserved_filter(dev + holdout)
            self.assertFalse(any(forbidden(row) for row in train + val))
            self.assertFalse(any(reserved_filter(holdout)(row) for row in dev))
            atoms_seen = {
                glyph for row in train for glyph in emojis_in(row["src"] + row["tgt"])
            }
            self.assertTrue(
                all(
                    set(emojis_in(row["src"])) <= atoms_seen
                    for row in dev
                    if row["task"] == "rebus"
                )
            )
            bare_rebuses = {row["src"] for row in train if row["task"] == "rebus"}
            self.assertTrue({"🐉", "🌧️", "🏹"} <= bare_rebuses)
