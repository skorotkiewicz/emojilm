import json
from pathlib import Path
import tempfile
import unittest

from emojilm.data import load_jsonl
from emojilm.tokenizer import Tokenizer


class StoryDataTests(unittest.TestCase):
    def test_augments_only_training_and_reserves_separate_stories(self):
        from emojilm.story_data import build_experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            output = root / "experiment"
            source = root / "stories.txt"
            stories = [
                f"The cat saw the dog behind door {i}. A butterfly was outside."
                for i in range(100)
            ]
            source.write_text(
                "\n<|endoftext|>\n".join(stories + stories[:1]), encoding="utf-8"
            )
            base_rows = [
                {"task": "rebus", "src": "🐱", "tgt": "cat", "tag": "naming"},
                {"task": "rebus", "src": "🐈", "tgt": "cat", "tag": "naming"},
                {"task": "rebus", "src": "🐕", "tgt": "dog", "tag": "naming"},
            ]
            (base / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8"
            )
            Tokenizer.train(["cat", "dog", "The cat saw the dog."], ["🐱", "🐕"]).save(
                base / "tokenizer.json"
            )
            for filename in ("val.jsonl", "composition-val.jsonl", "holdout.jsonl"):
                (base / filename).write_text(
                    json.dumps({"task": "rebus", "src": "🧈🪰", "tgt": "butterfly"})
                    + "\n",
                    encoding="utf-8",
                )

            build_experiment(base, source, output, train_limit=48, validation_limit=12)

            train = load_jsonl(output / "train.jsonl")
            self.assertEqual(train[: len(base_rows)], base_rows)
            added = train[len(base_rows) :]
            validation = load_jsonl(output / "context-val.jsonl")
            self.assertEqual((len(added), len(validation)), (48, 12))
            self.assertTrue(
                {r["story_id"] for r in added}.isdisjoint(
                    {r["story_id"] for r in validation}
                )
            )
            self.assertTrue(any(row["src"] == row["tgt"] for row in added))
            self.assertTrue(any("[cat] saw the [dog]" in row["src"] for row in added))
            self.assertFalse(any("🐈" in row["tgt"] for row in added + validation))
            self.assertFalse(
                any("butterfly" in row["tgt"] for row in added + validation)
            )
            for filename in (
                "tokenizer.json",
                "val.jsonl",
                "composition-val.jsonl",
                "holdout.jsonl",
            ):
                self.assertEqual(
                    (base / filename).read_bytes(), (output / filename).read_bytes()
                )

    def test_copy_metric_distinguishes_replacement_from_surrounding_text(self):
        from emojilm.eval import preserves_surrounding_text

        source = "The 🐱 sat by the 🐕."
        self.assertTrue(
            preserves_surrounding_text("decode", source, "The lion sat by the wolf.")
        )
        self.assertFalse(
            preserves_surrounding_text("decode", source, "A cat sat by the dog.")
        )
        self.assertTrue(
            preserves_surrounding_text("encode", "The [cat] sat.", "The 🐱 sat.")
        )
        self.assertFalse(
            preserves_surrounding_text("encode", "The cat sat.", "The cat ran.")
        )
