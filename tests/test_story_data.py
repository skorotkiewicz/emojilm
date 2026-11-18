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

            from emojilm.composition import bracket_words

            long_output = root / "long"
            source.write_text(
                "\n<|endoftext|>\n".join(
                    f"At door {i}, the cat greeted the dog and the cat watched another dog. "
                    "A cat and a dog quietly waited outside."
                    for i in range(300)
                ),
                encoding="utf-8",
            )
            build_experiment(
                output,
                source,
                long_output,
                long=True,
                train_limit=48,
                validation_limit=12,
            )
            long_rows = load_jsonl(long_output / "train.jsonl")
            long_validation = load_jsonl(long_output / "long-val.jsonl")
            self.assertEqual(long_rows[: len(train)], train)
            self.assertEqual(
                (len(long_rows) - len(train), len(long_validation)), (48, 12)
            )
            self.assertEqual(
                (output / "context-val.jsonl").read_bytes(),
                (long_output / "context-val.jsonl").read_bytes(),
            )
            long_encode = [
                row for row in long_rows[len(train) :] if row["task"] == "encode"
            ]
            self.assertTrue(
                any(len(list(bracket_words(r["src"]))) == 6 for r in long_encode)
            )
            self.assertTrue(any("[" in row["tgt"] for row in long_encode))
            self.assertTrue(
                {r["story_id"] for r in long_rows if "story_id" in r}.isdisjoint(
                    {r["story_id"] for r in validation + long_validation}
                )
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
