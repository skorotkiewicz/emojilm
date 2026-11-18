import unittest


class EncodeEditTests(unittest.TestCase):
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
