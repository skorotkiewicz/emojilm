# EmojiLM

EmojiLM is a Python-only emoji and text translator. It combines the Qwen
project’s decoder architecture, emoji-aware tokenizer and larger corpus with
GPT’s curated rebuses, plain-text encoding examples and rule engine. The
default model has about 11.2M parameters. Use `just train --model-size tiny`
for the original 3.9M scale.

## Train on another machine

Copy this directory, including the prepared files in `data/`. The prepared
mixture includes the curated examples, CLDR names and a sample of the local
Text2Emoji corpus. The files are ignored by Git. A Git-only copy needs to
rebuild the data first.

From the project directory, with Python 3.10 or later and `just` installed:

```sh
just setup
just train
just eval
just eval --hints
```

Training selects CUDA when PyTorch detects it, and uses CPU otherwise. Install
a PyTorch build that supports the GPU on the training machine. The default run
uses 4,000 steps, batch size 32 and a 256-token context. Set overrides on the
command line, for example `just train --device cuda --steps 8000 --batch 16`.
Every 250 steps and at the end, training reports validation loss by task. It
saves the best checkpoint to `ckpt/model.pt` with its tokenizer embedded.

## Rebuild the dataset

`just data` builds the curated dataset offline. `just data-full` reuses
`data/raw/cldr_en.json` and `data/raw/text2emoji.jsonl` if present. The first
run downloads missing inputs and converts the Text2Emoji CSV to JSONL. It
uses the [Unicode CLDR annotations](https://github.com/unicode-org/cldr-json/tree/main/cldr-json/cldr-annotations-full/annotations/en)
and the [Text2Emoji dataset](https://huggingface.co/datasets/KomeijiForce/Text2Emoji).

For an offline build, copy in both raw files or pass their paths as
positional arguments to `just data-full`. Both data recipes regenerate the
prepared files in `data/`. Use `just data --help` for source and sampling options.

Review the terms for the optional corpora before sharing the prepared
mixture. The code and the curated data are sufficient to build the offline
dataset.

Training draws about 35% decode, 30% encode, 25% rebus and 10% emojify examples
when all four tasks are available. Emoji summaries teach the `emojify` task,
while encode examples teach phrase substitution. Validation keeps variants of
the same input and reversible translation pairs together. The separate holdout contains 120 prompts across ten
reserved emoji combinations. Training data and tokenizer fitting exclude the
holdout. `just eval` measures exact matches using the model, without invoking
the rule engine.

## Translate

```sh
.venv/bin/python -m emojilm.generate --text "Don't be such an 🍑⛳"
.venv/bin/python -m emojilm.generate --mode encode --text "I baked an [apple] [pie]."
.venv/bin/python -m emojilm.generate --mode rebus --text '🐝🍃'
.venv/bin/python -m emojilm.generate --mode emojify --text 'I love pizza and my dog'
```

Auto mode applies curated rules when it finds a match and preserves the
surrounding text. With a checkpoint, it uses the model for unmatched inputs. Auto decode retains the input if the
model changes text outside the emoji runs. Choose `--engine model` to see neural results on a known rebus, or
`--engine rules` to use the dictionary alone. Rules work before training.
`emojify` requires a checkpoint.

`just smoke` runs eight CPU steps and reloads `ckpt/smoke/model.pt` for
inference. It checks the pipeline. Use `just test` for the regression tests.

The `just` recipes use `.venv/bin/python` by default. Set
`EMOJILM_PYTHON` to use an existing environment.
