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

## Train for direct composition

Use the composition recipe to teach the model to translate closed
compounds and preserve the text around them.

```sh
just data-composition
just train-composition --device cuda
just eval --engine model --ckpt ckpt/composition/model.pt --data data/composition/holdout.jsonl
```

The builder adds individual emoji readings, closed compounds, and
examples that distinguish adjacent emojis from emojis separated by
spaces. Compound targets use separately encoded word parts so the model
can learn how to join them. The recipe uses 12,000 steps, a batch size of
32, a learning rate of 0.0004, and a 256-token context.

Every 1,000 steps and at the end, the trainer tests direct generation
on 60 development prompts drawn from 12 reserved combinations. It
saves the checkpoint with the most exact matches, using regular
validation loss to break ties. The final 120-prompt holdout is reserved
for comparison after training.

The prepared experiment lives in `data/composition/`. Copy that
directory with the project to train on a different machine. The new
checkpoint is written to `ckpt/composition/model.pt`.

Training appends loss and validation scores to `train.log` in the
checkpoint directory. For a composition run, the log is
`ckpt/composition/train.log`.

```sh
tail -f ckpt/composition/train.log
```

## Train with TinyStories contexts

After preparing `data/composition`, build an experiment with up to 5,000
additional encode and decode examples. Each pair uses known emoji readings
and keeps the surrounding sentence intact. The builder downloads the small
TinyStories validation file on first use. [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)

```sh
just data-tinystories
just train-tinystories --device cuda
```

The recipe uses the same 12,000-step budget and saves to
`ckpt/tinystories/model.pt`. If you already have the TinyStories file, pass
its path with `just data-tinystories --stories PATH`.

```sh
just eval --engine model --copy-metrics --data data/tinystories/context-val.jsonl --ckpt ckpt/tinystories/model.pt
just eval --engine model --data data/composition/holdout.jsonl --ckpt ckpt/tinystories/model.pt
```

Run the same evaluations with `--ckpt ckpt/composition/model.pt` for the
baseline. On the held-out story sentences, `preserved_text` measures how
often the model copies the text outside emoji or bracketed spans. `exact`
also requires the expected translation.

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
holdout. `just eval` reports separate exact-match scores for direct model
generation, composition using the checkpoint's individual emoji readings,
and composition using the component lexicon. All three use the same
prompts and skip the curated phrase rules. Use `just eval --engine model`,
`--engine model-compose`, or `--engine compose` to run one evaluation.

## Translate

```sh
.venv/bin/python -m emojilm.generate --text "Don't be such an 🍑⛳"
.venv/bin/python -m emojilm.generate --mode encode --text "I baked an [apple] [pie]."
.venv/bin/python -m emojilm.generate --mode rebus --text '🐝🍃'
.venv/bin/python -m emojilm.generate --mode emojify --text 'I love pizza and my dog'
```

Auto mode applies curated phrase rules first. When encoding bracketed
text, it converts recognized components and keeps the rest of the
sentence intact. For rebuses it can also assemble compact emoji
components. Auto decode retains the input if neural generation changes
text outside the emoji runs.

Use `--engine model-compose` to translate components with the checkpoint,
or `--engine compose` to use the component lexicon. Adjacent emojis form
a compound and literal spaces stay in the output. Both modes encode
explicit bracketed components. Use `--engine model` for direct neural
generation or `--engine rules` for curated phrases. `emojify` uses direct
neural generation.

`just smoke` runs eight CPU steps and reloads `ckpt/smoke/model.pt` for
inference. It checks the pipeline. Use `just test` for the regression tests.

The `just` recipes use `.venv/bin/python` by default. Set
`EMOJILM_PYTHON` to use an existing environment.
