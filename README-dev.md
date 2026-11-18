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

With Python 3.10 or later and `just` installed, run `just setup` in the
project directory or select an existing Python environment. Then follow
the [final model training recipe](#train-the-final-model).

Training selects CUDA when PyTorch detects it, and uses CPU otherwise. The
final recipe uses 12,000 steps, batch size 32 and a 256-token context. Training
reports validation loss by task every 250 steps and at the end.

## Reproduce the composition experiment

Use the composition recipe to teach the model to translate closed
compounds and preserve the text around them.

```sh
.venv/bin/python -m emojilm.corpora --fetch-missing --composition --data data/composition
.venv/bin/python -m emojilm.train --device cuda --steps 12000 \
  --data data/composition --out ckpt/composition --composition-val data/composition/composition-val.jsonl
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

## Reproduce the TinyStories experiments

After preparing `data/composition`, build an experiment with up to 5,000
additional encode and decode examples. Each pair uses known emoji readings
and keeps the surrounding sentence intact. The builder downloads the small
TinyStories validation file on first use. [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)

```sh
.venv/bin/python -m emojilm.story_data
.venv/bin/python -m emojilm.train --device cuda --steps 12000 \
  --data data/tinystories --out ckpt/tinystories --composition-val data/tinystories/composition-val.jsonl
```

The recipe uses the same 12,000-step budget and saves to
`ckpt/tinystories/model.pt`. If you already have the TinyStories file, pass
its path with `.venv/bin/python -m emojilm.story_data --stories PATH`.

```sh
just eval --engine model --copy-metrics --data data/tinystories/context-val.jsonl --ckpt ckpt/tinystories/model.pt
just eval --engine model --data data/composition/holdout.jsonl --ckpt ckpt/tinystories/model.pt
```

Run the same evaluations with `--ckpt ckpt/composition/model.pt` for the
baseline. On the held-out story sentences, `preserved_text` measures how
often the model copies the text outside emoji or bracketed spans. `exact`
also requires the expected translation.

To test longer inputs, build on the prepared `data/tinystories`. This adds up
to 5,000 training rows with two to six marked spans and some bracketed words
that should be copied. The long-context holdout uses separate stories.

```sh
.venv/bin/python -m emojilm.story_data --long
.venv/bin/python -m emojilm.train --device cuda --steps 12000 \
  --data data/tinystories-long --out ckpt/tinystories-long --composition-val data/tinystories-long/composition-val.jsonl
just eval --engine model --copy-metrics --max-new-tokens 128 --data data/tinystories-long/long-val.jsonl --ckpt ckpt/tinystories-long/model.pt
```

Run the same evaluation with `--ckpt ckpt/tinystories/model.pt` for the
baseline.

## Reproduce the bracket replacement experiments

Run these commands after preparing `data/tinystories-long`. The builder adds
examples where the model sees the whole sentence and predicts one bracket's
emoji or `KEEP`. The recipe saves to `ckpt/encode-edits/model.pt`.

```sh
.venv/bin/python -m emojilm.edit_data
.venv/bin/python -m emojilm.train --device cuda --steps 12000 \
  --data data/encode-edits --out ckpt/encode-edits --composition-val data/encode-edits/edit-val.jsonl
```

Use `model-edit` with the new checkpoint for bracketed `encode` inputs.
It copies all surrounding text, including spaces between brackets.

```sh
.venv/bin/python -m emojilm.generate --engine model-edit --ckpt ckpt/encode-edits/model.pt --mode encode --text 'I baked an [apple] [pie].'
just eval --engine model-edit --copy-metrics --data data/encode-edits/long-val.jsonl --ckpt ckpt/encode-edits/model.pt
just eval --engine model-edit --copy-metrics --data data/encode-edits/edit-holdout.jsonl --ckpt ckpt/encode-edits/model.pt
```

`edit-holdout.jsonl` uses the original prompts and keeps spaces between
brackets in their references. Use those same references for comparisons.

To expand bracket-edit vocabulary, build from the existing `data/encode-edits`.
The builder supplements rare readings from the training split and pairs
known compound readings with examples of the same words used separately.

```sh
.venv/bin/python -m emojilm.edit_coverage
.venv/bin/python -m emojilm.train --device cuda --steps 12000 \
  --data data/encode-edits-expanded --out ckpt/encode-edits-expanded --composition-val data/encode-edits-expanded/edit-val.jsonl
```

The run uses the same 12,000 steps and saves to
`ckpt/encode-edits-expanded/model.pt`. Use the same evaluation commands with
that checkpoint to compare it against `ckpt/encode-edits/model.pt`.

After training, bundle the supported readings from the training split into a
portable checkpoint. `model-edit` uses a single supported reading directly,
uses the model to choose among several readings, and copies unfamiliar brackets.
In decode mode the bundle recognizes known compounds within longer emoji runs.
It copies the surrounding text exactly.

```sh
.venv/bin/python -m emojilm.edit_candidates
.venv/bin/python -m emojilm.generate --engine model-edit --ckpt ckpt/encode-edits-bundled/model.pt --mode encode --text 'I baked an [apple] [pie].'
.venv/bin/python -m emojilm.generate --engine model-edit --ckpt ckpt/encode-edits-bundled/model.pt --mode decode --text 'I baked a 🥧.'
```

## Train the final model

Use the prepared `data/encode-edits-expanded/` directory. From a fresh
checkout, `just data` builds all the data needed by `just train`.

```sh
just data
```

```sh
just train --device cuda
just eval
```

The trainer embeds the readings and tokenizer in `ckpt/final/model.pt` when it
saves the best checkpoint. It writes the run log to `ckpt/final/train.log` and
exports `ckpt/final/tokenizer.json` when training finishes. `just eval`
scores encode and decode on the long holdout. Pass
`--data data/encode-edits-expanded/edit-holdout.jsonl` to score the original
prompts with spaces preserved in the references.

## Fit the word-sense classifiers

With the ambiguity dataset and its trained checkpoint available, run the following commands from the project directory.

```sh
.venv/bin/python -m emojilm.sense_tune
just eval --ckpt ckpt/sense/model.pt --data data/ambiguity/ambiguity-holdout.jsonl
just eval --ckpt ckpt/sense/model.pt --data tests/fixtures/sense-transfer.jsonl
```

The CPU fit uses 84 existing ambiguity training edits and 84 additional sentences with contrasting meanings. For each marked word, a linear classifier learns to choose a reading from the Transformer's final prompt representation. The output checkpoint includes a classifier only when it improves exact matches for that word on the development split. In this run, mouse and bat qualified. Both classifiers together contain 1,605 coefficients, including biases.

## Rebuild the dataset

The final data build reuses `data/raw/cldr_en.json` and
`data/raw/text2emoji.jsonl` if present. The first run downloads missing
inputs and converts the Text2Emoji CSV to JSONL. It
uses the [Unicode CLDR annotations](https://github.com/unicode-org/cldr-json/tree/main/cldr-json/cldr-annotations-full/annotations/en)
and the [Text2Emoji dataset](https://huggingface.co/datasets/KomeijiForce/Text2Emoji).

For an offline final build, copy in both raw files and the TinyStories file
before running `just data`. To build a standalone corpus with custom sources,
use `.venv/bin/python -m emojilm.corpora --fetch-missing --cldr CLDR_PATH --t2e TEXT2EMOJI_PATH`.
Run the module with `--help` for sampling options.

Review the terms for the optional corpora before sharing the prepared
mixture. The code and the curated data are sufficient to build the offline
dataset.

Training draws about 35% decode, 30% encode, 25% rebus and 10% emojify examples
when all four tasks are available. Emoji summaries teach the `emojify` task,
while encode examples teach phrase substitution. Validation keeps variants of
the same input and reversible translation pairs together. The separate holdout contains 120 prompts across ten
reserved emoji combinations. Training data and tokenizer fitting exclude the
holdout. For experiments, `just eval --engine all` compares direct model
generation, composition using the checkpoint's individual emoji readings,
and composition using the component lexicon. Pass `--data` and `--ckpt` to
select the experiment's prompts and checkpoint. Use `--engine model`,
`--engine model-compose`, or `--engine compose` to run one evaluation.

## Translate

```sh
.venv/bin/python -m emojilm.generate --text "Don't be such an 🍑⛳"
.venv/bin/python -m emojilm.generate --mode encode --text "I baked an [apple] [pie]."
.venv/bin/python -m emojilm.generate --mode rebus --text '🐝🍃'
.venv/bin/python -m emojilm.generate --mode emojify --text 'I love pizza and my dog'
```

Auto mode applies curated phrase rules first and discovers
`ckpt/final/model.pt` when available. It uses the bundle to encode remaining
brackets and decode emoji when no phrase rule matches. With the rules alone,
bracketed encoding uses the component lexicon. Auto mode can also assemble
compact rebuses. When using direct neural decoding, auto retains the input
if generation changes text outside the emoji runs.

Use `--engine model-compose` to translate components with the checkpoint,
or `--engine compose` to use the component lexicon. Adjacent emojis form
a compound and literal spaces stay in the output. Both modes encode
explicit bracketed components. Use `--engine model` for direct neural
generation or `--engine rules` for curated phrases. `emojify` uses direct
neural generation.

`just smoke` builds the curated dataset in `data/smoke`, runs eight CPU steps,
and reloads `ckpt/smoke/model.pt` for
inference. It checks the pipeline. Use `just test` for the regression tests.

The `just` recipes use `.venv/bin/python` by default. Set
`EMOJILM_PYTHON` to use an existing environment.
