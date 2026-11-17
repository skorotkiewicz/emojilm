# EmojiLM

EmojiLM is a small Python project for translating emoji and text, solving rebuses, and training a local 11.2M-parameter model.

## Try it

From the project directory, use Python 3.10 or later. These examples use the built-in rules and work before training.

```sh
python3 -m emojilm.generate --text "Don't be such an 🍑⛳"
# Don't be such an asshole

python3 -m emojilm.generate --mode rebus --text '🐝🍃'
# believe (bee + leaf sounds similar to believe)

python3 -m emojilm.generate --mode encode --text 'I baked an [apple] [pie].'
# I baked an 🍎 🥧.
```

## Run a CPU smoke test

With `just` installed, create the local Python environment, build the offline dataset, and run eight training steps.

```sh
just setup
just data
just smoke
```

## Train and evaluate

Use a CUDA-enabled PyTorch environment for GPU training. The recipes use `.venv/bin/python` by default. Set `EMOJILM_PYTHON` to use another interpreter.

```sh
just data-composition
just train-composition --device cuda
just eval --engine model \
  --ckpt ckpt/composition/model.pt \
  --data data/composition/holdout.jsonl
```

The data recipe downloads missing corpora. The trainer runs for 12,000 steps and picks the checkpoint with the most exact matches on the composition development set. The evaluation command tests direct generation on 120 held-out prompts.

To translate with the trained checkpoint, use `--engine model-compose` to read each component and preserve the surrounding text.

```sh
.venv/bin/python -m emojilm.generate --engine model-compose \
  --ckpt ckpt/composition/model.pt --text 'A 🧈🪰 landed on her hand.'
```

Use `emojify` to turn a sentence into an emoji summary.

```sh
.venv/bin/python -m emojilm.generate --mode emojify --text 'I love pizza and my dog' \
  --ckpt ckpt/composition/model.pt
```

See [the developer guide](README-dev.md) for data sources, engine options, and training settings.
