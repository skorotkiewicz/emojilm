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

## Use the final model

With the checkpoint at `ckpt/final/model.pt`, use `model-edit` to translate emoji or bracketed text in place.

```sh
.venv/bin/python -m emojilm.generate --engine model-edit \
  --ckpt ckpt/final/model.pt --text 'A 🧈🪰 landed on her hand.'
```

Use `emojify` to turn a sentence into an emoji summary.

```sh
.venv/bin/python -m emojilm.generate --mode emojify --text 'I love pizza and my dog' \
  --ckpt ckpt/final/model.pt
```

## Train the final model

Use the [prepared training data](README-dev.md#train-the-final-model) and a CUDA-enabled PyTorch environment.

```sh
just train-final --device cuda
```

Training runs for 12,000 steps and saves `ckpt/final/model.pt` with the readings embedded. The training log is `ckpt/final/train.log`. The recipe uses `.venv/bin/python` by default. Set `EMOJILM_PYTHON` to use another interpreter.

See [the developer guide](README-dev.md) for data sources, engine options, and training settings.
