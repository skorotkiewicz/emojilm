# EmojiLM

[![Hugging Face model](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Grizzlykw/emojilm)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
[![MIT License](https://img.shields.io/badge/License-MIT-2EA44F?style=flat-square)](LICENSE)

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

To try all four modes in your browser, run the [Emoji Post Office web playground](docs/README.md). It includes the ONNX model and runs on your device.

## Run a CPU smoke test

With `just` installed, create the local Python environment. The smoke recipe builds its own offline dataset and runs eight training steps.

```sh
just setup
just smoke
```

## Use the final model

See the [EmojiLM model page on Hugging Face](https://huggingface.co/Grizzlykw/emojilm).

To run the checkpoint on CPU with a native Rust binary, see the [Rust CLI quick start](rust-cli/README.md).

EmojiLM uses `ckpt/final/model.pt` automatically when it is available. Use `--mode encode` for bracketed text and `--mode decode` for emoji.

```sh
.venv/bin/python -m emojilm.generate --mode decode --text 'A 🧈🪰 landed on her hand.'
```

Use `emojify` to turn a sentence into an emoji summary.

```sh
.venv/bin/python -m emojilm.generate --mode emojify --text 'I love pizza and my dog'
```

## Train the final model

From a fresh checkout, prepare the [training data](README-dev.md#train-the-final-model) and train with a CUDA-enabled PyTorch environment.

```sh
just data
just train --device cuda
just eval
```

Training runs for 12,000 steps and saves `ckpt/final/model.pt` with the readings and tokenizer embedded. When training finishes, it also writes `ckpt/final/tokenizer.json`. The training log is `ckpt/final/train.log`. The recipe uses `.venv/bin/python` by default. Set `EMOJILM_PYTHON` to use another interpreter.

See [the developer guide](README-dev.md) for data sources, engine options, and training settings.
