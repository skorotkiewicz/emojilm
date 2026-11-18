# Emoji Post Office

A browser playground for EmojiLM. Pick a card from the mailbag or type a message. Encode replaces bracketed words in context, Decode reads emoji and compounds, Rebus tries to sound out emoji puzzles, and Emojify generates an emoji summary. Press Ctrl+Enter or ⌘+Enter to translate.

## Run locally

From the EmojiLM project root:

```sh
python3 -m http.server --bind 127.0.0.1 --directory web 8000
```

Open [localhost:8000](http://localhost:8000). The directory is also ready for static hosting. Serve the JavaScript modules as JavaScript and the WASM binary as `application/wasm`. A browser with WebAssembly SIMD and module worker support can run the whole translator on-device. Messages stay in the browser.

## Model files

- `model/model.onnx` is the 11.5 MB export of the noun/verb release. It stores the large weight matrices as 8-bit integers and performs inference in floating point. Its input is `ids` (int64, shape `[1, sequence]`). Its outputs are `hidden` and `logits`. The maximum sequence length is 256.
- `model/bundle.json` supplies the matching tokenizer, bundled readings and trained classifiers. Deploy it together with its ONNX graph.
- `vendor/` contains ONNX Runtime Web and its single-threaded WASM backend files. The worker loads the neural model on demand. Bundled dictionary lookups can answer some requests on their own.

The web translator matched the PyTorch implementation on 197 comparison prompts across the four modes. This measures agreement between implementations. Translation quality scores are on the [model card](https://huggingface.co/Grizzlykw/emojilm).

## Re-export a checkpoint

In a Python environment with the project's PyTorch dependencies, run these commands from the project root:

```sh
python -m pip install onnx onnxruntime numpy
python web/export_model.py --ckpt ckpt/final/model.pt --out web/model --int8
```

Use a bundled checkpoint with edit candidates and decode readings. Omit `--int8` for a full-precision export. The exporter checks the graph and compares outputs against PyTorch at several sequence lengths. To exercise the shipped WASM runtime and model together with Node.js 24:

```sh
node --test tests/web-inference.test.mjs
```
