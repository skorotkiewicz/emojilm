# EmojiLM Rust CLI

Run EmojiLM locally on CPU. The Rust binary uses Candle to load the Transformer weights, tokenizer, bundled readings, and classifiers directly from `model.pt`. The binary runs independently of Python.

From the EmojiLM project root, build with Cargo. Place the downloaded checkpoint at `ckpt/final/model.pt` or pass its location with `--ckpt`. [Download the checkpoint](https://huggingface.co/Grizzlykw/emojilm/tree/main)

```sh
cargo build --locked --release --manifest-path rust-cli/Cargo.toml
rust-cli/target/release/emojilm encode 'I usually [train] before I catch the [train].' --ckpt ckpt/final/model.pt
# I usually [train] before I catch the 🚆.

rust-cli/target/release/emojilm decode 'I baked 🍎🥧 yesterday.'
# I baked apple pie yesterday.
```

`encode` edits bracketed words using the bundled candidates and trained classifiers. `decode` reads individual emoji and known compounds. `rebus` and `emojify` use greedy neural generation. Use `--raw` for direct neural generation in any mode and `--max-new-tokens` to set the generation limit (default 64). The Python CLI also offers curated phrase rules through its `auto` engine.
