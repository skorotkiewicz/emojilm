mod checkpoint;
mod model;
mod tokenizer;
mod translate;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use translate::Translator;

#[derive(Copy, Clone, ValueEnum)]
enum Mode {
    Encode,
    Decode,
    Rebus,
    Emojify,
}

#[derive(Parser)]
#[command(about = "Translate English and emoji with a local EmojiLM PyTorch checkpoint")]
struct Args {
    mode: Mode,
    /// Text to translate. Put square brackets around words to encode.
    text: String,
    #[arg(long, default_value = "ckpt/final/model.pt")]
    ckpt: PathBuf,
    /// Generate directly with the Transformer in encode/decode mode.
    #[arg(long)]
    raw: bool,
    #[arg(long, default_value = "64", value_parser=clap::value_parser!(u32).range(1..))]
    max_new_tokens: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let translator =
        Translator::open(&args.ckpt).with_context(|| format!("loading {}", args.ckpt.display()))?;
    println!(
        "{}",
        translator.translate(
            args.mode,
            &args.text,
            args.raw,
            args.max_new_tokens as usize
        )?
    );
    Ok(())
}
