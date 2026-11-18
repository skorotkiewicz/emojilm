use anyhow::{Result, ensure};
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;

pub const BOS: u32 = 256;
pub const EOS: u32 = 257;
pub const DECODE: u32 = 259;
pub const ENCODE: u32 = 260;
pub const REBUS: u32 = 261;
pub const SEP: u32 = 262;
pub const ALT: u32 = 263;
pub const HINT: u32 = 264;
pub const EMOJIFY: u32 = 265;
const SPECIALS: [&str; 10] = [
    "<bos>",
    "<eos>",
    "<pad>",
    "<decode>",
    "<encode>",
    "<rebus>",
    "<sep>",
    "<alt>",
    "<hint>",
    "<emojify>",
];
const MODS: [&str; 8] = [
    "\u{fe0f}", "\u{200d}", "\u{20e3}", "🏻", "🏼", "🏽", "🏾", "🏿",
];

fn base(c: char) -> bool {
    matches!(c as u32,0xa9|0xae|0x203c|0x2049|0x2122|0x2139|0x2194..=0x21aa|0x231a..=0x231b|0x2328|0x23cf|0x23e9..=0x23fa|0x24c2|0x25aa..=0x25fe|0x2600..=0x27bf|0x2b00..=0x2bff|0x3030|0x303d|0x3297..=0x3299|0x1f000..=0x1f3fa|0x1f400..=0x1faff|0x1fc00..=0x1fffd)
}

fn take_char(rest: &mut &str, predicate: impl FnOnce(char) -> bool) -> bool {
    if let Some(c) = rest.chars().next().filter(|&c| predicate(c)) {
        *rest = &rest[c.len_utf8()..];
        true
    } else {
        false
    }
}

// Return a byte slice so callers can copy the surrounding UTF-8 verbatim.
pub fn emoji_at(s: &str) -> Option<&str> {
    let first = s.chars().next()?;
    let mut rest = &s[first.len_utf8()..];
    if "#*0123456789".contains(first) {
        rest = rest.strip_prefix('\u{fe0f}').unwrap_or(rest);
        rest = rest.strip_prefix('\u{20e3}')?;
    } else if ('🇦'..='🇿').contains(&first) {
        if !take_char(&mut rest, |c| ('🇦'..='🇿').contains(&c)) {
            return None;
        }
    } else if base(first) {
        loop {
            take_char(&mut rest, |c| matches!(c, '\u{fe0f}' | '\u{fe0e}'));
            take_char(&mut rest, |c| ('🏻'..='🏿').contains(&c));
            let tag = |c| ('\u{e0020}'..='\u{e007f}').contains(&c);
            if first == '🏴' && take_char(&mut rest, tag) {
                while take_char(&mut rest, tag) {}
                break;
            }
            let Some(mut next) = rest.strip_prefix('\u{200d}') else {
                break;
            };
            if !take_char(&mut next, base) {
                break;
            }
            rest = next;
        }
    } else {
        return None;
    }
    Some(&s[..s.len() - rest.len()])
}

#[derive(Deserialize)]
pub struct Spec {
    format: String,
    size: usize,
    emoji: Vec<String>,
    merges: Vec<[String; 2]>,
    text_tokens: Vec<String>,
}
pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<String, HashMap<String, usize>>,
    known: HashMap<char, Vec<String>>,
    pieces: Regex,
}
impl Tokenizer {
    pub fn new(spec: Spec, vocab: usize) -> Result<Self> {
        ensure!(
            spec.format == "emojilm-tok-2" && spec.size == vocab,
            "incompatible tokenizer"
        );
        let mut tokens: Vec<_> = (0..256).map(|i| format!("<byte{i}>")).collect();
        tokens.extend(SPECIALS.into_iter().chain(MODS).map(str::to_owned));
        tokens.extend_from_slice(&spec.emoji);
        tokens.extend(spec.text_tokens);
        ensure!(tokens.len() <= vocab, "vocabulary overflow");
        let ids = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let mut known: HashMap<char, Vec<String>> = HashMap::new();
        for s in spec.emoji.into_iter().chain(MODS.map(str::to_owned)) {
            if let Some(c) = s.chars().next() {
                known.entry(c).or_default().push(s);
            }
        }
        for list in known.values_mut() {
            list.sort_by_key(|s| std::cmp::Reverse(s.len()));
        }
        let mut ranks: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for (i, [a, b]) in spec.merges.into_iter().enumerate() {
            ranks.entry(a).or_default().insert(b, i);
        }
        Ok(Self {
            tokens,
            ids,
            ranks,
            known,
            pieces: Regex::new(r"(?s) ?(?:[A-Za-z]+|[0-9]|[^A-Za-z0-9\s▁])|.")?,
        })
    }
    pub fn used(&self) -> usize {
        self.tokens.len()
    }
    fn emit(&self, s: &str, out: &mut Vec<u32>) {
        if let Some(id) = self.ids.get(s) {
            out.push(*id)
        } else {
            out.extend(s.bytes().map(u32::from));
        }
    }
    fn text(&self, s: &str, out: &mut Vec<u32>) {
        for (i, part) in s.split('▁').enumerate() {
            if i > 0 {
                out.extend("▁".bytes().map(u32::from));
            }
            for m in self.pieces.find_iter(part) {
                let mut symbols: Vec<String> = m.as_str().chars().map(|c| c.to_string()).collect();
                if symbols[0] == " " {
                    symbols[0] = "▁".into();
                }
                while let Some((_, index)) = symbols
                    .windows(2)
                    .enumerate()
                    .filter_map(|(i, p)| Some((*self.ranks.get(&p[0])?.get(&p[1])?, i)))
                    .min()
                {
                    let right = symbols.remove(index + 1);
                    symbols[index].push_str(&right);
                }
                for sym in symbols {
                    self.emit(&sym, out);
                }
            }
        }
    }
    pub fn encode(&self, s: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let (mut start, mut i) = (0, 0);
        while i < s.len() {
            let rest = &s[i..];
            let c = rest.chars().next().unwrap();
            let found = self
                .known
                .get(&c)
                .and_then(|v| {
                    v.iter()
                        .find(|g| rest.starts_with(g.as_str()))
                        .map(String::as_str)
                })
                .or_else(|| emoji_at(rest));
            if let Some(g) = found {
                self.text(&s[start..i], &mut out);
                if let Some(id) = self.ids.get(g) {
                    out.push(*id)
                } else {
                    for (j, part) in g.split('\u{200d}').enumerate() {
                        if j > 0 {
                            self.emit("\u{200d}", &mut out);
                        }
                        if let Some(id) = self.ids.get(part) {
                            out.push(*id)
                        } else {
                            for ch in part.chars() {
                                self.emit(&ch.to_string(), &mut out);
                            }
                        }
                    }
                }
                i += g.len();
                start = i;
            } else {
                i += c.len_utf8();
            }
        }
        self.text(&s[start..], &mut out);
        out
    }
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut bytes = Vec::new();
        for &id in ids {
            if id < 256 {
                bytes.push(id as u8);
                continue;
            }
            out.push_str(&String::from_utf8_lossy(&bytes));
            bytes.clear();
            if id == ALT {
                out.push_str(" / ")
            } else if id > EMOJIFY
                && let Some(s) = self.tokens.get(id as usize)
            {
                out.push_str(&s.replace('▁', " "));
            }
        }
        out.push_str(&String::from_utf8_lossy(&bytes));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn consumes_one_whole_emoji() {
        for glyph in ["🧑🏾‍🚀", "1️⃣", "🇧🇷", "❤️", "🐝"] {
            let input = format!("{glyph}🍃!");
            assert_eq!(emoji_at(&input), Some(glyph));
        }
        assert_eq!(emoji_at("1x"), None);
        assert_eq!(emoji_at("🦊‍?"), Some("🦊"));
    }
    #[test]
    fn bpe_and_byte_fallback_round_trip() -> Result<()> {
        let spec = Spec {
            format: "emojilm-tok-2".into(),
            size: 512,
            emoji: vec!["🍎".into()],
            merges: vec![["h".into(), "i".into()]],
            text_tokens: ["▁", "h", "i", "hi"].map(String::from).into(),
        };
        let tok = Tokenizer::new(spec, 512)?;
        assert_eq!(tok.encode("hi").len(), 1);
        let text = "hi  🍎\n你好 ▁🚀";
        assert_eq!(tok.decode(&tok.encode(text)), text);
        Ok(())
    }
}
