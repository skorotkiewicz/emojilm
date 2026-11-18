use anyhow::{Context, Result, ensure};

use crate::{
    Mode,
    checkpoint::Checkpoint,
    model::{Config, Model},
    tokenizer::*,
};
use regex::Regex;
use std::{collections::HashMap, path::Path};

type Readings = HashMap<String, Vec<String>>;

fn canonical(s: &str) -> String {
    s.replace(['\u{fe0e}', '\u{fe0f}'], "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn brackets(s: &str) -> Vec<(usize, usize)> {
    let (mut spans, mut depth, mut start) = (Vec::new(), 0, 0);
    for (i, c) in s.char_indices() {
        match c {
            '[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            ']' if depth > 0 => {
                depth -= 1;
                if depth == 0 && i > start + 1 && !s[start + 1..i].contains(['[', ']', '\r', '\n'])
                {
                    spans.push((start, i + 1));
                }
            }
            _ => {}
        }
    }
    spans
}

pub struct Translator {
    model: Model,
    tok: Tokenizer,
    edits: Option<Readings>,
    readings: Option<Readings>,
    heads: Readings,
    gated: Vec<String>,
    words: Regex,
    longest: usize,
}
impl Translator {
    pub fn open(path: &Path) -> Result<Self> {
        let ck = Checkpoint::open(path)?;
        let cfg: Config = serde_json::from_value(ck.meta["cfg"].clone())?;
        ensure!(
            ck.tensors.get("model.tok_emb.weight").map(|t| t.dims())
                == Some(&[cfg.vocab, cfg.dim][..]),
            "checkpoint vocabulary and embedding dimensions differ"
        );
        let tok = Tokenizer::new(
            serde_json::from_value(ck.meta["tokenizer"].clone())?,
            cfg.vocab,
        )?;
        let edits = serde_json::from_value(ck.meta["edit_candidates"].clone())?;
        let readings: Option<Readings> =
            serde_json::from_value(ck.meta["decode_readings"].clone())?;
        let longest = readings
            .iter()
            .flat_map(|r| r.keys())
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(1);
        let mut heads = HashMap::new();
        if let Some(map) = ck.meta["sense_heads"].as_object() {
            for (label, head) in map {
                heads.insert(
                    label.clone(),
                    serde_json::from_value(head["choices"].clone())?,
                );
            }
        }
        let gated = if ck.meta["noun_verb_gate"].is_null() {
            Vec::new()
        } else {
            serde_json::from_value(ck.meta["noun_verb_gate"]["labels"].clone())?
        };
        Ok(Self {
            model: Model {
                cfg,
                weights: ck.tensors,
            },
            tok,
            edits,
            readings,
            heads,
            gated,
            words: Regex::new(r"\w+|[^\w\s]")?,
            longest,
        })
    }
    fn noun(&self, s: &str, start: usize, end: usize, label: &str) -> Result<bool> {
        let left = self.tok.encode(s[..start].trim_end());
        let middle = self.tok.encode(&format!("[{label}]"));
        let available = self
            .model
            .cfg
            .ctx
            .checked_sub(middle.len() + 1)
            .context("marked word exceeds context")?;
        let count = left.len().min(24).min(available);
        let mut ids = vec![BOS];
        ids.extend_from_slice(&left[left.len() - count..]);
        ids.extend_from_slice(&middle);
        let marked = ids.len() - 1;
        ids.extend(
            self.tok
                .encode(&s[end..])
                .into_iter()
                .take(12.min(available - count)),
        );
        let h = self.model.hidden(&ids)?;
        let mut features = Vec::new();
        for i in [count, marked, ids.len() - 1] {
            features.extend(h.get(i)?.to_vec1::<f32>()?);
        }
        let before = s[..start].to_lowercase();
        let after = s[end..].to_lowercase();
        let mut last = ["", ""];
        for m in self.words.find_iter(&before) {
            last = [last[1], m.as_str()];
        }
        for word in [
            last[0],
            last[1],
            self.words.find(&after).map_or("", |m| m.as_str()),
        ] {
            for group in [
                "a an the this that these those my your his her its our their each every another some",
                "i you he she it we they",
                "can could will would may might must shall should do does did",
                "to",
            ] {
                features.push(if group.split_ascii_whitespace().any(|w| w == word) {
                    (self.model.cfg.dim as f32).sqrt()
                } else {
                    0.
                });
            }
        }
        let w = self
            .model
            .weight("noun_verb_gate.weight")?
            .to_vec1::<f32>()?;
        ensure!(
            w.len() == features.len(),
            "incompatible noun/verb classifier"
        );
        Ok(w.iter().zip(features).map(|(a, b)| a * b).sum::<f32>()
            + self
                .model
                .weight("noun_verb_gate.bias")?
                .to_scalar::<f32>()?
            > 0.)
    }
    fn encode(&self, s: &str, edits: &Readings, limit: usize) -> Result<String> {
        let mut out = String::new();
        let mut offset = 0;
        for (index, (start, end)) in brackets(s).into_iter().enumerate() {
            out.push_str(&s[offset..start]);
            let original = &s[start..end];
            let word = &s[start + 1..end - 1];
            let label = canonical(word);
            let mut value = None;
            if let Some(choices) = edits.get(&label) {
                let gated = self.gated.contains(&label);
                if !gated || self.noun(s, start, end, &label)? {
                    let choices: Vec<_> = choices
                        .iter()
                        .filter(|v| !gated || v.as_str() != "KEEP")
                        .map(String::as_str)
                        .collect();
                    if choices.len() == 1 {
                        value = Some(choices[0]);
                    } else if !choices.is_empty() {
                        let mut prefix = vec![BOS, ENCODE, HINT];
                        prefix.extend(self.tok.encode(&format!("{s}\n{} [{word}]", index + 1)));
                        prefix.push(SEP);
                        value = self.choose(&label, &prefix, &choices, limit)?;
                    }
                }
            } else {
                value = match label.as_str() {
                    "think" => Some("🤔"),
                    "eat" => Some("🍽️"),
                    _ => None,
                };
            }
            out.push_str(value.filter(|v| *v != "KEEP").unwrap_or(original));
            offset = end;
        }
        out.push_str(&s[offset..]);
        Ok(out)
    }
    fn choose<'a>(
        &self,
        label: &str,
        prefix: &[u32],
        choices: &[&'a str],
        limit: usize,
    ) -> Result<Option<&'a str>> {
        if prefix.len() > self.model.cfg.ctx {
            return Ok(None);
        }
        let limit = limit.min(self.model.cfg.ctx + 1 - prefix.len());
        let targets: Vec<_> = choices
            .iter()
            .map(|v| {
                let mut ids = self.tok.encode(v);
                ids.push(EOS);
                ids
            })
            .collect();
        if self.heads.get(label).is_some_and(|h| h == choices)
            && targets.iter().all(|t| t.len() <= limit)
        {
            let key = format!("sense_heads.{label}");
            let h = self
                .model
                .hidden(prefix)?
                .get(prefix.len() - 1)?
                .unsqueeze(1)?;
            let scores = self
                .model
                .weight(&format!("{key}.weight"))?
                .matmul(&h)?
                .squeeze(1)?
                .add(self.model.weight(&format!("{key}.bias"))?)?;
            return Ok(choices
                .get(scores.argmax(0)?.to_scalar::<u32>()? as usize)
                .copied());
        }
        let mut best = None;
        let mut high = f32::NEG_INFINITY;
        for (&choice, target) in choices.iter().zip(targets) {
            if target.len() <= limit {
                let score = self.model.score(prefix, &target)?;
                if score > high {
                    best = Some(choice);
                    high = score;
                }
            }
        }
        Ok(best)
    }
    fn read<'a>(readings: &'a Readings, glyph: &str) -> Option<&'a str> {
        readings
            .get(&canonical(glyph))
            .and_then(|v| v.first())
            .map(String::as_str)
            .or(match glyph {
                "🤔" => Some("thinking face"),
                "🍽️" => Some("fork and knife with plate"),
                _ => None,
            })
    }
    fn decode(&self, s: &str, readings: &Readings) -> String {
        let mut out = String::new();
        let mut i = 0;
        while i < s.len() {
            let start = i;
            let mut parts = Vec::new();
            while let Some(g) = emoji_at(&s[i..]) {
                parts.push(g);
                i += g.len();
            }
            if parts.is_empty() {
                let c = s[i..].chars().next().unwrap();
                out.push(c);
                i += c.len_utf8();
                continue;
            }
            let (mut words, mut j, mut compound, mut unknown) = (Vec::new(), 0, false, false);
            while j < parts.len() {
                let max = parts.len().min(j + self.longest);
                let found = (j + 2..=max).rev().find_map(|end| {
                    Self::read(readings, &parts[j..end].concat()).map(|r| (r, end))
                });
                if let Some((word, end)) = found {
                    words.push(word);
                    compound = true;
                    j = end;
                } else {
                    let reading = Self::read(readings, parts[j]);
                    unknown |= reading.is_none();
                    words.push(reading.unwrap_or(parts[j]));
                    j += 1;
                }
            }
            if unknown && !compound {
                out.push_str(&s[start..i]);
            } else {
                let space = compound || words.iter().any(|w| !w.chars().all(char::is_alphabetic));
                out.push_str(&words.join(if space { " " } else { "" }));
            }
        }
        out
    }
    pub fn translate(&self, mode: Mode, s: &str, raw: bool, max: usize) -> Result<String> {
        match (raw, mode, &self.edits, &self.readings) {
            (false, Mode::Encode, Some(edits), _) => return self.encode(s, edits, max.min(16)),
            (false, Mode::Decode, _, Some(readings)) => return Ok(self.decode(s, readings)),
            _ => {}
        }
        let task = match mode {
            Mode::Encode => ENCODE,
            Mode::Decode => DECODE,
            Mode::Rebus => REBUS,
            Mode::Emojify => EMOJIFY,
        };
        let mut ids = vec![BOS, task];
        ids.extend(self.tok.encode(s));
        ids.push(SEP);
        let mut answers = Vec::new();
        let mut current = Vec::new();
        for _ in 0..max {
            let h = self.model.hidden(&ids)?.narrow(0, ids.len() - 1, 1)?;
            let logits = self.model.logits(&h)?.squeeze(0)?.to_vec1::<f32>()?;
            let mut best = f32::NEG_INFINITY;
            let mut next = EOS;
            for (i, &score) in logits.iter().take(self.tok.used()).enumerate() {
                let id = i as u32;
                if (BOS..=EMOJIFY).contains(&id) && id != EOS && !(task == REBUS && id == ALT) {
                    continue;
                }
                if score > best {
                    next = id;
                    best = score;
                }
            }
            if next == EOS {
                break;
            }
            if next == ALT {
                answers.push(self.tok.decode(&current));
                current.clear();
            } else {
                current.push(next);
            }
            if ids.len() == self.model.cfg.ctx {
                break;
            }
            ids.push(next);
        }
        answers.push(self.tok.decode(&current));
        let mut seen = std::collections::HashSet::new();
        answers.retain(|s| !s.is_empty() && seen.insert(s.clone()));
        Ok(answers.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_and_multiline_brackets_stay_literal() {
        let text = "🍃 [] [[no]] [no\nagain] [train] [book]";
        let marked: Vec<_> = brackets(text).iter().map(|&(a, b)| &text[a..b]).collect();
        assert_eq!(marked, ["[train]", "[book]"]);
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    #[test]
    #[ignore = "set EMOJILM_TEST_MODEL to the released noun/verb model.pt"]
    fn released_checkpoint_translates_in_context() -> Result<()> {
        let path = std::env::var_os("EMOJILM_TEST_MODEL").context("set EMOJILM_TEST_MODEL")?;
        let app = Translator::open(Path::new(&path))?;
        for (mode, source, expected) in [
            (
                Mode::Encode,
                "I usually [train] before I catch the [train].",
                "I usually [train] before I catch the 🚆.",
            ),
            (
                Mode::Encode,
                "I paired the [mouse] with my new laptop.",
                "I paired the 🖱 with my new laptop.",
            ),
            (
                Mode::Decode,
                "I baked 🍎🥧 yesterday.",
                "I baked apple pie yesterday.",
            ),
        ] {
            assert_eq!(app.translate(mode, source, false, 64)?, expected);
        }
        Ok(())
    }
}
