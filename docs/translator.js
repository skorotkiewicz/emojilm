import { emojiAt, Tokenizer } from "./tokenizer.js";

const canonical = (s) =>
  s.replace(/[\uFE0E\uFE0F]/gu, "").trim().replace(/\s+/gu, " ").toLowerCase();
const dot = (w, x) => w.reduce((sum, v, i) => sum + v * x[i], 0);
const firstMax = (xs) => xs.indexOf(Math.max(...xs));
const cueWords = [
  "a an the this that these those my your his her its our their each every another some",
  "i you he she it we they",
  "can could will would may might must shall should do does did",
  "to",
].map((g) => new Set(g.split(" ")));
const wordParts = (s) =>
  s.toLowerCase().match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) ?? [];
function* brackets(s) {
  let depth = 0, start = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "[" && depth++ === 0) start = i;
    if (
      s[i] === "]" && depth && --depth === 0 && i > start + 1 &&
      !/[\[\]\r\n]/.test(s.slice(start + 1, i))
    ) yield [start, i + 1];
  }
}

export class Translator {
  constructor(bundle, forward) {
    Object.assign(this, bundle);
    this.tok = new Tokenizer(bundle.tokenizer);
    this.forward = forward;
    this.longest = Math.max(
      ...Object.keys(this.decode_readings).map((s) => [...s].length),
    );
  }
  async noun(s, start, end, label) {
    const middle = this.tok.encode("[" + label + "]"),
      slots = this.cfg.ctx - middle.length - 1;
    const left = [
      256,
      ...this.tok.encode(s.slice(0, start).trimEnd()).slice(
        -Math.min(24, slots),
      ),
    ];
    const right = this.tok.encode(s.slice(end)).slice(
      0,
      Math.min(12, slots - left.length + 1),
    );
    const ids = [...left, ...middle, ...right],
      h = await this.forward(ids, "hidden"),
      d = this.cfg.dim;
    const x = [left.length - 1, left.length + middle.length - 1, ids.length - 1]
      .flatMap((i) => Array.from(h.subarray(i * d, (i + 1) * d)));
    for (
      const word of [
        ...["", "", ...wordParts(s.slice(0, start))].slice(-2),
        wordParts(s.slice(end))[0] ?? "",
      ]
    ) {
      for (const group of cueWords) x.push(group.has(word) ? Math.sqrt(d) : 0);
    }
    return dot(this.noun_verb_gate.weight, x) + this.noun_verb_gate.bias > 0;
  }
  async choose(source, index, start, end) {
    const word = source.slice(start + 1, end - 1), label = canonical(word);
    let choices = Object.hasOwn(this.edit_candidates, label)
      ? this.edit_candidates[label]
      : [];
    if (!choices.length) {
      return label === "think" ? "🤔" : label === "eat" ? "🍽️" : null;
    }
    if (this.noun_verb_gate?.labels.includes(label)) {
      if (!await this.noun(source, start, end, label)) return null;
      choices = choices.filter((v) => v !== "KEEP");
    }
    if (choices.length <= 1) return choices[0];
    const prefix = [
      256,
      260,
      264,
      ...this.tok.encode(source + "\n" + (index + 1) + " [" + word + "]"),
      262,
    ];
    const limit = Math.min(16, this.cfg.ctx - prefix.length + 1);
    const targets = choices.map((v) => [...this.tok.encode(v), 257]);
    const head = this.sense_heads?.[label];
    if (
      head && head.choices?.join("|") === choices.join("|") &&
      targets.every((t) => t.length <= limit)
    ) {
      const h = (await this.forward(prefix, "hidden")).slice(-this.cfg.dim);
      return choices[
        firstMax(head.weight.map((w, i) => dot(w, h) + head.bias[i]))
      ];
    }
    let best = -Infinity, answer = null;
    for (const [i, target] of targets.entries()) {
      if (target.length > limit) continue;
      const logits = await this.forward(
        [...prefix, ...target.slice(0, -1)],
        "logits",
      );
      let score = 0;
      target.forEach((id, j) => {
        const offset = (prefix.length - 1 + j) * this.cfg.vocab,
          row = logits.subarray(offset, offset + this.cfg.vocab);
        const peak = Math.max(...row);
        score += row[id] - peak -
          Math.log(row.reduce((a, x) => a + Math.exp(x - peak), 0));
      });
      if (score > best) {
        best = score;
        answer = choices[i];
      }
    }
    return answer;
  }
  async encode(s) {
    let out = "", last = 0, index = 0;
    for (const [start, end] of brackets(s)) {
      const replacement = await this.choose(s, index++, start, end);
      out += s.slice(last, start) +
        (replacement && replacement !== "KEEP"
          ? replacement
          : s.slice(start, end));
      last = end;
    }
    return out + s.slice(last);
  }
  decode(s) {
    const read = (g) =>
      this.decode_readings[canonical(g)]?.[0] ??
        ({ "🤔": "thinking face", "🍽": "fork and knife with plate" })[
          canonical(g)
        ];
    let out = "";
    for (let i = 0; i < s.length;) {
      const start = i, parts = [];
      for (let g; (g = emojiAt(s, i));) {
        parts.push(g);
        i += g.length;
      }
      if (!parts.length) {
        const c = String.fromCodePoint(s.codePointAt(i));
        out += c;
        i += c.length;
        continue;
      }
      const words = [];
      let compound = false, unknown = false;
      for (let j = 0; j < parts.length;) {
        let word, end = Math.min(j + this.longest, parts.length);
        for (; end > j + 1; end--) {
          if ((word = read(parts.slice(j, end).join("")))) break;
        }
        if (word) {
          compound = true;
          words.push(word);
          j = end;
        } else {
          word = read(parts[j]);
          unknown ||= !word;
          words.push(word ?? parts[j]);
          j++;
        }
      }
      out += unknown && !compound ? s.slice(start, i) : words.join(
        compound || words.some((w) => !/^\p{L}+$/u.test(w)) ? " " : "",
      );
    }
    return out;
  }
  async translate(mode, s) {
    if (mode === "encode") return this.encode(s);
    if (mode === "decode") return this.decode(s);
    const ids = [256, mode === "rebus" ? 261 : 265, ...this.tok.encode(s), 262];
    if (ids.length > this.cfg.ctx) {
      throw Error(
        "This sentence exceeds the model’s 256-token context. Try a shorter one.",
      );
    }
    const answers = [];
    let current = [];
    for (let i = 0; i < 64 && ids.length <= this.cfg.ctx; i++) {
      const row = (await this.forward(ids, "logits")).slice(-this.cfg.vocab);
      let best = -Infinity, next = 257;
      for (let j = 0; j < this.tok.tokens.length; j++) {
        if (
          j >= 256 && j <= 265 && j !== 257 && !(mode === "rebus" && j === 263)
        ) continue;
        if (row[j] > best) {
          best = row[j];
          next = j;
        }
      }
      if (next === 257) break;
      if (next === 263) {
        answers.push(this.tok.decode(current));
        current = [];
      } else current.push(next);
      ids.push(next);
    }
    answers.push(this.tok.decode(current));
    return [...new Set(answers.filter(Boolean))].join("\n");
  }
}
