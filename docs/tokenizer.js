const specials = [
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
const modifiers = ["\uFE0F", "\u200D", "\u20E3", ..."🏻🏼🏽🏾🏿"];
const encoder = new TextEncoder();
const decoder = new TextDecoder();
const emoji =
  /(?:[#*0-9]\uFE0F?\u20E3|\p{Regional_Indicator}{2}|\p{Extended_Pictographic}[\uFE0E\uFE0F]?\p{Emoji_Modifier}*(?:\u200D\p{Extended_Pictographic}[\uFE0E\uFE0F]?\p{Emoji_Modifier}*)*[\u{E0020}-\u{E007F}]*)/uy;
export function emojiAt(text, index) {
  emoji.lastIndex = index;
  return emoji.exec(text)?.[0];
}

export class Tokenizer {
  constructor(spec) {
    this.tokens = [
      ...Array(256),
      ...specials,
      ...modifiers,
      ...spec.emoji,
      ...spec.text_tokens,
    ];
    this.ids = new Map(this.tokens.map((s, i) => [s, i]));
    this.rank = new Map(
      spec.merges.map(([a, b], i) => [JSON.stringify([a, b]), i]),
    );
    this.known = new Map();
    for (
      const glyph of [...spec.emoji, ...modifiers].sort((a, b) =>
        b.length - a.length
      )
    ) {
      const first = glyph.codePointAt(0);
      if (!this.known.has(first)) this.known.set(first, []);
      this.known.get(first).push(glyph);
    }
  }
  encode(text) {
    const ids = [];
    const emit = (s) =>
      this.ids.has(s)
        ? ids.push(this.ids.get(s))
        : ids.push(...encoder.encode(s));
    const flush = (text) => {
      for (const [n, run] of text.split("▁").entries()) {
        if (n) ids.push(...encoder.encode("▁"));
        for (
          const [piece] of run.matchAll(
            / ?(?:[A-Za-z]+|[0-9]|[^A-Za-z0-9\s▁])|./gsu,
          )
        ) {
          const units = [...piece.replace(/^ /, "▁")];
          while (units.length > 1) {
            let best = Infinity, at = -1;
            for (let i = 0; i + 1 < units.length; i++) {
              const rank = this.rank.get(
                JSON.stringify([units[i], units[i + 1]]),
              );
              if (rank < best) {
                best = rank;
                at = i;
              }
            }
            if (at < 0) break;
            units.splice(at, 2, units[at] + units[at + 1]);
          }
          units.forEach(emit);
        }
      }
    };
    let start = 0;
    for (let i = 0; i < text.length;) {
      const cp = text.codePointAt(i);
      const glyph = this.known.get(cp)?.find((g) => text.startsWith(g, i)) ??
        emojiAt(text, i);
      if (!glyph) {
        i += cp > 65535 ? 2 : 1;
        continue;
      }
      flush(text.slice(start, i));
      if (this.ids.has(glyph)) emit(glyph);
      else {glyph.split("\u200D").forEach((part, j) => {
          if (j) emit("\u200D");
          if (this.ids.has(part)) emit(part);
          else [...part].forEach(emit);
        });}
      i += glyph.length;
      start = i;
    }
    flush(text.slice(start));
    return ids;
  }
  decode(ids) {
    const bytes = [], parts = [];
    const flush = () => {
      parts.push(decoder.decode(Uint8Array.from(bytes)));
      bytes.length = 0;
    };
    for (const id of ids) {
      if (id < 256) {
        bytes.push(id);
        continue;
      }
      flush();
      if (id === 263) parts.push(" / ");
      if (id >= 266) parts.push(this.tokens[id].replaceAll("▁", " "));
    }
    flush();
    return parts.join("");
  }
}
