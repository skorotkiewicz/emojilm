"""Build a reproducible mixture of rebuses, emoji names, and emoji summaries."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re

from . import data
from .rebus_bank import (
    BANK,
    CARRIERS,
    CUSTOM,
    EXTRA_LEXICON,
    EXTRA_SINGLES,
    HOLDOUT_BANK,
)
from .rules import load_lexicon
from .tokenizer import COMMON_EMOJI, KEYCAPS, MODIFIERS, SPECIALS, Tokenizer, emoji_at


def example(task, src, tgt, tag):
    return {"task": task, "src": src, "tgt": tgt, "tag": tag}


def hint_dictionary():
    lexicon = {emoji: list(words) for emoji, words in data.LEXICON.items()}
    for emoji, words in EXTRA_LEXICON.items():
        lexicon[emoji] = list(dict.fromkeys(lexicon.get(emoji, []) + words))
    return lexicon


def hint_for(source, lexicon):
    return ";".join(
        emoji + ":" + "|".join(lexicon[emoji])
        for emoji in data.emojis_in(source)
        if emoji in lexicon
    )


def qwen_bank(entries, tag):
    for emoji, morph, phrase, category in entries:
        bracketed = " ".join(f"[{word}]" for word in morph)
        for frame in CUSTOM.get(emoji, CARRIERS[category]):
            yield example("decode", frame.format(emoji), frame.format(phrase), tag)
            yield example("encode", frame.format(bracketed), frame.format(emoji), tag)
            if tag != "holdout":
                yield example(
                    "encode", frame.format(f"[{phrase}]"), frame.format(emoji), tag
                )
        yield example("rebus", emoji, phrase, tag)


def qwen_singles():
    for emoji, readings in EXTRA_SINGLES:
        for meaning, frames in readings:
            for frame in frames:
                yield example(
                    "decode", frame.format(emoji), frame.format(meaning), "single"
                )
                yield example(
                    "encode",
                    frame.format(f"[{meaning}]"),
                    frame.format(emoji),
                    "single",
                )
        yield example(
            "rebus", emoji, " <alt> ".join(meaning for meaning, _ in readings), "naming"
        )


def gpt_examples(entries):
    for entry in entries:
        text, emoji = entry["text"], entry["emoji"]
        bracket, spaced = entry["bracket"], entry["emoji_spaced"]
        yield example("rebus", emoji, text, "gpt")
        for frame in ["{x}", *entry.get("frames", [])]:
            yield example("decode", frame.format(x=emoji), frame.format(x=text), "gpt")
            yield example("decode", frame.format(x=spaced), frame.format(x=text), "gpt")
            yield example("encode", frame.format(x=text), frame.format(x=emoji), "gpt")
            yield example(
                "encode", frame.format(x=bracket), frame.format(x=spaced), "gpt"
            )
        if entry["category"] == "compound":
            yield example("decode", "Meaning: " + emoji, "Meaning: " + text, "gpt")
            yield example("encode", "Emoji for " + text, "Emoji for " + emoji, "gpt")


def canonical(text):
    return " ".join(text.translate({0xFE0E: None, 0xFE0F: None}).split()).casefold()


def emoji_runs(text):
    """Yield adjacent or space-separated emoji runs with presentation selectors removed."""
    if text.isascii():
        return
    text = text.replace("\ufe0e", "")
    run = []
    i = 0
    while i < len(text):
        emoji = emoji_at(text, i)
        if emoji:
            run.append(canonical(emoji))
            i += len(emoji)
            continue
        if run and not text[i].isspace():
            yield "".join(run)
            run.clear()
        i += 1
    if run:
        yield "".join(run)


def reserved_filter(holdout):
    rebuses = [row for row in holdout if row["task"] == "rebus"]
    combinations = tuple("".join(emoji_runs(row["src"])) for row in rebuses)
    phrases = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(canonical(row["tgt"])) for row in rebuses)
        + r")(?!\w)"
    )

    bracket = re.compile(r"\[[^\[\]]+\]")
    decompositions = {
        tuple(canonical(part) for part in bracket.findall(row["src"]))
        for row in holdout
        if row["task"] == "encode"
    }

    def reserved(row):
        for text in (row["src"], row["tgt"]):
            if phrases.search(canonical(text)):
                return True
            if "[" in text:
                parts = tuple(canonical(part) for part in bracket.findall(text))
                if any(
                    parts[i : i + len(group)] == group
                    for group in decompositions
                    for i in range(len(parts) - len(group) + 1)
                ):
                    return True
            if any(combo in run for run in emoji_runs(text) for combo in combinations):
                return True
        return False

    return reserved


def cldr_examples(path, limit, rng):
    if path is None or limit == 0:
        return
    annotations = json.loads(Path(path).read_text(encoding="utf-8"))
    items = list(annotations["annotations"]["annotations"].items())
    rng.shuffle(items)
    for emoji, annotation in items[:limit]:
        names = annotation.get("tts") or annotation.get("keywords") or []
        if names and names[0].strip():
            name = names[0].strip()
            yield example("rebus", emoji, name, "cldr")
            yield example("decode", f"I saw the {emoji}.", f"I saw the {name}.", "cldr")


def text2emoji_examples(path, limit, rng, reserved):
    if path is None or limit == 0:
        return []
    sample = []
    eligible = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            text = " ".join((record.get("text") or "").split())
            emoji = (record.get("emoji") or "").strip()
            if not text or len(text) > 120 or not emoji or len(emoji) > 120:
                continue
            symbols = data.emojis_in(emoji)
            if not 1 <= len(symbols) <= 12 or "".join(symbols) != "".join(
                emoji.split()
            ):
                continue
            row = example("emojify", text, emoji, "t2e")
            if reserved(row):
                continue
            eligible += 1
            index = eligible - 1 if eligible <= limit else rng.randrange(eligible)
            if index < len(sample):
                sample[index] = row
            elif index < limit:
                sample.append(row)
    return sample


def merge_targets(rows):
    """Use the first curated translation and collect possible rebus readings."""
    merged = {}
    for row in rows:
        key = row["task"], row["src"]
        if key not in merged:
            merged[key] = row
        elif row["task"] == "rebus":
            first = merged[key]
            readings = first["tgt"].split(" <alt> ") + row["tgt"].split(" <alt> ")
            merged[key] = {**first, "tgt": " <alt> ".join(dict.fromkeys(readings))}
    return list(merged.values())


def group_key(text):
    # Spaced/unspaced emoji runs and case variants belong to the same group.
    return "".join(canonical(text).split())


def split_rows(rows, rng):
    # Keep hinted inputs and reversible encode/decode pairs in the same split.
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for row in rows:
        if row["task"] in {"decode", "encode"}:
            source, target = group_key(row["src"]), group_key(row["tgt"])
            parents[root(target)] = root(source)

    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["task"], row["src"], row["tgt"], row.get("hint")
        if key not in seen:
            groups[root(group_key(row["src"]))].append(row)
            seen.add(key)

    train, val = [], []
    keys = list(groups)
    rng.shuffle(keys)
    val_size = max(1, round(len(keys) * 0.04))
    for i, key in enumerate(keys):
        (val if i < val_size else train).extend(groups[key])
    return train, val


def build_mix(
    outdir="data",
    cldr_path=None,
    t2e_path=None,
    t2e_limit=12000,
    cldr_limit=1800,
    core_repeat=3,
    bpe_sample=6000,
    seed=7,
    size=4096,
):
    if min(t2e_limit, cldr_limit) < 0 or min(core_repeat, bpe_sample) < 1:
        raise ValueError(
            "Corpus limits must be non-negative and sampling/repeat counts positive"
        )
    rng = random.Random(seed)
    holdout = data.gen_holdout() + list(qwen_bank(HOLDOUT_BANK, "holdout"))
    reserved = reserved_filter(holdout)
    core = data.gen_examples()
    core += list(qwen_bank(BANK, "rebus")) + list(qwen_singles())
    gpt = [
        entry
        for entry in load_lexicon()
        if not reserved(example("rebus", entry["emoji"], entry["text"], "gpt"))
    ]
    core += list(gpt_examples(gpt))
    cldr = [
        row for row in cldr_examples(cldr_path, cldr_limit, rng) if not reserved(row)
    ]
    curated = merge_targets([row for row in core if not reserved(row)] + cldr)
    dictionary = hint_dictionary()
    hinted = []
    for row in curated:
        hint = hint_for(row["src"], dictionary)
        if row["tag"] != "cldr" and hint and rng.random() < 0.5:
            hinted.append({**row, "hint": hint})

    t2e = text2emoji_examples(t2e_path, t2e_limit, rng, reserved)
    train, val = split_rows(curated + hinted + t2e, rng)

    texts = [row[field] for row in train for field in ("src", "tgt")]
    texts += [row["hint"] for row in train if "hint" in row]
    rng.shuffle(texts)
    inventory = {}
    for row in train:
        for field in ("src", "tgt"):
            inventory.update(dict.fromkeys(data.emojis_in(row[field])))
    base = list(dict.fromkeys(COMMON_EMOJI + KEYCAPS))
    emoji_budget = max(
        0, size - 256 - len(SPECIALS) - len(MODIFIERS) - len(base) - 1200
    )
    tok = Tokenizer.train(
        texts[:bpe_sample], base + list(inventory)[:emoji_budget], size
    )

    train = [
        row
        for row in train
        for _ in range(1 if row["tag"] in {"cldr", "t2e"} else core_repeat)
    ]
    rng.shuffle(train)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tok.save(outdir / "tokenizer.json")
    for name, rows in (("train", train), ("val", val), ("holdout", holdout)):
        with (outdir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        name: len(rows)
        for name, rows in (
            ("train", train),
            ("val", val),
            ("holdout", holdout),
            ("cldr", cldr),
            ("t2e", t2e),
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--cldr", type=Path, help="CLDR English annotations JSON")
    parser.add_argument(
        "--t2e", type=Path, help="Local Text2Emoji JSONL with text and emoji fields"
    )
    parser.add_argument("--t2e-limit", type=int, default=12000)
    parser.add_argument("--cldr-limit", type=int, default=1800)
    parser.add_argument("--core-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        stats = build_mix(
            args.data,
            args.cldr,
            args.t2e,
            args.t2e_limit,
            args.cldr_limit,
            args.core_repeat,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(" ".join(f"{name}={count}" for name, count in stats.items()))


if __name__ == "__main__":
    main()
