"""Build paired emoji edits from TinyStories contexts."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

from .composition import default_composer
from .config import Config
from .corpora import canonical, group_key, reserved_filter
from .data import emojis_in, load_jsonl
from .fetch_data import fetch_source
from .tokenizer import Tokenizer

DEFAULT_STORIES = Path("data/raw/TinyStoriesV2-GPT4-valid.txt")
STORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)
STORIES_SHA256 = "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"
STORY_SEPARATOR = "<|endoftext|>"
SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+|(?<=[.!?]")\s+|(?<=[.!?]”)\s+')
BARE_LABEL = re.compile(r"[a-z]+(?: [a-z]+)*")
LITERAL_WORD = re.compile(r"\b[a-z]{5,}\b")


def cache_verified_stories(source, destination):
    with source.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != STORIES_SHA256:
        raise ValueError("TinyStories download checksum did not match")
    source.replace(destination)


def read_stories(path):
    with path.open(encoding="utf-8") as stream:
        parts = []
        for line in stream:
            chunks = line.split(STORY_SEPARATOR)
            parts.append(chunks[0])
            for chunk in chunks[1:]:
                yield "".join(parts).strip()
                parts = [chunk]
        if parts:
            yield "".join(parts).strip()


def known_mappings(rows):
    candidates = defaultdict(dict)
    for row in rows:
        if row["task"] != "rebus":
            continue
        emoji = row["src"]
        word = row["tgt"].split(" <alt> ", 1)[0]
        if len(word) < 3 or not BARE_LABEL.fullmatch(word):
            continue
        if not emoji or "".join(emojis_in(emoji)) != emoji:
            continue
        parts = row.get("target_parts", [word])
        brackets = " ".join(f"[{part}]" for part in parts)
        candidates[word][emoji] = brackets
    composer = default_composer()
    mappings = {}
    for word, variants in candidates.items():
        if len(variants) == 1:
            mappings[word] = next(iter(variants.items()))
            continue
        preferred = composer.write_component(word)
        if preferred in variants:
            mappings[word] = preferred, variants[preferred]
        elif preferred:
            for emoji, brackets in variants.items():
                if canonical(emoji) == canonical(preferred):
                    mappings[word] = emoji, brackets
                    break
    return mappings


def story_pair(sentence, matches, mappings, story_id, *, long=False, literal=None):
    edits = [(m.start(), m.end(), *mappings[m[0]], m[0]) for m in matches]
    if literal is not None:
        marked = f"[{literal[0]}]"
        edits.append((literal.start(), literal.end(), marked, marked, marked))
    emoji_parts, marker_parts, decoded_parts, offset = [], [], [], 0
    for start, end, emoji, markers, word in sorted(edits):
        unchanged = sentence[offset:start]
        emoji_parts.extend((unchanged, emoji))
        marker_parts.extend((unchanged, markers))
        decoded_parts.extend((unchanged, word))
        offset = end
    emoji_parts.append(sentence[offset:])
    marker_parts.append(sentence[offset:])
    decoded_parts.append(sentence[offset:])
    encoded, marked, decoded = map("".join, (emoji_parts, marker_parts, decoded_parts))
    metadata = {
        "tag": "tinystories-long" if long else "tinystories",
        "story_id": story_id,
    }
    return [
        {"task": "decode", "src": encoded, "tgt": decoded, **metadata},
        {"task": "encode", "src": marked, "tgt": encoded, **metadata},
    ]


def long_contexts(story):
    sentences = SENTENCE_BREAK.split(" ".join(story.split()))
    return [
        " ".join(sentences[start : start + count])
        for start in range(len(sentences))
        for count in range(1, min(3, len(sentences) - start) + 1)
    ]


def sample_pairs(
    stories,
    mappings,
    tok,
    seen,
    reserved,
    train_limit,
    validation_limit,
    seed,
    *,
    long=False,
    used_stories=(),
):
    if long:
        mappings = {
            word: pair for word, pair in mappings.items() if pair[1].count("[") == 1
        }
    if not mappings:
        raise ValueError(
            "The base training split has no unambiguous component readings"
        )
    words = sorted(mappings, key=lambda word: (-len(word), word))
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, words)) + r")(?!\w)")
    rng = random.Random(seed)
    train, validation = [], []
    used_stories = set(used_stories)
    ctx = Config().ctx
    for story in stories:
        if len(train) + 2 > train_limit and len(validation) + 2 > validation_limit:
            break
        if not story:
            continue
        digest = sha256(" ".join(story.split()).encode("utf-8")).digest()
        story_id = digest.hex()[:20]
        if story_id in used_stories:
            continue
        used_stories.add(story_id)
        split, limit = (
            (validation, validation_limit) if digest[0] < 51 else (train, train_limit)
        )
        if len(split) + 2 > limit:
            continue

        sentences = long_contexts(story) if long else SENTENCE_BREAK.split(story)
        rng.shuffle(sentences)
        required_spans = None
        if long:
            sentences.sort(key=len, reverse=True)
            if split is validation:
                required_spans = 2 + len(validation) // 2 % 5
        copy = not long and rng.randrange(8) == 0
        for sentence in sentences:
            sentence = sentence.strip()
            key = group_key(sentence)
            if (
                key in seen
                or not (80 if long else 12) <= len(sentence) <= 180
                or len(sentence.split()) < 3
                or any(mark in sentence for mark in "[]\n\r")
                or emojis_in(sentence)
            ):
                continue
            matches = [] if copy else list(pattern.finditer(sentence))
            literal = None
            if long:
                if len(matches) < 2:
                    continue
                if required_spans != 2 and rng.randrange(4) == 0:
                    candidates = [
                        word
                        for word in LITERAL_WORD.finditer(sentence)
                        if word[0] not in mappings
                        and not any(
                            m.start() < word.end() and word.start() < m.end()
                            for m in matches
                        )
                    ]
                    if candidates:
                        literal = rng.choice(candidates)
                max_edits = min(6 - (literal is not None), len(matches))
                if required_spans is None:
                    count = rng.randint(max(2, max_edits - 1), max_edits)
                else:
                    count = required_spans - (literal is not None)
                    if count > max_edits:
                        continue
                matches = rng.sample(matches, count)
            elif not copy:
                if not matches:
                    continue
                first = rng.choice(matches)
                others = [
                    match
                    for match in matches
                    if match != first
                    and sentence[
                        min(match.end(), first.end()) : max(
                            match.start(), first.start()
                        )
                    ].strip()
                ]
                matches = (
                    [first, rng.choice(others)]
                    if others and rng.random() < 0.5
                    else [first]
                )
            rows = story_pair(
                sentence, matches, mappings, story_id, long=long, literal=literal
            )
            if any(
                reserved(row)
                or len(tok.build(row["task"], row["src"], row["tgt"])[0]) - 1 > ctx
                or group_key(row["src"]) in seen
                for row in rows
            ):
                continue
            split.extend(rows)
            seen.update(
                group_key(row[field]) for row in rows for field in ("src", "tgt")
            )
            break
    return train, validation


def build_experiment(
    base, stories, output, *, train_limit=5000, validation_limit=120, seed=7, long=False
):
    base, stories, output = Path(base), Path(stories), Path(output)
    if base.resolve() == output.resolve():
        raise ValueError("Choose an output directory separate from the base dataset")
    if min(train_limit, validation_limit) < 2:
        raise ValueError("Each split needs room for at least one encode/decode pair")
    base_rows = load_jsonl(base / "train.jsonl")
    base_val = load_jsonl(base / "val.jsonl")
    fixed_eval = load_jsonl(base / "holdout.jsonl") + load_jsonl(
        base / "composition-val.jsonl"
    )
    context_val = load_jsonl(base / "context-val.jsonl") if long else []
    seen = {
        group_key(row[field])
        for row in chain(base_rows, base_val, fixed_eval, context_val)
        for field in ("src", "tgt")
    }
    additions, validation = sample_pairs(
        read_stories(stories),
        known_mappings(base_rows),
        Tokenizer.load(base / "tokenizer.json"),
        seen,
        reserved_filter(fixed_eval),
        train_limit,
        validation_limit,
        seed,
        long=long,
        used_stories=(
            r["story_id"] for r in chain(base_rows, context_val) if "story_id" in r
        ),
    )
    if any(
        not any(row["src"] != row["tgt"] for row in split)
        for split in (additions, validation)
    ):
        raise ValueError(
            "The story file yielded too few matching stories for both splits"
        )
    if long and len(validation) < validation_limit - 1:
        raise ValueError("Too few stories to fill the long-context holdout")

    output.mkdir(parents=True, exist_ok=True)
    evaluation = "long-val.jsonl" if long else "context-val.jsonl"
    with tempfile.TemporaryDirectory(dir=output) as directory:
        staged = Path(directory)
        for filename in (
            "tokenizer.json",
            "val.jsonl",
            "composition-val.jsonl",
            "holdout.jsonl",
        ):
            shutil.copyfile(base / filename, staged / filename)
        if long:
            shutil.copyfile(base / "context-val.jsonl", staged / "context-val.jsonl")
        for filename, rows in (
            ("train.jsonl", chain(base_rows, additions)),
            (evaluation, validation),
        ):
            with (staged / filename).open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for path in staged.iterdir():
            path.replace(output / path.name)
    return len(additions), len(validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--long", action="store_true", help="Add edits in longer contexts"
    )
    parser.add_argument("--stories", type=Path, default=DEFAULT_STORIES)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--validation-limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.base = args.base or Path(
        "data/tinystories" if args.long else "data/composition"
    )
    args.out = args.out or Path(
        "data/tinystories-long" if args.long else "data/tinystories"
    )
    if not (args.base / "train.jsonl").is_file():
        parser.error(
            f"Missing {args.base / 'train.jsonl'}. Run just data to prepare the default dataset."
        )
    try:
        if args.stories == DEFAULT_STORIES:
            fetch_source(STORIES_URL, args.stories, convert=cache_verified_stories)
        train, validation = build_experiment(
            args.base,
            args.stories,
            args.out,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            seed=args.seed,
            long=args.long,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    name = "TinyStories long contexts" if args.long else "TinyStories"
    print(f"{name} train={train} context_val={validation} -> {args.out}")


if __name__ == "__main__":
    main()
