"""Fetch and cache the optional corpora for a full dataset build."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

CLDR_PATH = Path("data/raw/cldr_en.json")
T2E_PATH = Path("data/raw/text2emoji.jsonl")
CLDR_URL = (
    "https://raw.githubusercontent.com/unicode-org/cldr-json/main/"
    "cldr-json/cldr-annotations-full/annotations/en/annotations.json"
)
T2E_URL = "https://huggingface.co/datasets/KomeijiForce/Text2Emoji/resolve/main/text2emoji.csv"


def validate_cldr(source):
    with source.open(encoding="utf-8") as stream:
        data = json.load(stream)
    try:
        annotations = data["annotations"]["annotations"]
    except (KeyError, TypeError):
        annotations = None
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError("Downloaded CLDR data has no annotations")


def convert_text2emoji(source, destination):
    with source.open(encoding="utf-8-sig", newline="") as csv_stream:
        reader = csv.DictReader(csv_stream, strict=True)
        if not {"text", "emoji"}.issubset(reader.fieldnames or []):
            raise ValueError("Downloaded Text2Emoji CSV needs text and emoji columns")
        with destination.open("w", encoding="utf-8") as jsonl:
            count = 0
            for row in reader:
                if row["text"] is None or row["emoji"] is None:
                    raise ValueError(
                        f"Incomplete Text2Emoji CSV row at line {reader.line_num}"
                    )
                jsonl.write(
                    json.dumps(
                        {"text": row["text"], "emoji": row["emoji"]}, ensure_ascii=False
                    )
                    + "\n"
                )
                count += 1
        if not count:
            raise ValueError("Downloaded Text2Emoji CSV has no rows")


def fetch_source(url, destination, convert=None):
    if destination is None or destination.is_file():
        return
    print(f"Downloading {destination.name}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            payload = Path(directory) / "download"
            with urlopen(url, timeout=60) as response, payload.open("wb") as output:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, output)
            if expected is not None and payload.stat().st_size != int(expected):
                raise ValueError("Download ended before the advertised size")
            prepared = payload
            if convert:
                prepared = Path(directory) / destination.name
                convert(payload, prepared)
            else:
                validate_cldr(payload)
            prepared.replace(destination)
    except (OSError, ValueError, csv.Error) as error:
        raise ValueError(
            f"Could not prepare {destination}. {error}. "
            f"Download {url} or supply a local input file."
        ) from error


def fetch_missing(cldr, t2e, cldr_url=CLDR_URL, t2e_url=T2E_URL):
    fetch_source(cldr_url, cldr)
    fetch_source(t2e_url, t2e, convert=convert_text2emoji)
