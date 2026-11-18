python := env_var_or_default("EMOJILM_PYTHON", ".venv/bin/python")

[private]
default:
    @just --list

setup:
    python3 -m venv .venv
    "{{python}}" -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

data *args:
    "{{python}}" -m emojilm.corpora {{args}}

data-full cldr="data/raw/cldr_en.json" t2e="data/raw/text2emoji.jsonl":
    "{{python}}" -m emojilm.corpora --fetch-missing --cldr {{quote(cldr)}} --t2e {{quote(t2e)}}

data-composition:
    "{{python}}" -m emojilm.corpora --fetch-missing --composition --data data/composition

data-tinystories *args:
    "{{python}}" -m emojilm.story_data {{args}}

data-tinystories-long *args:
    "{{python}}" -m emojilm.story_data --long {{args}}

data-encode-edits *args:
    "{{python}}" -m emojilm.edit_data {{args}}

data-encode-edits-expanded *args:
    "{{python}}" -m emojilm.edit_coverage {{args}}

train *args:
    "{{python}}" -m emojilm.train {{args}}

train-composition *args:
    "{{python}}" -m emojilm.train --data data/composition --out ckpt/composition --composition-val data/composition/composition-val.jsonl --steps 12000 {{args}}

train-tinystories *args:
    "{{python}}" -m emojilm.train --data data/tinystories --out ckpt/tinystories --composition-val data/tinystories/composition-val.jsonl --steps 12000 {{args}}

train-tinystories-long *args:
    "{{python}}" -m emojilm.train --data data/tinystories-long --out ckpt/tinystories-long --composition-val data/tinystories-long/composition-val.jsonl --steps 12000 {{args}}

train-encode-edits *args:
    "{{python}}" -m emojilm.train --data data/encode-edits --out ckpt/encode-edits --composition-val data/encode-edits/edit-val.jsonl --steps 12000 {{args}}

train-encode-edits-expanded *args:
    "{{python}}" -m emojilm.train --data data/encode-edits-expanded --out ckpt/encode-edits-expanded --composition-val data/encode-edits-expanded/edit-val.jsonl --steps 12000 {{args}}

bundle-encode-edits *args:
    "{{python}}" -m emojilm.edit_candidates {{args}}

smoke:
    "{{python}}" -m emojilm.train --device cpu --threads 2 --steps 8 --batch 4 --eval-every 8 --out ckpt/smoke
    "{{python}}" -m emojilm.generate --engine model --ckpt ckpt/smoke/model.pt --text '🍑⛳' --max-new-tokens 4

eval *args:
    "{{python}}" -m emojilm.eval {{args}}

test:
    "{{python}}" -m unittest discover -s tests -v
