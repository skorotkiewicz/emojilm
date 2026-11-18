python := env_var_or_default("EMOJILM_PYTHON", ".venv/bin/python")

[private]
default:
    @just --list

setup:
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

data:
    "{{python}}" -m emojilm.corpora --fetch-missing --composition --data data/composition
    "{{python}}" -m emojilm.story_data
    "{{python}}" -m emojilm.story_data --long
    "{{python}}" -m emojilm.edit_data
    "{{python}}" -m emojilm.edit_coverage

train *args:
    "{{python}}" -m emojilm.train --data data/encode-edits-expanded --out ckpt/final --composition-val data/encode-edits-expanded/edit-val.jsonl --steps 12000 --bundle-edits {{args}}

smoke:
    "{{python}}" -m emojilm.corpora --data data/smoke
    "{{python}}" -m emojilm.train --data data/smoke --device cpu --threads 2 --steps 8 --batch 4 --eval-every 8 --out ckpt/smoke
    "{{python}}" -m emojilm.generate --engine model --ckpt ckpt/smoke/model.pt --text '🍑⛳' --max-new-tokens 4

eval *args:
    "{{python}}" -m emojilm.eval --engine model-edit --ckpt ckpt/final/model.pt --data data/encode-edits-expanded/long-val.jsonl --copy-metrics {{args}}

test:
    "{{python}}" -m unittest discover -s tests -v
