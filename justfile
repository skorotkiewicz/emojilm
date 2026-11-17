python := env_var_or_default("EMOJILM_PYTHON", ".venv/bin/python")

[private]
default:
    @just --list

setup:
    python3 -m venv .venv
    "{{python}}" -m pip install -r requirements.txt

data *args:
    "{{python}}" -m emojilm.corpora {{args}}

data-full cldr="data/raw/cldr_en.json" t2e="data/raw/text2emoji.jsonl":
    "{{python}}" -m emojilm.corpora --cldr {{quote(cldr)}} --t2e {{quote(t2e)}}

train *args:
    "{{python}}" -m emojilm.train {{args}}

smoke:
    "{{python}}" -m emojilm.train --device cpu --threads 2 --steps 8 --batch 4 --eval-every 8 --out ckpt/smoke
    "{{python}}" -m emojilm.generate --engine model --ckpt ckpt/smoke/model.pt --text '🍑⛳' --max-new-tokens 4

eval *args:
    "{{python}}" -m emojilm.eval {{args}}

test:
    "{{python}}" -m unittest discover -s tests -v
