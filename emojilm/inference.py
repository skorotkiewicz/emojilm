"""Load a bundled checkpoint and generate answers with PyTorch."""

import torch

from .config import Config
from .model import EmojiLM
from .tokenizer import Tokenizer


def load(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "emojilm-1":
        raise ValueError("Expected an EmojiLM checkpoint from this project")
    tok = Tokenizer.from_dict(checkpoint["tokenizer"])
    cfg = Config.from_dict(checkpoint["cfg"])
    if cfg.vocab != tok.size:
        raise ValueError("Checkpoint model and tokenizer vocabulary sizes differ")
    model = EmojiLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, task, source, hint=None, max_new=64):
    prompt, _ = tok.build(task, source, hint=hint)
    if len(prompt) > model.cfg.ctx:
        raise ValueError(
            f"Prompt needs {len(prompt)} tokens but the model context is {model.cfg.ctx}"
        )
    device = model.tok_emb.weight.device
    tokens = torch.empty((1, len(prompt) + max_new), dtype=torch.long, device=device)
    tokens[0, : len(prompt)] = torch.tensor(prompt, device=device)
    eos, alt = tok.sid["<eos>"], tok.sid["<alt>"]
    forbidden = [
        token
        for token in tok.sid.values()
        if token != eos and (task != "rebus" or token != alt)
    ]
    answers, current = [], []
    prediction_count = min(max_new, model.cfg.ctx - len(prompt) + 1)
    for pos in range(len(prompt), len(prompt) + prediction_count):
        logits = model(tokens[:, :pos])[0, -1]
        logits[forbidden] = -torch.inf
        logits[tok.used :] = -torch.inf
        token = int(logits.argmax())
        if token == eos:
            break
        if token == alt:
            answers.append(tok.decode(current))
            current = []
        else:
            current.append(token)
        tokens[0, pos] = token
    answers.append(tok.decode(current))
    return list(dict.fromkeys(answer for answer in answers if answer))
