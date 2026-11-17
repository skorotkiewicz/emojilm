"""Load a bundled checkpoint and generate answers with PyTorch."""

import torch

from .composition import Composer
from .config import Config
from .corpora import canonical
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


class NeuralComposer(Composer):
    """Use the checkpoint's individual rebus readings in both directions."""

    def __init__(self, model, tok, max_new=64):
        super().__init__({})
        self.model = model
        self.tok = tok
        self.max_new = max_new
        self.device = model.tok_emb.weight.device
        self.glyphs = [emoji for emoji in tok.emoji if len(tok.encode(emoji)) == 1]
        self.emoji_variants = {}
        for emoji in sorted(self.glyphs, key=lambda glyph: "\ufe0f" not in glyph):
            self.emoji_variants.setdefault(canonical(emoji), emoji)
        self.prompts = None

    def component_readings(self, emoji):
        emoji = self.emoji_variants.get(canonical(emoji), emoji)
        if emoji not in self.readings:
            prompt, _ = self.tok.build("rebus", emoji)
            answers = (
                generate(self.model, self.tok, "rebus", emoji, max_new=self.max_new)
                if len(prompt) <= self.model.cfg.ctx
                else []
            )
            words = [answer.strip() for answer in answers]
            self.readings[emoji] = [word for word in words if word.isprintable()]
        return self.readings[emoji]

    def read_component(self, emoji):
        words = self.component_readings(emoji)
        return words[0] if words else None

    def write_component(self, word):
        key = " ".join(word.split())
        if key not in self.symbols:
            glyph = self.rank_glyphs(key)
            if glyph and any(
                reading.casefold() == key.casefold()
                for reading in self.component_readings(glyph)
            ):
                self.symbols[key] = glyph
            elif key.casefold() != key:
                self.symbols[key] = self.write_component(key.casefold())
            else:
                self.symbols[key] = None
        return self.symbols[key]

    @torch.inference_mode()
    def rank_glyphs(self, word):
        # A bracketed word inverts the learned single-emoji rebus task.
        labels = self.tok.encode(word)
        width = len(labels) + 4
        if not labels or not self.glyphs or width > self.model.cfg.ctx:
            return None
        if self.prompts is None:
            self.prompts = torch.tensor(
                [self.tok.build("rebus", emoji)[0] for emoji in self.glyphs],
                dtype=torch.long,
                device=self.device,
            )
        label_ids = torch.tensor(labels, dtype=torch.long, device=self.device)
        stops = [self.tok.sid["<eos>"], self.tok.sid["<alt>"]]
        batch_size = min(128, max(1, 2048 // width))
        best_score, best_glyph = float("-inf"), None
        for offset in range(0, len(self.glyphs), batch_size):
            prompts = self.prompts[offset : offset + batch_size]
            inputs = torch.cat((prompts, label_ids.expand(len(prompts), -1)), dim=1)
            logits = self.model(inputs)[:, prompts.size(1) - 1 :]
            log_probs = logits.log_softmax(dim=-1)
            targets = label_ids.view(1, -1, 1).expand(len(prompts), -1, -1)
            words = log_probs[:, :-1].gather(-1, targets).squeeze(-1).sum(-1)
            scores = words + log_probs[:, -1, stops].logsumexp(-1)
            index = int(scores.argmax())
            score = float(scores[index])
            if score > best_score:
                best_score, best_glyph = score, self.glyphs[offset + index]
        return best_glyph
