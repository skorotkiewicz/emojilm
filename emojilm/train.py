"""Train on target tokens and save the checkpoint with the lowest validation loss."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import tempfile

import torch
from torch.nn import functional as F

from .config import Config
from .model import EmojiLM
from .tokenizer import Tokenizer

TASK_WEIGHTS = {"decode": 35, "encode": 30, "rebus": 25, "emojify": 10}


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def pack_rows(rows, tok, ctx):
    packed = defaultdict(list)
    skipped = defaultdict(int)
    for row in rows:
        task = row["task"]
        ids, mask = tok.build(task, row["src"], row["tgt"], row.get("hint"))
        if len(ids) - 1 > ctx:
            skipped[task] += 1
            continue
        # Predict the target and EOS. Ignore the prompt and separator as labels.
        targets = [token if learn else -100 for token, learn in zip(ids[1:], mask[1:])]
        packed[task].append((ids[:-1], targets))
    return packed, dict(skipped)


def batch_tensors(batch, pad, device):
    width = max(len(ids) for ids, _ in batch)
    inputs, targets = [], []
    for ids, labels in batch:
        missing = width - len(ids)
        inputs.append(ids + [pad] * missing)
        targets.append(labels + [-100] * missing)
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def loss_for(model, inputs, targets, reduction="mean"):
    logits = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction=reduction)


@torch.inference_mode()
def validate(model, rows_by_task, pad, batch_size, device):
    model.eval()
    losses = {}
    for task, rows in rows_by_task.items():
        loss, tokens = 0.0, 0
        for i in range(0, len(rows), batch_size):
            inputs, targets = batch_tensors(rows[i : i + batch_size], pad, device)
            loss += loss_for(model, inputs, targets, reduction="sum").item()
            tokens += (targets != -100).sum().item()
        losses[task] = loss / tokens
    model.train()
    return losses


def save_checkpoint(path, model, tok, step, losses):
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        snapshot = Path(directory) / path.name
        torch.save(
            {
                "format": "emojilm-1",
                "cfg": model.cfg.to_dict(),
                "model": model.state_dict(),
                "tokenizer": tok.to_dict(),
                "step": step,
                "validation_loss": losses,
            },
            snapshot,
        )
        snapshot.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("ckpt"))
    parser.add_argument(
        "--model-size",
        choices=["base", "tiny"],
        default="base",
        help="base has 11.2M parameters, tiny has 3.9M",
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if min(args.steps, args.batch, args.ctx, args.eval_every) < 1:
        parser.error("steps, batch, ctx and eval-every must be positive")
    if (
        args.threads < 0
        or args.warmup < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        parser.error(
            "threads and warmup must be non-negative and lr must be finite and positive"
        )
    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    try:
        tok = Tokenizer.load(args.data / "tokenizer.json")
        train, train_skipped = pack_rows(
            load_jsonl(args.data / "train.jsonl"), tok, args.ctx
        )
        val, val_skipped = pack_rows(load_jsonl(args.data / "val.jsonl"), tok, args.ctx)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not train or not val:
        parser.error("Training and validation both need rows that fit --ctx")
    if train_skipped or val_skipped:
        print(
            f"Rows exceeding context were skipped. train={train_skipped} val={val_skipped}",
            flush=True,
        )

    config = Config(vocab=tok.size, ctx=args.ctx)
    if args.model_size == "tiny":
        config = Config(
            vocab=tok.size, ctx=args.ctx, layers=7, dim=192, heads=4, ffn=768
        )
    model = EmojiLM(config).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()
    tasks = sorted(train)
    weights = [TASK_WEIGHTS[task] for task in tasks]
    warmup = min(args.warmup, max(0, args.steps - 1))
    pad = tok.sid["<pad>"]

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "model.pt"
    best = math.inf
    print(
        f"EmojiLM parameters={model.num_params():,} device={device} "
        f"train={sum(map(len, train.values()))} val={sum(map(len, val.values()))} "
        f"bf16={mixed_precision}",
        flush=True,
    )
    for step in range(1, args.steps + 1):
        if step <= warmup:
            lr = args.lr * step / warmup
        else:
            progress = (step - warmup - 1) / max(1, args.steps - warmup - 1)
            lr = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in opt.param_groups:
            group["lr"] = lr

        batch_tasks = rng.choices(tasks, weights=weights, k=args.batch)
        inputs, targets = batch_tensors(
            [rng.choice(train[task]) for task in batch_tasks], pad, device
        )
        opt.zero_grad(set_to_none=True)
        precision = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if mixed_precision
            else nullcontext()
        )
        with precision:
            loss = loss_for(model, inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 25 == 0:
            print(f"step={step} loss={loss.item():.4f} lr={lr:.2e}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            losses = validate(model, val, pad, args.batch, device)
            if not all(math.isfinite(loss) for loss in losses.values()):
                raise RuntimeError("Non-finite validation loss")
            # Use the task mix here too so summary quality cannot dominate checkpoint selection.
            score = sum(
                loss * TASK_WEIGHTS[task] for task, loss in losses.items()
            ) / sum(TASK_WEIGHTS[task] for task in losses)
            print(
                f"step={step} validation={score:.4f} "
                + " ".join(
                    f"{task}={loss:.4f}" for task, loss in sorted(losses.items())
                ),
                flush=True,
            )
            if score < best:
                save_checkpoint(checkpoint, model, tok, step, losses)
                best = score
    print(f"Best checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
