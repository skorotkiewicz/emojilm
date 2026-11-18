"""Train on target tokens and save the best checkpoint on validation data."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import math
from pathlib import Path
import random
from shutil import get_terminal_size
import tempfile

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .config import Config
from .data import load_jsonl, matches_reference
from .edit_candidates import edit_bundle_metadata
from .edits import EDIT_TASK
from .inference import generate, generate_edits
from .model import EmojiLM
from .tokenizer import Tokenizer

TASK_WEIGHTS = {"decode": 35, "encode": 30, "rebus": 25, "emojify": 10}


def pack_rows(rows, tok, ctx):
    packed = defaultdict(list)
    skipped = defaultdict(int)
    for row in rows:
        task = row["task"]
        ids, mask = tok.build(
            task,
            row["src"],
            row["tgt"],
            row.get("hint"),
            target_parts=row.get("target_parts"),
        )
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


def validate_composition(model, tok, rows):
    correct = 0
    model.eval()
    try:
        for row in rows:
            task, source = row["task"], row["src"]
            if task == "encode" and tok.encode_edits:
                correct += matches_reference(row, [generate_edits(model, tok, source)])
                continue
            prompt, _ = tok.build(task, source)
            if len(prompt) <= model.cfg.ctx:
                answers = generate(model, tok, task, source)
                correct += matches_reference(row, answers)
    finally:
        model.train()
    return {"exact": correct, "total": len(rows)}


def save_checkpoint(
    path, model, tok, step, losses, composition_score=None, *, bundle=None
):
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        snapshot = Path(directory) / path.name
        score_metadata = (
            {}
            if composition_score is None
            else {
                "composition_validation": composition_score,
            }
        )
        torch.save(
            {
                "format": "emojilm-1",
                "cfg": model.cfg.to_dict(),
                "model": model.state_dict(),
                "tokenizer": tok.to_dict(),
                "step": step,
                "validation_loss": losses,
                **score_metadata,
                **(bundle or {}),
            },
            snapshot,
        )
        snapshot.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("ckpt"))
    parser.add_argument(
        "--bundle-edits",
        action="store_true",
        help="Embed encode and decode readings in model.pt",
    )
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
    parser.add_argument(
        "--composition-val",
        type=Path,
        help="Separate prompts for direct-generation checkpoint selection",
    )
    parser.add_argument("--composition-eval-every", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.batch,
            args.ctx,
            args.eval_every,
            args.composition_eval_every,
        )
        < 1
    ):
        parser.error("steps, batch, ctx and evaluation intervals must be positive")
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
        if args.bundle_edits and not tok.encode_edits:
            raise ValueError("Bundling requires the encode edits dataset")
        rows = load_jsonl(args.data / "train.jsonl")
        bundle = edit_bundle_metadata(rows) if args.bundle_edits else None
        train, train_skipped = pack_rows(rows, tok, args.ctx)
        del rows
        val, val_skipped = pack_rows(load_jsonl(args.data / "val.jsonl"), tok, args.ctx)
        composition_val = (
            load_jsonl(args.composition_val) if args.composition_val else []
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not train or not val:
        parser.error("Training and validation both need rows that fit --ctx")
    if tok.encode_edits and (EDIT_TASK not in train or EDIT_TASK not in val):
        parser.error(
            "Replacement training and validation both need rows that fit --ctx"
        )
    if args.composition_val and not composition_val:
        parser.error("Composition validation needs at least one prompt")
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
    task_weights = TASK_WEIGHTS.copy()
    if EDIT_TASK in train:
        task_weights[EDIT_TASK] = 20 if "encode" in train else 30
        task_weights["encode"] -= 20
    weights = [task_weights[task] for task in tasks]
    warmup = min(args.warmup, max(0, args.steps - 1))
    pad = tok.sid["<pad>"]

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "model.pt"
    best = (math.inf, math.inf)
    terminal = get_terminal_size()
    with (
        (args.out / "train.log").open("a", encoding="utf-8", buffering=1) as train_log,
        tqdm(
            range(1, args.steps + 1),
            desc="Training",
            unit="step",
            disable=None,
            ncols=terminal.columns or 80,
            nrows=terminal.lines or 24,
        ) as steps,
    ):

        def report(message, *, console=True):
            print(message, file=train_log)
            if console:
                with steps.external_write_mode():
                    print(message, flush=True)

        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        report(f"Run started at {started_at}", console=False)
        report(
            f"EmojiLM parameters={model.num_params():,} device={device} "
            f"train={sum(map(len, train.values()))} val={sum(map(len, val.values()))} "
            f"bf16={mixed_precision}"
        )
        for step in steps:
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
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            console_step = steps.disable and (step == 1 or step % args.eval_every == 0)
            if console_step or step == 1 or step % 25 == 0 or step == args.steps:
                step_loss = loss.item()
                report(
                    f"step={step} loss={step_loss:.4f} lr={lr:.2e}",
                    console=console_step,
                )
                if not steps.disable:
                    steps.set_postfix(
                        loss=f"{step_loss:.4f}", lr=f"{lr:.2e}", refresh=False
                    )
            composition_due = bool(composition_val) and (
                step % args.composition_eval_every == 0 or step == args.steps
            )
            if step % args.eval_every == 0 or step == args.steps or composition_due:
                losses = validate(model, val, pad, args.batch, device)
                if not all(math.isfinite(loss) for loss in losses.values()):
                    raise RuntimeError("Non-finite validation loss")
                # Use the task mix here too so summary quality cannot dominate checkpoint selection.
                score = sum(
                    loss * task_weights[task] for task, loss in losses.items()
                ) / sum(task_weights[task] for task in losses)
                report(
                    f"step={step} validation={score:.4f} "
                    + " ".join(
                        f"{task}={loss:.4f}" for task, loss in sorted(losses.items())
                    )
                )
                composition_score = None
                candidate = (0, score)
                if composition_due:
                    composition_score = validate_composition(
                        model, tok, composition_val
                    )
                    exact, total = (
                        composition_score["exact"],
                        composition_score["total"],
                    )
                    report(f"step={step} composition exact={exact}/{total}")
                    candidate = (-exact, score)
                if (not composition_val or composition_due) and candidate < best:
                    save_checkpoint(
                        checkpoint,
                        model,
                        tok,
                        step,
                        losses,
                        composition_score,
                        bundle=bundle,
                    )
                    best = candidate
        report(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
