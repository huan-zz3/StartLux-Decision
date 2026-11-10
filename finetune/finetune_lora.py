"""LoRA fine-tuning of a StartLux-Decision v1.1 Instant (prompt-bidirectional) model folder on your own typed decisions.

    python finetune/finetune_lora.py --model /path/to/StartLux-Decision-v1.1-0.8B-Instant \
        --train train.jsonl --dev dev.jsonl --out runs/mine
    python -m startlux_decide.cli serve --model runs/mine/merged --port 8090

Data, one JSON object per line (same shape as docs/finetuning.md):

    {"state": <string or object>, "questions": {key: <TypeSafe question>}, "targets": {key: {option: probability}}}

Target keys follow the question type: choice -> its criteria keys, noul -> "true" / "false", score -> "0".."n-1"
(lowest level first).  A hard label is a one-hot distribution; a spread of annotator votes can be given as is.

Each question is rendered exactly as StartLux-Decision renders it when serving, and the loss is the cross-entropy
between the target and the softmax over the option letters at the answer position (the readout these models use), so
the result is served with the same code.  Only LoRA weights are trained; they are merged into the saved model.

This script targets Instant folders only: their prompt is read with bidirectional attention in the full-attention
layers (``startlux_decide.backends.bidirectional``) and the answer comes from one read-out.  Causal AutoThink and v1
folders are out of scope; use ``v1/finetune/finetune_lora.py`` for a v1 folder.

The optional ``finetune`` extra installs torch, transformers, peft, safetensors and the CUDA kernels the linear layers
use on Linux (``pip install -e '.[finetune]'``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext

try:                                    # torch is optional so the data helpers (and their tests) import without it
    import torch
    import torch.nn.functional as F
except ImportError:                     # pragma: no cover - exercised only when torch is absent
    torch = F = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from startlux_decide.modelcfg import load_model_folder  # noqa: E402
from startlux_decide.prompt import (MAX_OPTIONS, check_tokenizer, from_systemone, messages,  # noqa: E402
                                    plain_prompt)

# Batched inputs are padded to one of these lengths: the linear-attention kernels are compiled once per sequence
# length, so padding to exact lengths would recompile for almost every batch.  Copied from the v1 trainer.
PAD_LENGTHS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
               5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536)

# Calibration grid: 801 log-spaced temperatures from 0.2 to 5.0.
GRID = [math.exp(math.log(0.2) + i * (math.log(5.0) - math.log(0.2)) / 800) for i in range(801)]


def padded_length(n):
    """The length a batch whose longest input has n tokens is padded to."""
    for length in PAD_LENGTHS:
        if n <= length:
            return length
    return -(-n // 8192) * 8192


def load_items(path, tok, max_length):
    """-> [(input_ids, number of options, target vector in option order, question type)], number skipped.

    Questions without targets are skipped, as are choice questions over 26 options and prompts longer than
    `max_length`.  A malformed row or target raises with the file and line instead of silently disappearing.
    """
    items, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from None
            if not isinstance(rec, dict) or not isinstance(rec.get("questions"), dict):
                raise ValueError(f'{path}:{n}: each line needs a "questions" object')
            targets = rec.get("targets") or {}
            if not isinstance(targets, dict):
                raise ValueError(f"{path}:{n}: targets must be an object")
            for key, q in rec["questions"].items():
                target = targets.get(key)
                if target is None:
                    continue
                if not isinstance(target, dict):
                    raise ValueError(f"{path}:{n} question {key!r}: target must be an object mapping option ids to "
                                     "probabilities")
                try:
                    row = from_systemone(rec.get("state"), q)
                    order = [o["id"] for o in row["options"]]
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    raise ValueError(f"{path}:{n} question {key!r}: {e}") from None
                if row["type"] == "choice" and len(order) > MAX_OPTIONS:
                    skipped += 1            # split very wide lists into groups of up to 26 before training
                    continue
                try:
                    text = plain_prompt(tok, messages(row))
                    ids = tok.encode(text, add_special_tokens=False)
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    raise ValueError(f"{path}:{n} question {key!r}: {e}") from None
                try:
                    vec = [float(target.get(o, 0.0)) for o in order]
                except (TypeError, ValueError):
                    raise ValueError(f"{path}:{n} question {key!r}: target probabilities must be numbers") from None
                total = math.fsum(vec)
                if total <= 0:
                    raise ValueError(f"{path}:{n} question {key!r}: target has no mass on the listed options")
                if len(ids) > max_length:
                    skipped += 1
                    continue
                items.append((ids, len(order), [v / total for v in vec], row["type"]))
    return items, skipped


def batches(items, max_tokens, shuffle, seed):
    """Length-sorted buckets of up to max_tokens padded tokens, shuffled bucket order."""
    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    out, cur, longest = [], [], 0
    for i in order:
        n = padded_length(len(items[i][0]))
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def step_loss(decoder, letter_rows, items, idx, pad, device):
    """Cross-entropy over the option letters at the answer position.  Returns (loss, correct, logits, target)."""
    from startlux_decide.backends import bidirectional

    longest = padded_length(max(len(items[i][0]) for i in idx))     # a few fixed lengths: kernels compile once each
    k = max(items[i][1] for i in idx)
    ids = torch.full((len(idx), longest), pad, dtype=torch.long)
    attn = torch.zeros_like(ids)
    target = torch.zeros((len(idx), k))
    mask = torch.zeros((len(idx), k), dtype=torch.bool)
    for j, i in enumerate(idx):
        t, c, vec, _ = items[i]
        ids[j, :len(t)] = torch.tensor(t)
        attn[j, :len(t)] = 1
        target[j, :c] = torch.tensor(vec)
        mask[j, :c] = True
    ids, attn, target, mask = ids.to(device), attn.to(device), target.to(device), mask.to(device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with ctx:
        hidden = bidirectional.text_forward(decoder, input_ids=ids, attention_mask=attn, valid=attn, causal=False)
    h = hidden[torch.arange(len(idx), device=device), attn.sum(1) - 1].float()
    logits = (h @ letter_rows[:k].T).masked_fill(~mask, -torch.inf)
    logp = F.log_softmax(logits, -1).masked_fill(~mask, 0.0)
    loss = -(target * logp).sum(-1).mean()
    correct = (logits.argmax(-1) == target.argmax(-1)).float().sum().item()
    return loss, correct, logits.detach(), target.detach()


# ---------------------------------------------------------------- calibration

def nll(pairs, temp):
    """Mean negative log-likelihood of the targets under logits / temperature."""
    if torch is None:
        raise RuntimeError("calibration needs torch")
    total = 0.0
    for z, t in pairs:
        logp = torch.log_softmax(z / temp, -1)
        total -= float((t * logp).sum())
    return total / max(1, len(pairs))


def ece(pairs, temp, bins=10):
    """Expected calibration error of the top option under logits / temperature."""
    if torch is None:
        raise RuntimeError("calibration needs torch")
    conf, ok = [], []
    for z, t in pairs:
        p = torch.softmax(z / temp, -1)
        conf.append(float(p.max()))
        ok.append(float(int(p.argmax()) == int(t.argmax())))
    tot = 0.0
    for b in range(bins):
        sel = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins]
        if sel:
            tot += len(sel) / len(conf) * abs(sum(conf[i] for i in sel) / len(sel)
                                              - sum(ok[i] for i in sel) / len(sel))
    return round(tot, 4)


def fit_temperatures(by_type, min_examples=200):
    """Fit one temperature per question type on (logits, target) pairs -> (pooled, {type: temp}, report).

    Temperatures minimize NLL over GRID.  A type with fewer than `min_examples` pairs gets the pooled temperature.
    """
    pooled_pairs = [pair for pairs in by_type.values() for pair in pairs]
    t_pool = min(GRID, key=lambda T: nll(pooled_pairs, T)) if pooled_pairs else 1.0
    temps, report = {}, {}
    for typ, pairs in by_type.items():
        temps[typ] = min(GRID, key=lambda T: nll(pairs, T)) if len(pairs) >= min_examples else t_pool
        if pairs:
            report[typ] = {"n": len(pairs), "temperature": round(temps[typ], 4),
                           "ece_before": ece(pairs, 1.0), "ece_after": ece(pairs, temps[typ]),
                           "nll_before": round(nll(pairs, 1.0), 4), "nll_after": round(nll(pairs, temps[typ]), 4)}
    return t_pool, temps, report


# ---------------------------------------------------------------- training

def train(a, tokenizer=None):
    if torch is None:
        raise SystemExit("fine-tuning needs torch: pip install -e '.[finetune]'")
    folder = load_model_folder(a.model)
    if not folder.bidirectional:
        raise SystemExit(f"{a.model} is not an Instant (prompt-bidirectional) folder; this script only fine-tunes "
                         "Instant folders.  For a v1 folder use v1/finetune/finetune_lora.py.")
    if a.calibrate and not a.dev:
        raise SystemExit("--calibrate needs --dev (held-out data to fit the temperatures on)")
    if a.calibrate and a.no_merge:
        raise SystemExit("--calibrate writes decision_config.json into the merged model; drop --no-merge")

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model)
    tok = tokenizer
    letters = check_tokenizer(tok)
    if folder.letter_token_ids is not None and list(letters) != list(folder.letter_token_ids):
        raise ValueError("tokenizer letter ids differ from decision_config.json")

    train_items, skipped = load_items(a.train, tok, a.max_length)
    dev_items, _ = load_items(a.dev, tok, a.max_length) if a.dev else ([], 0)
    print(f"{len(train_items)} training questions ({skipped} skipped: too long or over {MAX_OPTIONS} options), "
          f"{len(dev_items)} dev questions", flush=True)
    if not train_items:
        raise SystemExit("no training questions left after rendering and filtering")
    if a.calibrate and not dev_items:
        raise SystemExit("--calibrate needs at least one dev question after rendering and filtering; check --dev")

    import transformers
    from peft import LoraConfig, get_peft_model

    from startlux_decide.backends import bidirectional

    torch.manual_seed(a.seed)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    config = transformers.AutoConfig.from_pretrained(a.model)
    cls = getattr(transformers, config.architectures[0])
    model = cls.from_pretrained(a.model, dtype=dtype, device_map={"": device})
    inner = model.model
    body = getattr(inner, "language_model", inner)
    bidirectional.enable(body)              # before wrapping: the checks run on the plain decoder
    head = model.get_output_embeddings().weight
    letter_rows = head.index_select(0, torch.tensor(letters, device=head.device)).detach().float()
    peft_model = get_peft_model(body, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                                 target_modules="all-linear"))
    decoder = peft_model.base_model.model
    peft_model.print_trainable_parameters()
    params = [p for p in peft_model.parameters() if p.requires_grad]

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    plan = batches(train_items, a.max_tokens, True, a.seed)
    total_steps = max(1, math.ceil(len(plan) * a.epochs / a.accum))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, int(a.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    def evaluate(collect=False):
        if not dev_items:
            return (None, {"choice": [], "noul": [], "score": []}) if collect else None
        peft_model.eval()
        loss_sum, correct, n = 0.0, 0.0, 0
        pairs = {"choice": [], "noul": [], "score": []}
        with torch.no_grad():
            for idx in batches(dev_items, a.max_tokens, False, 0):
                loss, c, logits, target = step_loss(decoder, letter_rows, dev_items, idx, pad, device)
                loss_sum += loss.item() * len(idx)
                correct += c
                n += len(idx)
                if collect:
                    for j, i in enumerate(idx):
                        k = dev_items[i][1]
                        pairs[dev_items[i][3]].append((logits[j, :k].cpu(), target[j, :k].cpu()))
        peft_model.train()
        stats = {"dev_loss": round(loss_sum / n, 4), "dev_accuracy": round(correct / n, 4)}
        return (stats, pairs) if collect else stats

    print(json.dumps({"optimizer_steps": total_steps, "micro_batches_per_epoch": len(plan),
                      "before": evaluate()}), flush=True)
    peft_model.train()
    step, micro, t0 = 0, 0, time.time()
    epoch = 0
    while step < total_steps:
        for idx in batches(train_items, a.max_tokens, True, a.seed + epoch):
            loss, _, _, _ = step_loss(decoder, letter_rows, train_items, idx, pad, device)
            (loss / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                print(json.dumps({"step": step, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0],
                                  "epoch": epoch, "elapsed_s": round(time.time() - t0)}), flush=True)
            if step >= total_steps:
                break
        epoch += 1
        if step < total_steps:
            print(json.dumps({"epoch": epoch, **(evaluate() or {})}), flush=True)

    if a.calibrate:
        final, pairs = evaluate(collect=True)
    else:
        final, pairs = evaluate(), None
    print(json.dumps({"done": {"optimizer_steps": step, "elapsed_s": round(time.time() - t0), **(final or {})}}),
          flush=True)

    os.makedirs(a.out, exist_ok=True)
    adapter = os.path.join(a.out, "adapter")
    peft_model.save_pretrained(adapter)
    print("adapter written to", adapter, flush=True)
    if a.no_merge:
        return
    peft_model.merge_and_unload()           # merges the LoRA weights into the decoder in place
    merged = os.path.join(a.out, "merged")
    model.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    for f in os.listdir(a.model):           # processor configs and decision_config.json travel along
        if f.endswith(".json") and not os.path.exists(os.path.join(merged, f)):
            shutil.copy(os.path.join(a.model, f), merged)

    if a.calibrate:
        pooled, temps, report = fit_temperatures(pairs)
        print(json.dumps({"pooled_temperature": round(pooled, 4), "by_type": report}, indent=1), flush=True)
        path = os.path.join(merged, "decision_config.json")
        with open(path) as f:
            cfg = json.load(f)
        cfg["temperature_by_type"] = {k: round(v, 4) for k, v in temps.items()}
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("updated", path, flush=True)

    print("merged model written to", merged, "- serve it with: startlux-decide serve --model", merged, flush=True)


def main(argv=None, tokenizer=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Instant StartLux-Decision model folder to start from")
    ap.add_argument("--train", required=True, help="training JSONL")
    ap.add_argument("--dev", help="held-out JSONL for dev loss/accuracy (and --calibrate)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-merge", action="store_true", help="only save the LoRA adapter")
    ap.add_argument("--calibrate", action="store_true", help="fit temperatures on --dev and write them into the "
                                                             "merged decision_config.json")
    ap.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    a = ap.parse_args(argv)
    return train(a, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
