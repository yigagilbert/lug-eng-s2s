#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


IGNORE_INDEX = -100


def _split_csv(arg: str) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def _truncate_pair(src: torch.Tensor, tgt: torch.Tensor, max_content_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    if src.numel() + tgt.numel() <= max_content_tokens:
        return src, tgt

    # Keep at least half for target whenever possible.
    tgt_keep = min(tgt.numel(), max(1, max_content_tokens // 2))
    src_keep = max_content_tokens - tgt_keep
    if src.numel() < src_keep:
        extra = src_keep - src.numel()
        src_keep = src.numel()
        tgt_keep = min(tgt.numel(), tgt_keep + extra)
    elif tgt.numel() < tgt_keep:
        extra = tgt_keep - tgt.numel()
        tgt_keep = tgt.numel()
        src_keep = min(src.numel(), src_keep + extra)
    return src[:src_keep], tgt[:tgt_keep]


def _take_tokens(x: torch.Tensor, n: int, mode: str) -> torch.Tensor:
    if n <= 0:
        return x[:0]
    if n >= x.numel():
        return x
    if mode == "tail":
        return x[-n:]
    return x[:n]


def _truncate_pair_with_policy(
    src: torch.Tensor,
    tgt: torch.Tensor,
    max_content_tokens: int,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if src.numel() + tgt.numel() <= max_content_tokens:
        return src, tgt

    # balanced: keep at least half budget for target when possible.
    if policy == "balanced":
        return _truncate_pair(src, tgt, max_content_tokens)

    # head/tail: allocate budget proportionally to original lengths,
    # and choose which side of each sequence to keep.
    total = src.numel() + tgt.numel()
    src_keep = int(round(max_content_tokens * (src.numel() / max(1, total))))
    tgt_keep = max_content_tokens - src_keep

    # Ensure both sides keep at least one token when both are non-empty.
    if src.numel() > 0 and tgt.numel() > 0:
        src_keep = max(1, src_keep)
        tgt_keep = max(1, tgt_keep)
        if src_keep + tgt_keep > max_content_tokens:
            # Remove one token from the larger side to stay on budget.
            if src_keep >= tgt_keep:
                src_keep -= 1
            else:
                tgt_keep -= 1

    # Respect available lengths and redistribute any leftover budget.
    src_keep = min(src_keep, src.numel())
    tgt_keep = min(tgt_keep, tgt.numel())
    leftover = max_content_tokens - (src_keep + tgt_keep)
    if leftover > 0:
        src_room = src.numel() - src_keep
        add_src = min(leftover, src_room)
        src_keep += add_src
        leftover -= add_src
    if leftover > 0:
        tgt_room = tgt.numel() - tgt_keep
        add_tgt = min(leftover, tgt_room)
        tgt_keep += add_tgt

    src_out = _take_tokens(src, src_keep, mode=policy)
    tgt_out = _take_tokens(tgt, tgt_keep, mode=policy)
    return src_out, tgt_out


@dataclass
class PackedIds:
    input_ids: torch.Tensor
    labels: torch.Tensor


class MimiS2SDataset(Dataset):
    def __init__(
        self,
        manifests: list[Path],
        codes_root: Path | None,
        max_seq_len: int,
        cardinality: int,
        num_codebooks: int,
        audio_token_offset: int,
        task_marker_id: int,
        src_marker_id: int,
        sep_marker_id: int,
        tgt_marker_id: int,
        eos_id: int,
        bos_id: int | None = None,
        truncate_policy: str = "balanced",
    ):
        self.items: list[dict[str, Any]] = []
        for manifest in manifests:
            rows = _load_jsonl(manifest)
            for r in rows:
                rec = dict(r)
                rec["_manifest_dir"] = str(manifest.parent)
                self.items.append(rec)

        if not self.items:
            raise RuntimeError("No samples found in manifest(s).")

        self.codes_root = codes_root
        self.max_seq_len = max_seq_len
        self.cardinality = cardinality
        self.num_codebooks = num_codebooks
        self.audio_token_offset = audio_token_offset
        self.task_marker_id = task_marker_id
        self.src_marker_id = src_marker_id
        self.sep_marker_id = sep_marker_id
        self.tgt_marker_id = tgt_marker_id
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.truncate_policy = truncate_policy

        self._fixed_prefix_len = 4 + (1 if bos_id is not None else 0)
        # Prefix: [BOS?] TASK_MARKER SRC_MARKER SRC_TOKENS SEP_MARKER TGT_MARKER
        # and we always keep one EOS token at the end.
        if self.max_seq_len <= self._fixed_prefix_len + 1:
            raise RuntimeError("max_seq_len too small for required control tokens.")

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, raw: str, manifest_dir: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        if self.codes_root is not None:
            return self.codes_root / p
        return Path(manifest_dir) / p

    def _load_codes(self, path: Path) -> torch.Tensor:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            codes = payload["codes"]
        else:
            codes = payload
        codes = torch.as_tensor(codes)
        if codes.dim() == 3:
            codes = codes.squeeze(0)
        if codes.dim() != 2:
            raise RuntimeError(f"Unexpected code shape {tuple(codes.shape)} at {path}")
        if codes.shape[0] < self.num_codebooks:
            raise RuntimeError(
                f"Codebooks mismatch at {path}: expected >= {self.num_codebooks}, got {codes.shape[0]}"
            )
        codes = codes[: self.num_codebooks].long()  # [K, T]
        if codes.min().item() < 0 or codes.max().item() >= self.cardinality:
            raise RuntimeError(
                f"Token out of range at {path}: min={codes.min().item()} max={codes.max().item()} "
                f"cardinality={self.cardinality}"
            )
        return codes

    def _pack_audio_codes(self, codes: torch.Tensor) -> torch.Tensor:
        # [K, T] -> frame-major [T*K], offset each codebook into one shared audio vocab.
        offsets = (torch.arange(self.num_codebooks).unsqueeze(1) * self.cardinality).long()
        packed = (codes + offsets).transpose(0, 1).reshape(-1)
        return packed + self.audio_token_offset

    def _build_sequence(self, src_audio: torch.Tensor, tgt_audio: torch.Tensor) -> PackedIds:
        max_content = self.max_seq_len - self._fixed_prefix_len - 1  # minus EOS
        src_audio, tgt_audio = _truncate_pair_with_policy(
            src_audio, tgt_audio, max_content, policy=self.truncate_policy
        )

        prefix = []
        if self.bos_id is not None:
            prefix.append(self.bos_id)
        prefix.append(self.task_marker_id)
        prefix.append(self.src_marker_id)
        prefix.extend(src_audio.tolist())
        prefix.append(self.sep_marker_id)
        prefix.append(self.tgt_marker_id)

        input_ids = prefix + tgt_audio.tolist() + [self.eos_id]
        labels = [IGNORE_INDEX] * len(prefix) + tgt_audio.tolist() + [self.eos_id]

        return PackedIds(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        src_path = self._resolve_path(item["src_codes"], item["_manifest_dir"])
        tgt_path = self._resolve_path(item["tgt_codes"], item["_manifest_dir"])

        src_codes = self._load_codes(src_path)
        tgt_codes = self._load_codes(tgt_path)

        src_audio = self._pack_audio_codes(src_codes)
        tgt_audio = self._pack_audio_codes(tgt_codes)

        packed = self._build_sequence(src_audio, tgt_audio)
        return {"input_ids": packed.input_ids, "labels": packed.labels}


class CausalCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].shape[0] for x in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, sample in enumerate(batch):
            l = sample["input_ids"].shape[0]
            input_ids[i, :l] = sample["input_ids"]
            labels[i, :l] = sample["labels"]
            attention_mask[i, :l] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}


def _build_target_modules(model, target_modules_arg: str) -> list[str]:
    if target_modules_arg != "auto":
        out = _split_csv(target_modules_arg)
        if not out:
            raise RuntimeError("No target_modules parsed from argument.")
        return out

    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    found = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)
    if not found:
        raise RuntimeError(
            "Could not auto-detect LoRA target modules. "
            "Set --target-modules explicitly."
        )
    return sorted(found)


def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int = 0) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = _move_to_device(batch, device)
            out = model(**batch)
            valid_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
            if valid_tokens > 0:
                total += float(out.loss.item()) * valid_tokens
                n += valid_tokens
            if max_batches > 0 and (i + 1) >= max_batches:
                break
    model.train()
    if n == 0:
        return float("nan")
    return total / n


def save_artifacts(
    save_dir: Path,
    model,
    tokenizer,
    training_args: dict[str, Any],
    mapping: dict[str, Any],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    (save_dir / "train_config.json").write_text(
        json.dumps(training_args, indent=2), encoding="utf-8"
    )
    (save_dir / "mimi_token_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PEFT/LoRA S2S model on Mimi tokenized manifests."
    )
    parser.add_argument("--train-manifests", type=str, required=True,
                        help="Comma-separated JSONL manifest paths for training.")
    parser.add_argument("--valid-manifests", type=str, default="",
                        help="Comma-separated JSONL manifest paths for validation.")
    parser.add_argument("--codes-root", type=str, default="",
                        help="Optional base directory for relative src_codes/tgt_codes paths.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument(
        "--task-token",
        type=str,
        default="<TASK_S2ST_EN_LG>",
        help="Human-readable task token name stored in mapping metadata.",
    )

    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--truncate-policy",
        type=str,
        default="balanced",
        choices=["balanced", "head", "tail"],
        help=(
            "How to truncate src/tgt audio tokens when sequence is too long: "
            "balanced=half-budget target-first; head=keep starts; tail=keep ends."
        ),
    )
    parser.add_argument(
        "--context-overflow",
        type=str,
        default="error",
        choices=["error", "clamp"],
        help=(
            "Behavior when --max-seq-len exceeds model context. "
            "error=stop; clamp=reduce to model max_position_embeddings."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="If > 0, stops after this many optimizer steps.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=0,
                        help="0 means full validation.")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--cardinality", type=int, default=2048)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, default="auto",
                        help='Comma-separated module names or "auto".')
    parser.add_argument("--modules-to-save", type=str, default="embed_tokens,lm_head")
    args = parser.parse_args()

    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifests = [Path(p) for p in _split_csv(args.train_manifests)]
    if not train_manifests:
        raise SystemExit("No train manifests provided.")
    valid_manifests = [Path(p) for p in _split_csv(args.valid_manifests)] if args.valid_manifests else []
    codes_root = Path(args.codes_root) if args.codes_root else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device)

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    base_vocab = model.get_input_embeddings().num_embeddings
    audio_vocab = args.num_codebooks * args.cardinality
    n_special = 4
    audio_token_offset = base_vocab + n_special
    new_vocab = base_vocab + n_special + audio_vocab

    task_marker_id = base_vocab
    src_marker_id = base_vocab + 1
    sep_marker_id = base_vocab + 2
    tgt_marker_id = base_vocab + 3

    print(
        f"Resizing embeddings from {base_vocab} -> {new_vocab} "
        f"(audio_vocab={audio_vocab}, specials={n_special})"
    )
    model.resize_token_embeddings(new_vocab)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id; specify a base model with EOS.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    bos_id = tokenizer.bos_token_id

    model_ctx = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_ctx, int) and model_ctx > 0 and args.max_seq_len > model_ctx:
        msg = (
            f"--max-seq-len ({args.max_seq_len}) exceeds model context "
            f"max_position_embeddings ({model_ctx})."
        )
        if args.context_overflow == "clamp":
            print(f"{msg} Clamping max_seq_len to {model_ctx}.")
            args.max_seq_len = model_ctx
        else:
            raise RuntimeError(msg + " Use --context-overflow clamp to auto-adjust.")

    target_modules = _build_target_modules(model, args.target_modules)
    modules_to_save = _split_csv(args.modules_to_save)
    print(f"LoRA target modules: {target_modules}")
    print(f"LoRA modules_to_save: {modules_to_save}")
    print(f"Task token: {args.task_token} (id={task_marker_id})")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save if modules_to_save else None,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    train_ds = MimiS2SDataset(
        manifests=train_manifests,
        codes_root=codes_root,
        max_seq_len=args.max_seq_len,
        cardinality=args.cardinality,
        num_codebooks=args.num_codebooks,
        audio_token_offset=audio_token_offset,
        task_marker_id=task_marker_id,
        src_marker_id=src_marker_id,
        sep_marker_id=sep_marker_id,
        tgt_marker_id=tgt_marker_id,
        eos_id=eos_id,
        bos_id=bos_id,
        truncate_policy=args.truncate_policy,
    )
    valid_ds = None
    if valid_manifests:
        valid_ds = MimiS2SDataset(
            manifests=valid_manifests,
            codes_root=codes_root,
            max_seq_len=args.max_seq_len,
            cardinality=args.cardinality,
            num_codebooks=args.num_codebooks,
            audio_token_offset=audio_token_offset,
            task_marker_id=task_marker_id,
            src_marker_id=src_marker_id,
            sep_marker_id=sep_marker_id,
            tgt_marker_id=tgt_marker_id,
            eos_id=eos_id,
            bos_id=bos_id,
            truncate_policy=args.truncate_policy,
        )

    collator = CausalCollator(pad_id=pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    print(
        f"Training samples={len(train_ds)} "
        f"valid_samples={(len(valid_ds) if valid_ds is not None else 0)} "
        f"updates={total_updates} warmup={warmup_steps}"
    )
    if args.max_steps > 0:
        print(
            f"--max-steps={args.max_steps} is set; training will continue across epochs "
            "until that many optimizer updates are reached."
        )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_count = 0
    epoch = 0

    while global_step < total_updates:
        epoch += 1
        micro_since_update = 0
        for step, batch in enumerate(train_loader):
            batch = _move_to_device(batch, device)
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            micro_since_update += 1

            running_loss += float(out.loss.item())
            running_count += 1

            is_last_micro = (step + 1) == len(train_loader)
            if micro_since_update == args.grad_accum or is_last_micro:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                micro_since_update = 0

                if global_step % args.log_every == 0:
                    avg_loss = running_loss / max(1, running_count)
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"step={global_step}/{total_updates} "
                        f"epoch={epoch} loss={avg_loss:.4f} lr={lr:.2e}"
                    )
                    running_loss = 0.0
                    running_count = 0

                if valid_loader is not None and args.eval_every > 0 and global_step % args.eval_every == 0:
                    val_loss = evaluate(model, valid_loader, device, max_batches=args.eval_max_batches)
                    print(f"[eval] step={global_step} val_loss={val_loss:.4f}")

                if args.save_every > 0 and global_step % args.save_every == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    save_artifacts(
                        ckpt_dir,
                        model,
                        tokenizer,
                        training_args=vars(args),
                        mapping={
                            "base_model": args.base_model,
                            "audio_token_offset": audio_token_offset,
                            "num_codebooks": args.num_codebooks,
                            "cardinality": args.cardinality,
                            "task_token": args.task_token,
                            "task_marker_id": task_marker_id,
                            "src_marker_id": src_marker_id,
                            "sep_marker_id": sep_marker_id,
                            "tgt_marker_id": tgt_marker_id,
                            "eos_id": eos_id,
                            "bos_id": bos_id,
                            "pad_id": pad_id,
                        },
                    )
                    print(f"Saved checkpoint to {ckpt_dir}")

                if global_step >= total_updates:
                    break

    if valid_loader is not None:
        final_val_loss = evaluate(model, valid_loader, device, max_batches=args.eval_max_batches)
        print(f"[final eval] val_loss={final_val_loss:.4f}")

    final_dir = output_dir / "final"
    save_artifacts(
        final_dir,
        model,
        tokenizer,
        training_args=vars(args),
        mapping={
            "base_model": args.base_model,
            "audio_token_offset": audio_token_offset,
            "num_codebooks": args.num_codebooks,
            "cardinality": args.cardinality,
            "task_token": args.task_token,
            "task_marker_id": task_marker_id,
            "src_marker_id": src_marker_id,
            "sep_marker_id": sep_marker_id,
            "tgt_marker_id": tgt_marker_id,
            "eos_id": eos_id,
            "bos_id": bos_id,
            "pad_id": pad_id,
        },
    )
    print(f"Training complete. Final adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
