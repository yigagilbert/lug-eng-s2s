#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Train a PEFT speech-to-text translation model on Mimi-tokenized manifests.

This trainer reuses the same Mimi-token dataset format as the STS pipeline, but
switches the supervised target from target audio codes to target text tokens.
The resulting model maps Luganda speech input to English text output.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)


IGNORE_INDEX = -100


def _split_csv(arg: str) -> list[str]:
    """Split a comma-separated CLI/config value into trimmed entries."""
    return [x.strip() for x in arg.split(",") if x.strip()]


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    """Pick an execution dtype compatible with the requested device."""
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


def _configure_torch_backend(device: torch.device) -> None:
    """Enable CUDA backend settings that improve training throughput."""
    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _set_seed(seed: int) -> None:
    """Seed Python and torch RNGs for more reproducible training runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest into memory."""
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def _load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON config file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected top-level JSON object in config: {path}")
    return data


def _resolve_path(value: str, base_dir: Path) -> Path:
    """Resolve a potentially relative path against a config directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _safe_repo_dir_name(repo_id: str) -> str:
    """Turn a repo id into a stable local snapshot directory name."""
    return repo_id.replace("/", "__")


def _take_tokens(x: torch.Tensor, n: int, mode: str) -> torch.Tensor:
    """Take `n` tokens from the head or tail of a 1D tensor."""
    if n <= 0:
        return x[:0]
    if n >= x.numel():
        return x
    if mode == "tail":
        return x[-n:]
    return x[:n]


def _truncate_pair(src: torch.Tensor, tgt: torch.Tensor, max_content_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate a source/target pair while reserving a reasonable target budget."""
    if src.numel() + tgt.numel() <= max_content_tokens:
        return src, tgt

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


def _truncate_to_frame_boundary(n_tokens: int, num_codebooks: int) -> int:
    """Round a packed-audio token count down to a full number of Mimi frames."""
    return (n_tokens // num_codebooks) * num_codebooks


def _truncate_pair_with_policy(
    src: torch.Tensor,
    tgt: torch.Tensor,
    max_content_tokens: int,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the configured truncation strategy to source audio and target text tokens."""
    if src.numel() + tgt.numel() <= max_content_tokens:
        return src, tgt

    if policy == "src_first":
        min_tgt_keep = 1 if tgt.numel() > 0 else 0
        src_keep = min(src.numel(), max(0, max_content_tokens - min_tgt_keep))
        tgt_keep = max_content_tokens - src_keep
        tgt_keep = min(tgt.numel(), max(0, tgt_keep))
        if src.numel() > 0 and src_keep == 0 and max_content_tokens > 0:
            src_keep = min(src.numel(), max_content_tokens)
            tgt_keep = max(0, max_content_tokens - src_keep)
        src_out = _take_tokens(src, src_keep, mode="head")
        tgt_out = _take_tokens(tgt, tgt_keep, mode="head")
        return src_out, tgt_out

    if policy == "balanced":
        return _truncate_pair(src, tgt, max_content_tokens)

    total = src.numel() + tgt.numel()
    src_keep = int(round(max_content_tokens * (src.numel() / max(1, total))))
    tgt_keep = max_content_tokens - src_keep

    if src.numel() > 0 and tgt.numel() > 0:
        src_keep = max(1, src_keep)
        tgt_keep = max(1, tgt_keep)
        if src_keep + tgt_keep > max_content_tokens:
            if src_keep >= tgt_keep:
                src_keep -= 1
            else:
                tgt_keep -= 1

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
    """Prepared causal-LM inputs and labels for one training example."""
    input_ids: torch.Tensor
    labels: torch.Tensor


class MimiS2TTDataset(Dataset):
    """Dataset that turns Mimi source codes plus target text into causal-LM examples."""
    def __init__(
        self,
        manifests: list[Path],
        codes_root: Path | None,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        cardinality: int,
        num_codebooks: int,
        audio_token_offset: int,
        task_marker_id: int,
        src_marker_id: int,
        sep_marker_id: int,
        tgt_marker_id: int,
        eos_id: int,
        target_text_key: str,
        bos_id: int | None = None,
        truncate_policy: str = "balanced",
    ):
        self.items: list[dict[str, Any]] = []
        for manifest in manifests:
            rows = _load_jsonl(manifest)
            for row in rows:
                rec = dict(row)
                rec["_manifest_dir"] = str(manifest.parent)
                self.items.append(rec)

        if not self.items:
            raise RuntimeError("No samples found in manifest(s).")

        self.codes_root = codes_root
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cardinality = cardinality
        self.num_codebooks = num_codebooks
        self.audio_token_offset = audio_token_offset
        self.task_marker_id = task_marker_id
        self.src_marker_id = src_marker_id
        self.sep_marker_id = sep_marker_id
        self.tgt_marker_id = tgt_marker_id
        self.eos_id = eos_id
        self.target_text_key = target_text_key
        self.bos_id = bos_id
        self.truncate_policy = truncate_policy

        self._fixed_prefix_len = 4 + (1 if bos_id is not None else 0)
        if self.max_seq_len <= self._fixed_prefix_len + 1:
            raise RuntimeError("max_seq_len too small for required control tokens.")

    def __len__(self) -> int:
        """Return the number of manifest rows available for training."""
        return len(self.items)

    def _resolve_path(self, raw: str, manifest_dir: str) -> Path:
        """Resolve relative Mimi code paths against `codes_root` or the manifest directory."""
        p = Path(raw)
        if p.is_absolute():
            return p
        if self.codes_root is not None:
            return self.codes_root / p
        return Path(manifest_dir) / p

    def _load_codes(self, path: Path, expected_frames: int | None = None) -> torch.Tensor:
        """Load one Mimi code tensor and validate codebook/frame metadata."""
        payload = torch.load(path, map_location="cpu")
        stored_frames = None
        if isinstance(payload, dict):
            stored_cardinality = payload.get("cardinality")
            if stored_cardinality is not None and int(stored_cardinality) != self.cardinality:
                raise RuntimeError(
                    f"Cardinality mismatch at {path}: expected {self.cardinality}, "
                    f"file has {stored_cardinality}"
                )
            stored_num_codebooks = payload.get("num_codebooks")
            if stored_num_codebooks is not None and int(stored_num_codebooks) < self.num_codebooks:
                raise RuntimeError(
                    f"Codebooks mismatch at {path}: expected >= {self.num_codebooks}, "
                    f"file has {stored_num_codebooks}"
                )
            stored_frames = payload.get("num_frames")
            codes = payload["codes"]
        else:
            codes = payload
        codes = torch.as_tensor(codes)
        if codes.dim() == 3:
            codes = codes.squeeze(0)
        if codes.dim() != 2:
            raise RuntimeError(f"Unexpected code shape {tuple(codes.shape)} at {path}")
        if stored_frames is not None and expected_frames is not None and int(stored_frames) != int(expected_frames):
            raise RuntimeError(
                f"Frame-count mismatch at {path}: manifest expects {expected_frames}, "
                f"file stores {stored_frames}"
            )
        resolved_frames = int(stored_frames) if stored_frames is not None else (
            int(expected_frames) if expected_frames is not None else codes.shape[-1]
        )
        if resolved_frames <= 0 or resolved_frames > codes.shape[-1]:
            raise RuntimeError(
                f"Invalid frame count at {path}: resolved_frames={resolved_frames}, "
                f"available={codes.shape[-1]}"
            )
        codes = codes[:, :resolved_frames]
        if codes.shape[0] < self.num_codebooks:
            raise RuntimeError(
                f"Codebooks mismatch at {path}: expected >= {self.num_codebooks}, got {codes.shape[0]}"
            )
        codes = codes[: self.num_codebooks].long()
        if codes.min().item() < 0 or codes.max().item() >= self.cardinality:
            raise RuntimeError(
                f"Token out of range at {path}: min={codes.min().item()} max={codes.max().item()} "
                f"cardinality={self.cardinality}"
            )
        return codes

    def _pack_audio_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Pack `[codebook, frame]` Mimi indices into one flat audio-token stream."""
        offsets = (torch.arange(self.num_codebooks).unsqueeze(1) * self.cardinality).long()
        packed = (codes + offsets).transpose(0, 1).reshape(-1)
        return packed + self.audio_token_offset

    def _tokenize_target_text(self, text: str) -> torch.Tensor:
        """Normalize and tokenize the target-language text without extra special tokens."""
        normalized = " ".join(str(text).strip().split())
        text_ids = self.tokenizer.encode(normalized, add_special_tokens=False)
        return torch.tensor(text_ids, dtype=torch.long)

    def _build_sequence(self, src_audio: torch.Tensor, tgt_text_ids: torch.Tensor) -> PackedIds:
        """Assemble the causal-LM prompt and supervised target sequence for STT."""
        max_content = self.max_seq_len - self._fixed_prefix_len - 1
        src_audio, tgt_text_ids = _truncate_pair_with_policy(
            src_audio, tgt_text_ids, max_content, policy=self.truncate_policy
        )
        src_audio = src_audio[:_truncate_to_frame_boundary(src_audio.numel(), self.num_codebooks)]

        prefix = []
        if self.bos_id is not None:
            prefix.append(self.bos_id)
        prefix.append(self.task_marker_id)
        prefix.append(self.src_marker_id)
        prefix.extend(src_audio.tolist())
        prefix.append(self.sep_marker_id)
        prefix.append(self.tgt_marker_id)

        input_ids = prefix + tgt_text_ids.tolist() + [self.eos_id]
        labels = [IGNORE_INDEX] * len(prefix) + tgt_text_ids.tolist() + [self.eos_id]

        return PackedIds(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load one manifest row and convert it into padded-model-ready tensors."""
        item = self.items[idx]
        src_path = self._resolve_path(item["src_codes"], item["_manifest_dir"])
        src_codes = self._load_codes(src_path, expected_frames=item.get("src_frames"))
        src_audio = self._pack_audio_codes(src_codes)

        target_text = str(item.get(self.target_text_key, ""))
        tgt_text_ids = self._tokenize_target_text(target_text)

        packed = self._build_sequence(src_audio, tgt_text_ids)
        return {"input_ids": packed.input_ids, "labels": packed.labels}


class CausalCollator:
    """Pad variable-length causal-LM samples into a batch."""
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Create padded `input_ids`, `labels`, and `attention_mask` tensors."""
        max_len = max(x["input_ids"].shape[0] for x in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, sample in enumerate(batch):
            length = sample["input_ids"].shape[0]
            input_ids[i, :length] = sample["input_ids"]
            labels[i, :length] = sample["labels"]
            attention_mask[i, :length] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a collated batch onto the training device."""
    return {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}


def _build_target_modules(model, target_modules_arg: str) -> list[str]:
    """Resolve the LoRA target module list from CLI input or model inspection."""
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
    """Compute average token loss over a validation loader."""
    was_training = model.training
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
    if was_training:
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
    """Save adapter weights, tokenizer files, config, and token mapping metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    (save_dir / "train_config.json").write_text(
        json.dumps(training_args, indent=2), encoding="utf-8"
    )
    (save_dir / "mimi_token_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )


def _apply_config_overrides(
    args: argparse.Namespace, config: dict[str, Any], config_path: Path | None
) -> dict[str, Any]:
    """Merge JSON config values into parsed CLI args and return the data section."""
    known_keys = set(vars(args).keys())
    merged_training = {
        key: value for key, value in config.items()
        if key in known_keys and key not in {"config", "hf_token"}
    }
    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, dict):
        raise RuntimeError("Config field `training` must be an object when provided.")
    merged_training.update(training_cfg)

    base_dir = config_path.parent if config_path is not None else Path.cwd()
    path_like_keys = {"output_dir", "codes_root"}

    for key, value in merged_training.items():
        if key not in known_keys:
            continue
        if isinstance(value, list) and key in {"train_manifests", "valid_manifests"}:
            value = ",".join(str(x) for x in value)
        if isinstance(value, str) and value and key in path_like_keys:
            value = str(_resolve_path(value, base_dir))
        setattr(args, key, value)

    if "hf_token" in config:
        args.hf_token = str(config["hf_token"])

    data_cfg = config.get("data", {})
    if data_cfg and not isinstance(data_cfg, dict):
        raise RuntimeError("Config field `data` must be an object when provided.")
    return data_cfg if isinstance(data_cfg, dict) else {}


def _resolve_hf_token(args: argparse.Namespace, config: dict[str, Any]) -> str | None:
    """Resolve the Hugging Face token from CLI first, then configured env var."""
    if args.hf_token:
        return args.hf_token
    token_env = config.get("hf_token_env", "HF_TOKEN")
    token = os.environ.get(token_env)
    return token if token else None


def _normalize_repo_entry(entry: Any) -> dict[str, Any]:
    """Normalize a repo config entry into dict form."""
    if isinstance(entry, str):
        return {"repo_id": entry}
    if isinstance(entry, dict):
        return dict(entry)
    raise RuntimeError("Each data repo entry must be either a string repo id or an object.")


def _resolve_manifest_paths(
    repo_root: Path,
    train_manifest_name: str,
    valid_manifest_name: str,
    require_valid: bool,
) -> tuple[Path, Path | None]:
    """Resolve train/validation manifests under one local dataset snapshot."""
    train_manifest = repo_root / train_manifest_name
    if not train_manifest.exists():
        raise RuntimeError(f"Missing train manifest at {train_manifest}")

    valid_manifest = repo_root / valid_manifest_name
    if valid_manifest.exists():
        return train_manifest, valid_manifest
    if require_valid:
        raise RuntimeError(f"Missing validation manifest at {valid_manifest}")
    return train_manifest, None


def _resolve_repo_manifests(
    args: argparse.Namespace,
    data_cfg: dict[str, Any],
    config_path: Path | None,
    token: str | None,
) -> tuple[list[Path], list[Path]]:
    """Resolve training and validation manifests from CLI paths or configured repos."""
    if args.train_manifests:
        train_manifests = [Path(p) for p in _split_csv(args.train_manifests)]
        valid_manifests = [Path(p) for p in _split_csv(args.valid_manifests)] if args.valid_manifests else []
        return train_manifests, valid_manifests

    repo_entries = data_cfg.get("repos", [])
    if not repo_entries:
        raise SystemExit(
            "No training data provided. Set --train-manifests or add data.repos to the config."
        )
    if not isinstance(repo_entries, list):
        raise RuntimeError("Config field `data.repos` must be a list.")

    base_dir = config_path.parent if config_path is not None else Path.cwd()
    snapshot_dir_value = data_cfg.get("snapshot_dir", "data/hf_mimi_snapshots")
    snapshot_root = _resolve_path(str(snapshot_dir_value), base_dir)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    cache_dir = data_cfg.get("cache_dir")
    cache_dir_path = _resolve_path(str(cache_dir), base_dir) if cache_dir else None

    default_train_manifest = str(data_cfg.get("train_manifest", "dataset.train.jsonl"))
    default_valid_manifest = str(data_cfg.get("valid_manifest", "dataset.validation.jsonl"))
    require_valid = bool(data_cfg.get("require_valid_manifest", False))

    train_manifests: list[Path] = []
    valid_manifests: list[Path] = []

    for raw_repo in repo_entries:
        repo = _normalize_repo_entry(raw_repo)
        repo_id = repo.get("repo_id")
        local_path_value = repo.get("local_path")
        if local_path_value:
            repo_root = _resolve_path(str(local_path_value), base_dir)
            source_desc = str(repo_root)
        else:
            if not repo_id:
                raise RuntimeError("Repo entries must define `repo_id` unless `local_path` is used.")
            local_dir_value = repo.get("local_dir")
            if local_dir_value:
                local_dir = _resolve_path(str(local_dir_value), base_dir)
            else:
                local_dir = snapshot_root / _safe_repo_dir_name(str(repo_id))
            local_dir.mkdir(parents=True, exist_ok=True)
            repo_token = str(repo.get("token")) if repo.get("token") else token
            print(f"Syncing dataset repo: {repo_id} -> {local_dir}")
            repo_root = Path(
                snapshot_download(
                    repo_id=str(repo_id),
                    repo_type=str(repo.get("repo_type", "dataset")),
                    token=repo_token,
                    local_dir=str(local_dir),
                    cache_dir=str(cache_dir_path) if cache_dir_path is not None else None,
                )
            )
            source_desc = str(repo_id)

        train_manifest_name = str(repo.get("train_manifest", default_train_manifest))
        valid_manifest_name = str(repo.get("valid_manifest", default_valid_manifest))
        include_train = bool(repo.get("include_train", True))
        include_valid = bool(repo.get("include_valid", True))

        train_manifest, valid_manifest = _resolve_manifest_paths(
            repo_root=repo_root,
            train_manifest_name=train_manifest_name,
            valid_manifest_name=valid_manifest_name,
            require_valid=require_valid and include_valid,
        )
        if include_train:
            train_manifests.append(train_manifest)
        if include_valid and valid_manifest is not None:
            valid_manifests.append(valid_manifest)
        print(
            f"Resolved manifests from {source_desc}: "
            f"train={train_manifest} "
            f"valid={valid_manifest if valid_manifest is not None else '(none)'}"
        )

    if not train_manifests:
        raise SystemExit("No train manifests were resolved from the configured repos.")
    return train_manifests, valid_manifests


def main() -> None:
    """Run the full STT PEFT training workflow from config parsing to final save."""
    parser = argparse.ArgumentParser(
        description="Train a PEFT/LoRA speech-to-text translation model on Mimi tokenized manifests."
    )
    parser.add_argument("--config", type=str, default="",
                        help="Optional JSON config file with training settings and data repos.")
    parser.add_argument("--hf-token", type=str, default="",
                        help="Optional HF token for private dataset repos/models.")
    parser.add_argument("--train-manifests", type=str, default="",
                        help="Comma-separated JSONL manifest paths for training.")
    parser.add_argument("--valid-manifests", type=str, default="",
                        help="Comma-separated JSONL manifest paths for validation.")
    parser.add_argument("--codes-root", type=str, default="",
                        help="Optional base directory for relative src_codes paths.")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument(
        "--task-token",
        type=str,
        default="<TASK_S2TT_LG_EN>",
        help="Human-readable task token label stored in mapping metadata.",
    )
    parser.add_argument(
        "--target-text-key",
        type=str,
        default="tgt_text",
        help="Manifest field containing target-language text.",
    )

    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--truncate-policy",
        type=str,
        default="balanced",
        choices=["balanced", "head", "tail", "src_first"],
        help=(
            "How to truncate source-audio and target-text content when sequence is too long: "
            "balanced=share budget, head=keep starts, tail=keep ends, src_first=prefer source audio."
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
    parser.add_argument("--persistent-workers", action="store_true",
                        help="Keep dataloader workers alive between epochs.")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Batches prefetched per worker when num_workers > 0.")
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

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = _load_json(config_path) if config_path is not None else {}
    data_cfg = _apply_config_overrides(args, config, config_path)
    hf_token = _resolve_hf_token(args, config)

    if not args.output_dir:
        raise SystemExit("Missing output directory. Set --output-dir or `training.output_dir` in the config.")
    if not args.base_model:
        raise SystemExit("Missing base model. Set --base-model or `training.base_model` in the config.")

    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifests, valid_manifests = _resolve_repo_manifests(args, data_cfg, config_path, hf_token)
    codes_root = Path(args.codes_root) if args.codes_root else None
    resolved_args = vars(args).copy()
    resolved_args["config"] = str(config_path) if config_path is not None else ""
    resolved_args["train_manifests"] = [str(p) for p in train_manifests]
    resolved_args["valid_manifests"] = [str(p) for p in valid_manifests]
    resolved_args["hf_token"] = "<redacted>" if hf_token else ""
    data_cfg_for_save = json.loads(json.dumps(data_cfg)) if data_cfg else {}
    if isinstance(data_cfg_for_save.get("repos"), list):
        for repo in data_cfg_for_save["repos"]:
            if isinstance(repo, dict) and repo.get("token"):
                repo["token"] = "<redacted>"
    resolved_args["data"] = data_cfg_for_save

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_torch_backend(device)
    dtype = _resolve_dtype(args.dtype, device)
    print(f"Using device={device} dtype={dtype}")

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, token=hf_token)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    base_vocab = model.get_input_embeddings().num_embeddings
    audio_vocab = args.num_codebooks * args.cardinality
    n_special = 4
    audio_token_offset = base_vocab + n_special
    new_vocab = base_vocab + n_special + audio_vocab
    max_audio_token_id = audio_token_offset + audio_vocab - 1
    if new_vocab <= max_audio_token_id:
        raise RuntimeError(
            f"Vocab too small for audio token range: max_audio_token_id={max_audio_token_id}, "
            f"new_vocab={new_vocab}"
        )

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

    train_ds = MimiS2TTDataset(
        manifests=train_manifests,
        codes_root=codes_root,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        cardinality=args.cardinality,
        num_codebooks=args.num_codebooks,
        audio_token_offset=audio_token_offset,
        task_marker_id=task_marker_id,
        src_marker_id=src_marker_id,
        sep_marker_id=sep_marker_id,
        tgt_marker_id=tgt_marker_id,
        eos_id=eos_id,
        target_text_key=args.target_text_key,
        bos_id=bos_id,
        truncate_policy=args.truncate_policy,
    )
    valid_ds = None
    if valid_manifests:
        valid_ds = MimiS2TTDataset(
            manifests=valid_manifests,
            codes_root=codes_root,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            cardinality=args.cardinality,
            num_codebooks=args.num_codebooks,
            audio_token_offset=audio_token_offset,
            task_marker_id=task_marker_id,
            src_marker_id=src_marker_id,
            sep_marker_id=sep_marker_id,
            tgt_marker_id=tgt_marker_id,
            eos_id=eos_id,
            target_text_key=args.target_text_key,
            bos_id=bos_id,
            truncate_policy=args.truncate_policy,
        )

    collator = CausalCollator(pad_id=pad_id)
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = args.persistent_workers
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    valid_loader = None
    if valid_ds is not None:
        valid_loader_kwargs: dict[str, Any] = {
            "batch_size": args.eval_batch_size,
            "shuffle": False,
            "collate_fn": collator,
            "num_workers": args.num_workers,
            "pin_memory": (device.type == "cuda"),
        }
        if args.num_workers > 0:
            valid_loader_kwargs["persistent_workers"] = args.persistent_workers
            valid_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        valid_loader = DataLoader(valid_ds, **valid_loader_kwargs)

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
                        training_args=resolved_args,
                        mapping={
                            "base_model": args.base_model,
                            "target_kind": "text",
                            "target_text_key": args.target_text_key,
                            "audio_token_offset": audio_token_offset,
                            "text_vocab_size": base_vocab,
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
        training_args=resolved_args,
        mapping={
            "base_model": args.base_model,
            "target_kind": "text",
            "target_text_key": args.target_text_key,
            "audio_token_offset": audio_token_offset,
            "text_vocab_size": base_vocab,
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
