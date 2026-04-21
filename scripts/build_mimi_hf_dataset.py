#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Build a Mimi-tokenized speech translation dataset from Hugging Face audio data.

This script is the shared data-preparation entrypoint for both of the PEFT flows
we use in this repo:

- speech-to-speech (STS): Luganda speech in, English speech out
- speech-to-text translation (STT): Luganda speech in, English text out

The generated dataset contains both the source/target Mimi code tensors and the
aligned source/target text fields so the downstream STS and STT trainers can
reuse the same manifests.
"""

import argparse
import concurrent.futures
import io
import json
import math
import re
import tempfile
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, load_dataset
from huggingface_hub import HfApi, hf_hub_download

from moshi.models import loaders


def _try_import_soundfile():
    """Import `soundfile` lazily so we can fall back to torchaudio when absent."""
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        return None
    return sf


def _try_import_torchaudio():
    """Import `torchaudio` lazily for optional decode and resample support."""
    try:
        import torchaudio  # type: ignore
    except Exception:
        return None
    return torchaudio


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    """Collapse multi-channel audio to mono using a simple channel mean."""
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """Resample a waveform to the Mimi sample rate when the source rate differs."""
    if sr == target_sr:
        return wav
    ta = _try_import_torchaudio()
    if ta is not None:
        return ta.functional.resample(wav, sr, target_sr)

    # Fallback: linear interpolation with torch only.
    in_wav = wav.to(dtype=torch.float32).unsqueeze(0)
    target_len = int(round(wav.shape[-1] * float(target_sr) / float(sr)))
    if target_len <= 0:
        raise RuntimeError(f"Invalid resample target length: {target_len}")
    out_wav = torch.nn.functional.interpolate(
        in_wav,
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return out_wav.squeeze(0)


def _pad_to_frame(wav: torch.Tensor, frame_size: int) -> tuple[torch.Tensor, int, int]:
    """Pad a waveform so its length is an exact multiple of the Mimi frame size."""
    length = wav.shape[-1]
    frames = math.ceil(length / frame_size)
    target_len = frames * frame_size
    if target_len == length:
        return wav, frames, 0
    pad = target_len - length
    wav = torch.nn.functional.pad(wav, (0, pad), mode="constant")
    return wav, frames, pad


def _load_audio(path: str | None = None, raw_bytes: bytes | None = None) -> tuple[torch.Tensor, int]:
    """Decode audio from a file path or raw bytes into a `[channels, time]` tensor."""
    sf = _try_import_soundfile()
    # Prefer raw bytes when present. Some HF audio entries expose a relative path
    # (e.g. "1.wav") that is not resolvable locally.
    if raw_bytes is not None:
        if sf is None:
            raise RuntimeError(
                "Audio bytes decoding requires `soundfile` when no file path is available."
            )
        if isinstance(raw_bytes, memoryview):
            raw_bytes = raw_bytes.tobytes()
        elif not isinstance(raw_bytes, (bytes, bytearray)):
            raw_bytes = bytes(raw_bytes)
        wav, sr = sf.read(io.BytesIO(raw_bytes), always_2d=True)
        wav = torch.from_numpy(wav).t().contiguous()  # [C, T]
        return wav, int(sr)

    if path:
        if sf is not None:
            wav, sr = sf.read(path, always_2d=True)
            wav = torch.from_numpy(wav).t().contiguous()  # [C, T]
            return wav, int(sr)
        ta = _try_import_torchaudio()
        if ta is None:
            raise RuntimeError(
                "No audio backend found. Install either `soundfile` or `torchaudio`."
            )
        wav, sr = ta.load(path)  # [C, T]
        return wav, int(sr)

    raise RuntimeError("Audio object had neither path nor bytes.")


def _audio_to_tensor(audio_obj: Any) -> tuple[torch.Tensor, int]:
    """Normalize dataset audio objects into a float tensor and sample rate.

    The Hugging Face `datasets` library can surface audio in several different
    representations depending on decode mode and version. This helper accepts
    the supported variants and converts them into one consistent tensor layout.
    """
    arr = None
    sr = None

    if isinstance(audio_obj, dict):
        arr = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate")
        if arr is None or sr is None:
            # `Audio(decode=False)` path/bytes format.
            path = audio_obj.get("path")
            raw_bytes = audio_obj.get("bytes")
            wav, out_sr = _load_audio(path=path, raw_bytes=raw_bytes)
            return wav.to(dtype=torch.float32).contiguous(), out_sr
    elif hasattr(audio_obj, "get_all_samples"):
        # Newer `datasets` may return an AudioDecoder object.
        samples = audio_obj.get_all_samples()
        if isinstance(samples, dict):
            arr = samples.get("data", samples.get("array"))
            sr = samples.get("sample_rate", samples.get("sampling_rate"))
        else:
            arr = getattr(samples, "data", None)
            sr = getattr(samples, "sample_rate", None)
    elif isinstance(audio_obj, (tuple, list)) and len(audio_obj) == 2:
        arr, sr = audio_obj

    if arr is None or sr is None:
        raise RuntimeError(
            f"Unsupported audio object type: {type(audio_obj)}. "
            "Expected dict with array/sampling_rate or decoder with get_all_samples()."
        )

    arr = torch.as_tensor(arr)
    sr = int(sr)

    if arr.dim() == 1:
        wav = arr.unsqueeze(0)
    elif arr.dim() == 2:
        # Accept either [T, C] or [C, T].
        if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
            wav = arr.t()
        else:
            wav = arr
    else:
        raise RuntimeError(f"Unsupported audio array shape: {tuple(arr.shape)}")

    return wav.to(dtype=torch.float32).contiguous(), sr


def _save_codes(
    path: Path,
    codes: torch.Tensor,
    frames: int,
    sample_rate: int,
    frame_rate: float,
    num_codebooks: int,
    cardinality: int,
    source: str,
):
    """Persist Mimi code tensors together with the metadata needed downstream."""
    payload = {
        "codes": codes[:, :, :frames].cpu().short(),
        "num_frames": int(frames),
        "num_codebooks": num_codebooks,
        "cardinality": cardinality,
        "frame_rate": frame_rate,
        "sample_rate": sample_rate,
        "source": source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _encode_batch(mimi, wavs: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Pad a batch of waveforms and run Mimi encoding on the selected device."""
    max_len = max(w.shape[-1] for w in wavs)
    padded = []
    for w in wavs:
        if w.shape[-1] < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.shape[-1]), mode="constant")
        padded.append(w)
    batch_wav = torch.stack(padded, dim=0).contiguous()
    if device.type == "cuda":
        batch_wav = batch_wav.pin_memory()
    batch_wav = batch_wav.to(device=device, non_blocking=(device.type == "cuda"))
    with torch.no_grad():
        return mimi.encode(batch_wav)


def _sanitize_key(raw: str) -> str:
    """Convert a record id into a filesystem-safe basename."""
    key = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")
    return key or "item"


def _split_list(arg: str) -> list[str]:
    """Split a comma-separated CLI argument into trimmed entries."""
    return [x.strip() for x in arg.split(",") if x.strip()]


def _resolve_device(device_arg: str) -> torch.device:
    """Resolve `auto`, `cpu`, or `cuda` into a concrete torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _configure_torch_backend(device: torch.device) -> None:
    """Enable CUDA backend settings that improve Mimi encoding throughput."""
    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _default_repo_id(dataset_name: str) -> str:
    """Derive a default output dataset repo id from the input HF dataset name."""
    if "/" in dataset_name:
        namespace, name = dataset_name.split("/", 1)
        return f"{namespace}/{name}_mimi_token_version"
    return f"{dataset_name}_mimi_token_version"


def _write_dataset_card(
    output_dir: Path,
    dataset_name: str,
    dataset_cfg: str | None,
    repo_id: str,
    split_names: list[str],
    stats: dict[str, Any],
) -> None:
    """Write a short dataset card describing the exported Mimi-token layout."""
    config_line = dataset_cfg if dataset_cfg else "(default)"
    lines = [
        "# Mimi Token Dataset",
        "",
        f"- Source dataset: `{dataset_name}`",
        f"- Source config: `{config_line}`",
        f"- Generated repo: `{repo_id}`",
        f"- Splits: `{', '.join(split_names)}`",
        f"- Total items: `{stats['total_items']}`",
        "",
        "## Layout",
        "",
        "- `dataset.<split>.jsonl`: manifest for each split",
        "- `codes/lug/<split>/*.lug.pt`: source Mimi tokens",
        "- `codes/eng/<split>/*.eng.pt`: target Mimi tokens",
        "- `stats.json`: aggregate build stats",
        "- `build_config.json`: build-time settings",
        "",
        "The manifest paths are relative to the dataset repo root so they stay valid after `snapshot_download`.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_item(
    *,
    idx: int,
    row: dict[str, Any],
    split: str,
    id_col: str,
    src_audio_col: str,
    tgt_audio_col: str,
    src_text_col: str,
    tgt_text_col: str,
    sample_rate: int,
    frame_size: int,
) -> dict[str, Any]:
    """Decode and normalize one dataset row ahead of batched Mimi encoding.

    This stage performs the CPU-heavy work for a single example:

    - load source and target audio
    - convert to mono
    - resample to Mimi's sample rate
    - pad to full Mimi frames
    - collect aligned text metadata
    """
    row_id = row.get(id_col, f"{split}-{idx:09d}")
    row_id_str = str(row_id)
    safe_id = _sanitize_key(f"{split}-{idx:09d}-{row_id_str}")

    src_wav, src_sr = _audio_to_tensor(row[src_audio_col])
    src_wav = _to_mono(src_wav)
    src_wav = _resample_if_needed(src_wav, src_sr, sample_rate)
    src_wav, src_frames, src_pad = _pad_to_frame(src_wav, frame_size)

    tgt_wav, tgt_sr = _audio_to_tensor(row[tgt_audio_col])
    tgt_wav = _to_mono(tgt_wav)
    tgt_wav = _resample_if_needed(tgt_wav, tgt_sr, sample_rate)
    tgt_wav, tgt_frames, tgt_pad = _pad_to_frame(tgt_wav, frame_size)

    return {
        "idx": idx,
        "id": row_id_str,
        "safe_id": safe_id,
        "src_wav": src_wav.contiguous(),
        "tgt_wav": tgt_wav.contiguous(),
        "src_frames": src_frames,
        "tgt_frames": tgt_frames,
        "src_pad": src_pad,
        "tgt_pad": tgt_pad,
        "src_text": str(row.get(src_text_col, "")),
        "tgt_text": str(row.get(tgt_text_col, "")),
    }


def _flush_pending(
    mf,
    pending: list[dict[str, Any]],
    split: str,
    output_dir: Path,
    codes_dir_name: str,
    dataset_name: str,
    src_audio_col: str,
    tgt_audio_col: str,
    mimi,
    device: torch.device,
) -> int:
    """Encode a pending batch, write Mimi code files, and append manifest rows."""
    if not pending:
        return 0

    src_codes = _encode_batch(mimi, [it["src_wav"] for it in pending], device=device)
    tgt_codes = _encode_batch(mimi, [it["tgt_wav"] for it in pending], device=device)

    for i, item in enumerate(pending):
        src_rel = Path(codes_dir_name) / "lug" / split / f"{item['safe_id']}.lug.pt"
        tgt_rel = Path(codes_dir_name) / "eng" / split / f"{item['safe_id']}.eng.pt"
        src_path = output_dir / src_rel
        tgt_path = output_dir / tgt_rel

        _save_codes(
            src_path,
            src_codes[i].unsqueeze(0),
            item["src_frames"],
            mimi.sample_rate,
            mimi.frame_rate,
            mimi.num_codebooks,
            mimi.cardinality,
            f"{dataset_name}:{split}:{item['idx']}:{src_audio_col}",
        )
        _save_codes(
            tgt_path,
            tgt_codes[i].unsqueeze(0),
            item["tgt_frames"],
            mimi.sample_rate,
            mimi.frame_rate,
            mimi.num_codebooks,
            mimi.cardinality,
            f"{dataset_name}:{split}:{item['idx']}:{tgt_audio_col}",
        )

        record = {
            "split": split,
            "id": item["id"],
            "src_text": item["src_text"],
            "tgt_text": item["tgt_text"],
            "src_codes": str(src_rel),
            "tgt_codes": str(tgt_rel),
            "src_frames": item["src_frames"],
            "tgt_frames": item["tgt_frames"],
            "src_pad_samples": item["src_pad"],
            "tgt_pad_samples": item["tgt_pad"],
            "sample_rate": mimi.sample_rate,
            "frame_rate": mimi.frame_rate,
            "num_codebooks": mimi.num_codebooks,
            "cardinality": mimi.cardinality,
        }
        mf.write(json.dumps(record, ensure_ascii=True) + "\n")

    return len(pending)


def main() -> None:
    """Run the end-to-end dataset build and optional Hugging Face upload flow."""
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a Hugging Face speech-translation dataset with Mimi. "
            "Expected columns: audio_lug, audio_eng, text_lug, text_eng."
        )
    )
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset id.")
    parser.add_argument("--config", type=str, default="", help="HF dataset config name.")
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Comma-separated splits to process, or `all`.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional local staging directory. If omitted, a temporary directory is used.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="",
        help="Target HF dataset repo. Defaults to <dataset>_mimi_token_version.",
    )
    parser.add_argument("--codes-dir", type=str, default="codes", help="Codes subdirectory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=4,
        help=(
            "Number of CPU worker threads used to load/decode/resample audio ahead of GPU Mimi encoding. "
            "Set to 0 or 1 to disable parallel preprocessing."
        ),
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="How many future GPU batches to prepare on CPU ahead of time.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run Mimi encoding on: cuda, cpu, or auto.",
    )
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--src-audio-col", type=str, default="audio_lug")
    parser.add_argument("--tgt-audio-col", type=str, default="audio_eng")
    parser.add_argument("--src-text-col", type=str, default="text_lug")
    parser.add_argument("--tgt-text-col", type=str, default="text_eng")
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="HF token for private datasets. If empty, uses local HF login/env.",
    )
    parser.add_argument(
        "--mimi",
        type=str,
        default="",
        help="Path to Mimi weights (.safetensors) or empty to download.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo for Mimi weights when --mimi is not provided.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the generated HF dataset repo as private.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Keep the tokenized dataset locally and skip Hugging Face repo upload.",
    )
    args = parser.parse_args()

    token = args.hf_token if args.hf_token else None
    repo_id = args.repo_id or _default_repo_id(args.dataset)
    dataset_name = args.dataset
    dataset_cfg = args.config if args.config else None

    if args.skip_upload and not args.output:
        raise SystemExit("When --skip-upload is set, provide --output so the local dataset is retained.")

    if args.splits == "all":
        ds_dict = load_dataset(dataset_name, dataset_cfg, token=token)
        split_names = list(ds_dict.keys())
    else:
        split_names = _split_list(args.splits)
        if not split_names:
            raise SystemExit("No valid split names provided.")

    if args.mimi:
        mimi_path = Path(args.mimi)
        if not mimi_path.exists():
            raise SystemExit(f"Mimi weights not found: {mimi_path}")
    else:
        mimi_path = Path(hf_hub_download(args.hf_repo, loaders.MIMI_NAME))

    device = _resolve_device(args.device)
    _configure_torch_backend(device)
    print(f"Using device: {device}")
    mimi = loaders.get_mimi(mimi_path, device=device, num_codebooks=args.num_codebooks)
    mimi.eval()
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    global_stats = {
        "dataset": dataset_name,
        "config": dataset_cfg,
        "repo_id": repo_id,
        "splits": {},
        "total_items": 0,
    }
    staging_ctx = nullcontext(args.output) if args.output else tempfile.TemporaryDirectory(
        prefix="mimi_hf_dataset_"
    )
    with staging_ctx as staging_dir:
        output_dir = Path(staging_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in split_names:
            ds = load_dataset(dataset_name, dataset_cfg, split=split, token=token)
            # Disable dataset-side decode to avoid hard dependency on librosa.
            ds = ds.cast_column(args.src_audio_col, Audio(decode=False))
            ds = ds.cast_column(args.tgt_audio_col, Audio(decode=False))

            manifest_path = output_dir / f"dataset.{split}.jsonl"
            stats = {"items": 0}

            with manifest_path.open("w", encoding="utf-8") as mf:
                pending: list[dict[str, Any]] = []
                row_iter = enumerate(ds)
                max_inflight = max(args.batch_size, args.batch_size * max(1, args.prefetch_batches))

                def submit_next(executor, queue: deque[concurrent.futures.Future]) -> bool:
                    """Queue one more preprocessing task while source rows remain."""
                    try:
                        idx, row = next(row_iter)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        _prepare_item,
                        idx=idx,
                        row=row,
                        split=split,
                        id_col=args.id_col,
                        src_audio_col=args.src_audio_col,
                        tgt_audio_col=args.tgt_audio_col,
                        src_text_col=args.src_text_col,
                        tgt_text_col=args.tgt_text_col,
                        sample_rate=mimi.sample_rate,
                        frame_size=frame_size,
                    )
                    queue.append(future)
                    return True

                if args.preprocess_workers > 1:
                    inflight: deque[concurrent.futures.Future] = deque()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=args.preprocess_workers) as executor:
                        while len(inflight) < max_inflight and submit_next(executor, inflight):
                            pass

                        while inflight:
                            item = inflight.popleft().result()
                            pending.append(item)
                            while len(inflight) < max_inflight and submit_next(executor, inflight):
                                pass

                            if len(pending) == args.batch_size:
                                stats["items"] += _flush_pending(
                                    mf=mf,
                                    pending=pending,
                                    split=split,
                                    output_dir=output_dir,
                                    codes_dir_name=args.codes_dir,
                                    dataset_name=dataset_name,
                                    src_audio_col=args.src_audio_col,
                                    tgt_audio_col=args.tgt_audio_col,
                                    mimi=mimi,
                                    device=device,
                                )
                                pending = []
                else:
                    for idx, row in row_iter:
                        pending.append(
                            _prepare_item(
                                idx=idx,
                                row=row,
                                split=split,
                                id_col=args.id_col,
                                src_audio_col=args.src_audio_col,
                                tgt_audio_col=args.tgt_audio_col,
                                src_text_col=args.src_text_col,
                                tgt_text_col=args.tgt_text_col,
                                sample_rate=mimi.sample_rate,
                                frame_size=frame_size,
                            )
                        )

                        if len(pending) == args.batch_size:
                            stats["items"] += _flush_pending(
                                mf=mf,
                                pending=pending,
                                split=split,
                                output_dir=output_dir,
                                codes_dir_name=args.codes_dir,
                                dataset_name=dataset_name,
                                src_audio_col=args.src_audio_col,
                                tgt_audio_col=args.tgt_audio_col,
                                mimi=mimi,
                                device=device,
                            )
                            pending = []

                stats["items"] += _flush_pending(
                    mf=mf,
                    pending=pending,
                    split=split,
                    output_dir=output_dir,
                    codes_dir_name=args.codes_dir,
                    dataset_name=dataset_name,
                    src_audio_col=args.src_audio_col,
                    tgt_audio_col=args.tgt_audio_col,
                    mimi=mimi,
                    device=device,
                )

            global_stats["splits"][split] = stats
            global_stats["total_items"] += stats["items"]
            print(f"[{split}] wrote {stats['items']} items to {manifest_path}")

        build_config = {
            "source_dataset": dataset_name,
            "source_config": dataset_cfg,
            "generated_repo_id": repo_id,
            "splits": split_names,
            "codes_dir": args.codes_dir,
            "id_col": args.id_col,
            "src_audio_col": args.src_audio_col,
            "tgt_audio_col": args.tgt_audio_col,
            "src_text_col": args.src_text_col,
            "tgt_text_col": args.tgt_text_col,
            "num_codebooks": mimi.num_codebooks,
            "cardinality": mimi.cardinality,
            "sample_rate": mimi.sample_rate,
            "frame_rate": mimi.frame_rate,
        }
        (output_dir / "build_config.json").write_text(
            json.dumps(build_config, indent=2), encoding="utf-8"
        )
        stats_path = output_dir / "stats.json"
        stats_path.write_text(json.dumps(global_stats, indent=2), encoding="utf-8")
        _write_dataset_card(output_dir, dataset_name, dataset_cfg, repo_id, split_names, global_stats)

        if args.skip_upload:
            print(f"Skipping upload. Local Mimi-tokenized dataset retained at {output_dir}")
            return

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=(
                f"Add Mimi-tokenized dataset for {dataset_name} ({global_stats['total_items']} items)"
            ),
        )
        print(f"Uploaded Mimi-tokenized dataset to {repo_id}")
        if args.output:
            print(f"Retained local staging directory at {output_dir}")


if __name__ == "__main__":
    main()
