#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download

from moshi.models import loaders


def _try_import_soundfile():
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        return None
    return sf


def _try_import_torchaudio():
    try:
        import torchaudio  # type: ignore
    except Exception:
        return None
    return torchaudio


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
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
    length = wav.shape[-1]
    frames = math.ceil(length / frame_size)
    target_len = frames * frame_size
    if target_len == length:
        return wav, frames, 0
    pad = target_len - length
    wav = torch.nn.functional.pad(wav, (0, pad), mode="constant")
    return wav, frames, pad


def _load_audio(path: str | None = None, raw_bytes: bytes | None = None) -> tuple[torch.Tensor, int]:
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
    payload = {
        "codes": codes[:, :, :frames].cpu().short(),
        "num_codebooks": num_codebooks,
        "cardinality": cardinality,
        "frame_rate": frame_rate,
        "sample_rate": sample_rate,
        "source": source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _encode_batch(mimi, wavs: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    max_len = max(w.shape[-1] for w in wavs)
    padded = []
    for w in wavs:
        if w.shape[-1] < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.shape[-1]), mode="constant")
        padded.append(w)
    batch_wav = torch.stack(padded, dim=0).to(device=device)
    with torch.no_grad():
        return mimi.encode(batch_wav)


def _sanitize_key(raw: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")
    return key or "item"


def _split_list(arg: str) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a Hugging Face speech-translation dataset with Mimi. "
            "Expected columns: audio_eng, audio_lug, text_eng, text_lug."
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
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--codes-dir", type=str, default="codes", help="Codes subdirectory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--src-audio-col", type=str, default="audio_eng")
    parser.add_argument("--tgt-audio-col", type=str, default="audio_lug")
    parser.add_argument("--src-text-col", type=str, default="text_eng")
    parser.add_argument("--tgt-text-col", type=str, default="text_lug")
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
    args = parser.parse_args()

    output_dir = Path(args.output)
    codes_dir = output_dir / args.codes_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    token = args.hf_token if args.hf_token else None
    dataset_name = args.dataset
    dataset_cfg = args.config if args.config else None

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

    device = torch.device(args.device)
    mimi = loaders.get_mimi(mimi_path, device=device, num_codebooks=args.num_codebooks)
    mimi.eval()
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    global_stats = {
        "dataset": dataset_name,
        "splits": {},
        "total_items": 0,
    }

    for split in split_names:
        ds = load_dataset(dataset_name, dataset_cfg, split=split, token=token)
        # Disable dataset-side decode to avoid hard dependency on librosa.
        ds = ds.cast_column(args.src_audio_col, Audio(decode=False))
        ds = ds.cast_column(args.tgt_audio_col, Audio(decode=False))

        manifest_path = output_dir / f"dataset.{split}.jsonl"
        stats = {"items": 0}

        with manifest_path.open("w", encoding="utf-8") as mf:
            pending: list[dict[str, Any]] = []

            for idx, row in enumerate(ds):
                row_id = row.get(args.id_col, f"{split}-{idx:09d}")
                row_id_str = str(row_id)
                safe_id = _sanitize_key(f"{split}-{idx:09d}-{row_id_str}")

                src_wav, src_sr = _audio_to_tensor(row[args.src_audio_col])
                src_wav = _to_mono(src_wav)
                src_wav = _resample_if_needed(src_wav, src_sr, mimi.sample_rate)
                src_wav, src_frames, src_pad = _pad_to_frame(src_wav, frame_size)

                tgt_wav, tgt_sr = _audio_to_tensor(row[args.tgt_audio_col])
                tgt_wav = _to_mono(tgt_wav)
                tgt_wav = _resample_if_needed(tgt_wav, tgt_sr, mimi.sample_rate)
                tgt_wav, tgt_frames, tgt_pad = _pad_to_frame(tgt_wav, frame_size)

                pending.append(
                    {
                        "idx": idx,
                        "id": row_id_str,
                        "safe_id": safe_id,
                        "src_wav": src_wav,
                        "tgt_wav": tgt_wav,
                        "src_frames": src_frames,
                        "tgt_frames": tgt_frames,
                        "src_pad": src_pad,
                        "tgt_pad": tgt_pad,
                        "src_text": str(row.get(args.src_text_col, "")),
                        "tgt_text": str(row.get(args.tgt_text_col, "")),
                    }
                )

                if len(pending) == args.batch_size:
                    src_codes = _encode_batch(
                        mimi, [it["src_wav"] for it in pending], device=device
                    )
                    tgt_codes = _encode_batch(
                        mimi, [it["tgt_wav"] for it in pending], device=device
                    )
                    for i, item in enumerate(pending):
                        src_rel = Path(args.codes_dir) / "eng" / split / f"{item['safe_id']}.eng.pt"
                        tgt_rel = Path(args.codes_dir) / "lug" / split / f"{item['safe_id']}.lug.pt"
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
                            f"{dataset_name}:{split}:{item['idx']}:{args.src_audio_col}",
                        )
                        _save_codes(
                            tgt_path,
                            tgt_codes[i].unsqueeze(0),
                            item["tgt_frames"],
                            mimi.sample_rate,
                            mimi.frame_rate,
                            mimi.num_codebooks,
                            mimi.cardinality,
                            f"{dataset_name}:{split}:{item['idx']}:{args.tgt_audio_col}",
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
                        stats["items"] += 1

                    pending = []

            if pending:
                src_codes = _encode_batch(mimi, [it["src_wav"] for it in pending], device=device)
                tgt_codes = _encode_batch(mimi, [it["tgt_wav"] for it in pending], device=device)
                for i, item in enumerate(pending):
                    src_rel = Path(args.codes_dir) / "eng" / split / f"{item['safe_id']}.eng.pt"
                    tgt_rel = Path(args.codes_dir) / "lug" / split / f"{item['safe_id']}.lug.pt"
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
                        f"{dataset_name}:{split}:{item['idx']}:{args.src_audio_col}",
                    )
                    _save_codes(
                        tgt_path,
                        tgt_codes[i].unsqueeze(0),
                        item["tgt_frames"],
                        mimi.sample_rate,
                        mimi.frame_rate,
                        mimi.num_codebooks,
                        mimi.cardinality,
                        f"{dataset_name}:{split}:{item['idx']}:{args.tgt_audio_col}",
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
                    stats["items"] += 1

        global_stats["splits"][split] = stats
        global_stats["total_items"] += stats["items"]
        print(f"[{split}] wrote {stats['items']} items to {manifest_path}")

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(global_stats, indent=2), encoding="utf-8")
    print(f"Wrote global stats to {stats_path}")


if __name__ == "__main__":
    main()
