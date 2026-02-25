#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
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


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    sf = _try_import_soundfile()
    if sf is not None:
        wav, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(wav).t().contiguous()  # [C, T]
        return wav, sr

    ta = _try_import_torchaudio()
    if ta is None:
        raise RuntimeError(
            "No audio backend found. Install either `soundfile` or `torchaudio`."
        )
    wav, sr = ta.load(str(path))  # [C, T]
    return wav, sr


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    ta = _try_import_torchaudio()
    if ta is None:
        raise RuntimeError(
            "Resampling requires `torchaudio`. Please install it or resample ahead of time."
        )
    return ta.functional.resample(wav, sr, target_sr)


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _iter_audio_files(root: Path, exts: Iterable[str]) -> list[Path]:
    exts = [e.lower().lstrip(".") for e in exts]
    files: list[Path] = []
    if root.is_file():
        return [root]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            files.append(p)
    return sorted(files)


def _pad_to_frame(wav: torch.Tensor, frame_size: int) -> tuple[torch.Tensor, int]:
    length = wav.shape[-1]
    frames = math.ceil(length / frame_size)
    target_len = frames * frame_size
    if target_len == length:
        return wav, frames
    pad = target_len - length
    wav = torch.nn.functional.pad(wav, (0, pad), mode="constant")
    return wav, frames


def _rel_key(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    return str(path.relative_to(root))


def _save_codes(
    out_path: Path,
    codes: torch.Tensor,
    frames: int,
    num_codebooks: int,
    cardinality: int,
    frame_rate: float,
    sample_rate: int,
    source: Path,
):
    payload = {
        "codes": codes[:, :, :frames].cpu().short(),
        "num_codebooks": num_codebooks,
        "cardinality": cardinality,
        "frame_rate": frame_rate,
        "sample_rate": sample_rate,
        "source": str(source),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def _tokenize_batch(
    mimi,
    batch: list[tuple[Path, torch.Tensor, int]],
    out_dir: Path,
    rel_root: Path,
    suffix: str,
    device: torch.device,
    num_codebooks: int,
    cardinality: int,
    frame_rate: float,
    sample_rate: int,
):
    max_len = max(w.shape[-1] for _, w, _ in batch)
    padded = []
    frames_list = []
    paths = []
    for p, w, fr in batch:
        if w.shape[-1] < max_len:
            w = torch.nn.functional.pad(
                w, (0, max_len - w.shape[-1]), mode="constant"
            )
        padded.append(w)
        frames_list.append(fr)
        paths.append(p)

    batch_wav = torch.stack(padded, dim=0).to(device=device)
    with torch.no_grad():
        codes = mimi.encode(batch_wav)

    out_paths = []
    for p, fr, code in zip(paths, frames_list, codes):
        rel = _rel_key(p, rel_root)
        out_path = out_dir / f"{rel}.{suffix}.pt"
        _save_codes(
            out_path,
            code.unsqueeze(0),
            fr,
            num_codebooks,
            cardinality,
            frame_rate,
            sample_rate,
            p,
        )
        out_paths.append(out_path)
    return out_paths


def main():
    parser = argparse.ArgumentParser(
        description="Build a Mimi-tokenized dataset manifest (JSONL)."
    )
    parser.add_argument("--src", type=str, required=True,
                        help="Source audio file or directory.")
    parser.add_argument("--tgt", type=str, default="",
                        help="Target audio file or directory (optional).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for codes + manifest.")
    parser.add_argument("--codes-dir", type=str, default="codes",
                        help="Subdir under output to place token files.")
    parser.add_argument("--manifest", type=str, default="dataset.jsonl",
                        help="Manifest filename under output.")
    parser.add_argument("--exts", type=str, default="wav,flac,mp3,ogg",
                        help="Comma-separated list of audio extensions.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--mimi", type=str, default="",
                        help="Path to Mimi weights or empty to download.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo for Mimi weights if --mimi not provided.")
    parser.add_argument("--target-sr", type=int, default=24000)
    args = parser.parse_args()

    src_root = Path(args.src)
    tgt_root = Path(args.tgt) if args.tgt else None
    output_dir = Path(args.output)
    codes_dir = output_dir / args.codes_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    src_files = _iter_audio_files(src_root, exts)
    if not src_files:
        raise SystemExit(f"No source audio files found under {src_root}")

    tgt_map = {}
    if tgt_root is not None:
        tgt_files = _iter_audio_files(tgt_root, exts)
        if not tgt_files:
            raise SystemExit(f"No target audio files found under {tgt_root}")
        for p in tgt_files:
            tgt_map[_rel_key(p, tgt_root)] = p

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
    target_sr = args.target_sr

    manifest_path = output_dir / args.manifest
    stats = {
        "items": 0,
        "missing_targets": 0,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        batch = []
        pending = []
        for idx, src in enumerate(src_files, start=1):
            rel = _rel_key(src, src_root)
            tgt = tgt_map.get(rel) if tgt_root is not None else None
            if tgt_root is not None and tgt is None:
                stats["missing_targets"] += 1
                continue

            wav, sr = _load_audio(src)
            wav = _to_mono(wav)
            wav = _resample_if_needed(wav, sr, target_sr)
            wav, frames = _pad_to_frame(wav, frame_size)
            batch.append((src, wav, frames))
            pending.append((rel, src, tgt, frames))

            if len(batch) == args.batch_size or idx == len(src_files):
                out_paths = _tokenize_batch(
                    mimi,
                    batch,
                    codes_dir / "src",
                    src_root,
                    "src",
                    device,
                    mimi.num_codebooks,
                    mimi.cardinality,
                    mimi.frame_rate,
                    target_sr,
                )

                tgt_out_paths = None
                if tgt_root is not None:
                    tgt_batch = []
                    for (rel_key, _src_path, tgt_path, _src_frames) in pending:
                        assert tgt_path is not None
                        wav_t, sr_t = _load_audio(tgt_path)
                        wav_t = _to_mono(wav_t)
                        wav_t = _resample_if_needed(wav_t, sr_t, target_sr)
                        wav_t, frames_t = _pad_to_frame(wav_t, frame_size)
                        tgt_batch.append((tgt_path, wav_t, frames_t))
                    tgt_out_paths = _tokenize_batch(
                        mimi,
                        tgt_batch,
                        codes_dir / "tgt",
                        tgt_root,
                        "tgt",
                        device,
                        mimi.num_codebooks,
                        mimi.cardinality,
                        mimi.frame_rate,
                        target_sr,
                    )

                for i, (rel_key, src_path, tgt_path, src_frames) in enumerate(pending):
                    item = {
                        "id": rel_key,
                        "src_audio": str(src_path),
                        "src_codes": str(out_paths[i]),
                        "src_frames": src_frames,
                        "sample_rate": target_sr,
                        "frame_rate": mimi.frame_rate,
                        "num_codebooks": mimi.num_codebooks,
                        "cardinality": mimi.cardinality,
                    }
                    if tgt_root is not None:
                        item["tgt_audio"] = str(tgt_path)
                        assert tgt_out_paths is not None
                        item["tgt_codes"] = str(tgt_out_paths[i])
                        item["tgt_frames"] = tgt_batch[i][2]
                    f.write(json.dumps(item, ensure_ascii=True) + "\n")
                    stats["items"] += 1

                batch = []
                pending = []

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {stats['items']} items to {manifest_path}")
    if stats["missing_targets"] > 0:
        print(f"Skipped {stats['missing_targets']} items with missing targets")


if __name__ == "__main__":
    main()
