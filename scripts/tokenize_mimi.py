#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Batch-tokenize audio files with Mimi and save discrete codes."
    )
    parser.add_argument("--input", type=str, required=True, help="File or directory.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--exts", type=str, default="wav,flac,mp3,ogg",
                        help="Comma-separated list of audio extensions.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--mimi", type=str, default="",
                        help="Path to Mimi weights (.safetensors) or empty to download.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo for Mimi weights if --mimi not provided.")
    parser.add_argument("--target-sr", type=int, default=24000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    files = _iter_audio_files(input_path, exts)
    if not files:
        raise SystemExit(f"No audio files found under {input_path}")

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

    def _save_codes(src: Path, codes: torch.Tensor, frames: int):
        rel = src.name
        out_path = output_dir / f"{rel}.mimi.pt"
        payload = {
            "codes": codes[:, :, :frames].cpu().short(),
            "num_codebooks": mimi.num_codebooks,
            "cardinality": mimi.cardinality,
            "frame_rate": mimi.frame_rate,
            "sample_rate": target_sr,
            "source": str(src),
        }
        torch.save(payload, out_path)

    batch = []
    for idx, path in enumerate(files, start=1):
        wav, sr = _load_audio(path)
        wav = _to_mono(wav)
        wav = _resample_if_needed(wav, sr, target_sr)
        wav, frames = _pad_to_frame(wav, frame_size)
        batch.append((path, wav, frames))

        if len(batch) == args.batch_size or idx == len(files):
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
            for p, fr, code in zip(paths, frames_list, codes):
                _save_codes(p, code.unsqueeze(0), fr)
            batch = []

    print(f"Tokenized {len(files)} files into {output_dir}")


if __name__ == "__main__":
    main()
