#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch

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


def _save_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    sf = _try_import_soundfile()
    wav = wav.detach().cpu()
    if sf is not None:
        sf.write(str(path), wav.t().numpy(), sample_rate)
        return

    ta = _try_import_torchaudio()
    if ta is None:
        raise RuntimeError(
            "No audio backend found. Install either `soundfile` or `torchaudio`."
        )
    ta.save(str(path), wav, sample_rate)


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    ta = _try_import_torchaudio()
    if ta is not None:
        return ta.functional.resample(wav, sr, target_sr)

    # Fallback: linear interpolation with torch only.
    # wav: [C, T] -> [1, C, T] for interpolate, then squeeze back.
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


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _pad_to_frame(wav: torch.Tensor, frame_size: int) -> tuple[torch.Tensor, int, int]:
    length = wav.shape[-1]
    frames = math.ceil(length / frame_size)
    target_len = frames * frame_size
    if target_len == length:
        return wav, frames, 0
    pad = target_len - length
    wav = torch.nn.functional.pad(wav, (0, pad), mode="constant")
    return wav, frames, pad


def _default_tokens_path(output_audio: Path) -> Path:
    return output_audio.with_suffix(output_audio.suffix + ".mimi.pt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Streaming Mimi round-trip for a single audio file: "
            "audio -> tokens -> reconstructed audio."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Input audio file path.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output reconstructed audio path (e.g. out.wav).",
    )
    parser.add_argument(
        "--tokens-output",
        type=str,
        default="",
        help="Optional token output path (.pt). Default: <output>.mimi.pt",
    )
    parser.add_argument(
        "--mimi",
        type=str,
        default="",
        help="Path to Mimi weights (.safetensors) or empty to download from HF repo.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo for Mimi weights when --mimi is not provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.device_count() else "cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=8,
        help="Number of active codebooks for Mimi.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    tokens_path = Path(args.tokens_output) if args.tokens_output else _default_tokens_path(out_path)

    if not in_path.exists():
        raise SystemExit(f"Input audio not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mimi:
        mimi_path = Path(args.mimi)
        if not mimi_path.exists():
            raise SystemExit(f"Mimi weights not found: {mimi_path}")
    else:
        mimi_path = Path(hf_hub_download(args.hf_repo, loaders.MIMI_NAME))

    device = torch.device(args.device)
    mimi = loaders.get_mimi(mimi_path, device=device, num_codebooks=args.num_codebooks)
    mimi.eval()

    wav, sr = _load_audio(in_path)
    wav = _to_mono(wav)
    wav = _resample_if_needed(wav, sr, mimi.sample_rate)
    original_samples = wav.shape[-1]
    wav, padded_frames, pad_samples = _pad_to_frame(wav, mimi.frame_size)
    wav = wav.unsqueeze(0).to(device=device)  # [1, 1, T]

    with torch.no_grad():
        encoded_chunks: list[torch.Tensor] = []
        with mimi.streaming(batch_size=1):
            for offset in range(0, wav.shape[-1], mimi.frame_size):
                frame = wav[:, :, offset: offset + mimi.frame_size]
                codes = mimi.encode(frame)
                if codes.shape[-1] > 0:
                    encoded_chunks.append(codes)

        if not encoded_chunks:
            raise RuntimeError("No codes were produced by streaming encoder.")
        all_codes = torch.cat(encoded_chunks, dim=-1)  # [1, K, frames]

        decoded_chunks: list[torch.Tensor] = []
        with mimi.streaming(batch_size=1):
            for i in range(all_codes.shape[-1]):
                code_step = all_codes[:, :, i:i + 1]
                pcm = mimi.decode(code_step)
                if pcm.shape[-1] > 0:
                    decoded_chunks.append(pcm)
        if not decoded_chunks:
            raise RuntimeError("No audio was produced by streaming decoder.")
        decoded = torch.cat(decoded_chunks, dim=-1)[:, :, :original_samples]

    decoded_wav = decoded[0]
    _save_audio(out_path, decoded_wav, mimi.sample_rate)

    payload = {
        "codes": all_codes.cpu().short(),
        "num_codebooks": mimi.num_codebooks,
        "cardinality": mimi.cardinality,
        "frame_rate": mimi.frame_rate,
        "sample_rate": mimi.sample_rate,
        "source": str(in_path),
        "original_samples": original_samples,
        "padded_frames": padded_frames,
        "pad_samples": pad_samples,
    }
    torch.save(payload, tokens_path)

    print(f"Input:   {in_path}")
    print(f"Tokens:  {tokens_path} (shape={tuple(all_codes.shape)})")
    print(f"Output:  {out_path} (samples={decoded_wav.shape[-1]}, sr={mimi.sample_rate})")


if __name__ == "__main__":
    main()
