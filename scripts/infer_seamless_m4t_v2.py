#!/usr/bin/env python3

import argparse
import io
import subprocess
from pathlib import Path

import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model


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


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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


def _configure_torch_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _load_audio_ffmpeg(path: Path, target_sr: int) -> torch.Tensor:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = torch.frombuffer(proc.stdout, dtype=torch.float32)
    if audio.numel() == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples from {path}")
    return audio.contiguous()


def _load_audio(path: Path, target_sr: int) -> torch.Tensor:
    sf = _try_import_soundfile()
    if sf is not None:
        try:
            wav, sr = sf.read(str(path), always_2d=True)
            wav = torch.from_numpy(wav).t().to(dtype=torch.float32).contiguous()
            wav = wav.mean(dim=0)
            if sr != target_sr:
                ta = _try_import_torchaudio()
                if ta is not None:
                    wav = ta.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
                else:
                    return _load_audio_ffmpeg(path, target_sr)
            return wav.contiguous()
        except Exception:
            pass

    ta = _try_import_torchaudio()
    if ta is not None:
        try:
            wav, sr = ta.load(str(path))
            wav = wav.to(dtype=torch.float32).mean(dim=0)
            if sr != target_sr:
                wav = ta.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
            return wav.contiguous()
        except Exception:
            pass

    return _load_audio_ffmpeg(path, target_sr)


def _save_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf = _try_import_soundfile()
    wav = wav.detach().cpu().to(dtype=torch.float32)
    if sf is not None:
        sf.write(str(path), wav.numpy(), sample_rate)
        return
    ta = _try_import_torchaudio()
    if ta is not None:
        ta.save(str(path), wav.unsqueeze(0), sample_rate)
        return
    raise RuntimeError("No audio backend found to save output audio.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run separate SeamlessM4T v2 inference without touching the Mimi/Qwen training pipeline. "
            "Supports Luganda->English speech-to-text or speech-to-speech."
        )
    )
    parser.add_argument("--model-id", type=str, default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--src-audio", type=str, default="")
    parser.add_argument("--src-text", type=str, default="")
    parser.add_argument("--src-lang", type=str, default="lug")
    parser.add_argument("--tgt-lang", type=str, default="eng")
    parser.add_argument(
        "--task",
        type=str,
        default="s2tt",
        choices=["s2tt", "s2st", "t2tt", "t2st"],
        help="Speech/text input and text/speech output mode.",
    )
    parser.add_argument("--output-text", type=str, default="")
    parser.add_argument("--output-wav", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    if bool(args.src_audio) == bool(args.src_text):
        raise SystemExit("Provide exactly one of --src-audio or --src-text.")

    expects_audio = args.task.startswith("s2")
    if expects_audio and not args.src_audio:
        raise SystemExit(f"{args.task} requires --src-audio.")
    if not expects_audio and not args.src_text:
        raise SystemExit(f"{args.task} requires --src-text.")

    generate_speech = args.task.endswith("st")
    if generate_speech and not args.output_wav:
        raise SystemExit(f"{args.task} requires --output-wav.")
    if not generate_speech and not args.output_text:
        raise SystemExit(f"{args.task} requires --output-text.")

    device = _resolve_device(args.device)
    _configure_torch_backend(device)
    dtype = _resolve_dtype(args.dtype, device)
    print(f"Using device={device} dtype={dtype}")
    print(f"Loading model={args.model_id}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = SeamlessM4Tv2Model.from_pretrained(args.model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    if expects_audio:
        target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
        audio = _load_audio(Path(args.src_audio), target_sr=target_sr)
        inputs = processor(audios=audio, sampling_rate=target_sr, return_tensors="pt")
    else:
        inputs = processor(text=args.src_text, src_lang=args.src_lang, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            tgt_lang=args.tgt_lang,
            generate_speech=generate_speech,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )

    if generate_speech:
        if isinstance(outputs, tuple):
            wav = outputs[0][0].detach().cpu().to(dtype=torch.float32)
        else:
            wav = outputs[0].detach().cpu().to(dtype=torch.float32)
        sample_rate = int(getattr(model.config, "sampling_rate", 16000))
        _save_audio(Path(args.output_wav), wav, sample_rate)
        print(f"Saved translated audio to {args.output_wav}")
        return

    if hasattr(outputs, "sequences"):
        seq = outputs.sequences[0]
    else:
        seq = outputs[0]
        if seq.dim() > 1:
            seq = seq[0]
    text = processor.decode(seq.tolist(), skip_special_tokens=True)
    Path(args.output_text).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_text).write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Saved translated text to {args.output_text}")


if __name__ == "__main__":
    main()
