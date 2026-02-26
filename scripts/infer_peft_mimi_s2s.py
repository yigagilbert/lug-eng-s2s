#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

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


def _save_audio(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    sf = _try_import_soundfile()
    wav = wav.detach().cpu()
    if sf is not None:
        sf.write(str(path), wav.t().numpy(), sample_rate)
        return
    ta = _try_import_torchaudio()
    if ta is None:
        raise RuntimeError("No audio backend found. Install either `soundfile` or `torchaudio`.")
    ta.save(str(path), wav, sample_rate)


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    sf = _try_import_soundfile()
    if sf is not None:
        wav, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(wav).t().contiguous()  # [C, T]
        return wav, int(sr)

    ta = _try_import_torchaudio()
    if ta is None:
        raise RuntimeError("No audio backend found. Install either `soundfile` or `torchaudio`.")
    wav, sr = ta.load(str(path))  # [C, T]
    return wav, int(sr)


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

    # Fallback: linear interpolation if torchaudio is unavailable.
    in_wav = wav.to(dtype=torch.float32).unsqueeze(0)  # [1, C, T]
    target_len = int(round(wav.shape[-1] * float(target_sr) / float(sr)))
    if target_len <= 0:
        raise RuntimeError(f"Invalid resample target length: {target_len}")
    out_wav = torch.nn.functional.interpolate(
        in_wav, size=target_len, mode="linear", align_corners=False
    )
    return out_wav.squeeze(0)


def _pad_to_frame(wav: torch.Tensor, frame_size: int) -> tuple[torch.Tensor, int]:
    length = wav.shape[-1]
    frames = math.ceil(length / frame_size)
    target_len = frames * frame_size
    if target_len == length:
        return wav, frames
    pad = target_len - length
    wav = torch.nn.functional.pad(wav, (0, pad), mode="constant")
    return wav, frames


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


def _load_mapping(mapping_path: Path) -> dict[str, Any]:
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def _load_codes(path: Path, num_codebooks: int, cardinality: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    codes = payload["codes"] if isinstance(payload, dict) and "codes" in payload else payload
    codes = torch.as_tensor(codes)
    if codes.dim() == 3:
        codes = codes.squeeze(0)
    if codes.dim() != 2:
        raise RuntimeError(f"Unexpected code shape {tuple(codes.shape)} in {path}")
    if codes.shape[0] < num_codebooks:
        raise RuntimeError(
            f"Need at least {num_codebooks} codebooks, found {codes.shape[0]} in {path}"
        )
    codes = codes[:num_codebooks].long()
    if codes.min().item() < 0 or codes.max().item() >= cardinality:
        raise RuntimeError(
            f"Code values out of range in {path} for cardinality={cardinality}: "
            f"min={codes.min().item()} max={codes.max().item()}"
        )
    return codes


def _pack_audio_codes(codes: torch.Tensor, cardinality: int, audio_token_offset: int) -> torch.Tensor:
    # [K, T] -> [T*K], same layout used during training.
    k = codes.shape[0]
    offsets = (torch.arange(k).unsqueeze(1) * cardinality).long()
    packed = (codes + offsets).transpose(0, 1).reshape(-1)
    return packed + audio_token_offset


def _extract_audio_tokens(
    generated: torch.Tensor,
    audio_token_offset: int,
    num_codebooks: int,
    cardinality: int,
    eos_id: int | None,
) -> torch.Tensor:
    audio_low = audio_token_offset
    audio_high = audio_token_offset + (num_codebooks * cardinality)
    out: list[int] = []
    for tok in generated.tolist():
        if eos_id is not None and tok == eos_id:
            break
        if audio_low <= tok < audio_high:
            out.append(tok)
    if not out:
        raise RuntimeError(
            "Model did not generate any audio tokens. "
            "Try increasing --max-new-tokens or enabling --do-sample."
        )
    return torch.tensor(out, dtype=torch.long)


def _unpack_audio_tokens(
    packed_tokens: torch.Tensor,
    audio_token_offset: int,
    num_codebooks: int,
    cardinality: int,
) -> torch.Tensor:
    vals = packed_tokens - audio_token_offset
    usable = (vals.numel() // num_codebooks) * num_codebooks
    if usable == 0:
        raise RuntimeError(
            f"Need at least {num_codebooks} generated audio tokens, got {vals.numel()}."
        )
    vals = vals[:usable]
    frames = usable // num_codebooks
    vals = vals.view(frames, num_codebooks).transpose(0, 1).contiguous()  # [K, T]

    offsets = (torch.arange(num_codebooks).unsqueeze(1) * cardinality).long()
    codes = vals - offsets
    if codes.min().item() < 0 or codes.max().item() >= cardinality:
        raise RuntimeError(
            "Recovered Mimi codes are out of range. "
            "Check mapping/cardinality consistency."
        )
    return codes.unsqueeze(0).long()  # [1, K, T]


def _trim_to_whole_frames(audio_tokens: torch.Tensor, num_codebooks: int) -> torch.Tensor:
    usable = (audio_tokens.numel() // num_codebooks) * num_codebooks
    if usable <= 0:
        raise RuntimeError(
            f"Need at least {num_codebooks} source audio tokens to form one frame, got {audio_tokens.numel()}."
        )
    return audio_tokens[:usable]


class AudioOnlyLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        vocab_size: int,
        audio_low: int,
        audio_high: int,
        eos_id: int | None,
    ):
        super().__init__()
        if not (0 <= audio_low < audio_high <= vocab_size):
            raise RuntimeError(
                f"Invalid audio token range [{audio_low}, {audio_high}) for vocab_size={vocab_size}."
            )
        self.audio_low = audio_low
        self.audio_high = audio_high
        self.eos_id = eos_id
        valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
        valid_mask[audio_low:audio_high] = True
        if eos_id is not None:
            if eos_id < 0 or eos_id >= vocab_size:
                raise RuntimeError(f"Invalid eos_id={eos_id} for vocab_size={vocab_size}.")
            valid_mask[eos_id] = True
        self.valid_mask = valid_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[-1] != self.valid_mask.numel():
            raise RuntimeError(
                f"Logits size mismatch: logits={scores.shape[-1]}, mask={self.valid_mask.numel()}."
            )
        valid_mask = self.valid_mask.to(device=scores.device)
        return scores.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))


def _load_source_from_manifest(manifest: Path, index: int) -> Path:
    with manifest.open("r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i != index:
                continue
            item = json.loads(ln)
            raw = item["src_codes"]
            p = Path(raw)
            if p.is_absolute():
                return p
            return manifest.parent / p
    raise RuntimeError(f"Index {index} out of range in {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample inference for PEFT Mimi S2S model: "
            "English source (audio or Mimi codes) -> generated Luganda audio."
        )
    )
    parser.add_argument("--adapter-dir", type=str, required=True,
                        help="Path to LoRA adapter dir, e.g. .../checkpoints/lora_mimi_s2s/final")
    parser.add_argument("--mapping", type=str, default="",
                        help="Optional path to mimi_token_mapping.json. Defaults to <adapter-dir>/mimi_token_mapping.json")
    parser.add_argument("--base-model", type=str, default="",
                        help="Optional override for base model. Defaults to mapping['base_model'].")
    parser.add_argument("--task-token", type=str, default="",
                        help="Optional task token label override for logs only.")
    parser.add_argument("--src-audio", type=str, default="",
                        help="Path to source English audio file to encode with Mimi.")
    parser.add_argument("--src-codes", type=str, default="",
                        help="Path to source Mimi code .pt file.")
    parser.add_argument("--manifest", type=str, default="",
                        help="Optional manifest JSONL to read source path from.")
    parser.add_argument("--index", type=int, default=0,
                        help="Row index in manifest when --manifest is used.")
    parser.add_argument("--output-wav", type=str, required=True)
    parser.add_argument("--output-codes", type=str, default="",
                        help="Optional output path for generated target Mimi codes (.pt).")
    parser.add_argument("--save-src-codes", type=str, default="",
                        help="Optional output path for encoded source Mimi codes when using --src-audio.")
    parser.add_argument("--mimi", type=str, default="",
                        help="Path to Mimi weights (.safetensors). If empty, downloads from --hf-repo.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--auto-max-new-tokens",
        action="store_true",
        help="Estimate max_new_tokens from source frames using --src-to-tgt-frame-ratio.",
    )
    parser.add_argument(
        "--src-to-tgt-frame-ratio",
        type=float,
        default=1.0,
        help="Expected target/source frame ratio for auto max_new_tokens.",
    )
    parser.add_argument(
        "--tgt-frame-bias",
        type=int,
        default=0,
        help="Extra target frames to add when auto estimating length.",
    )
    parser.add_argument(
        "--max-new-tokens-cap",
        type=int,
        default=0,
        help="Optional hard cap for auto max_new_tokens (0 disables cap).",
    )
    parser.add_argument("--max-src-audio-tokens", type=int, default=0,
                        help="If > 0, truncate packed source-audio tokens to this length.")
    parser.add_argument(
        "--disable-logit-mask",
        action="store_true",
        help="Disable logits masking to audio/EOS token space during generation.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    mapping_path = Path(args.mapping) if args.mapping else adapter_dir / "mimi_token_mapping.json"
    mapping = _load_mapping(mapping_path)

    base_model = args.base_model if args.base_model else mapping["base_model"]
    audio_token_offset = int(mapping["audio_token_offset"])
    num_codebooks = int(mapping["num_codebooks"])
    cardinality = int(mapping["cardinality"])
    task_marker_id = mapping.get("task_marker_id", None)
    task_marker_id = int(task_marker_id) if task_marker_id is not None else None
    src_marker_id = int(mapping["src_marker_id"])
    sep_marker_id = int(mapping["sep_marker_id"])
    tgt_marker_id = int(mapping["tgt_marker_id"])
    task_token_label = args.task_token if args.task_token else mapping.get("task_token", "")
    eos_id = mapping.get("eos_id", None)
    eos_id = int(eos_id) if eos_id is not None else None
    bos_id = mapping.get("bos_id", None)
    bos_id = int(bos_id) if bos_id is not None else None
    pad_id = mapping.get("pad_id", None)
    pad_id = int(pad_id) if pad_id is not None else None

    source_mode_count = int(bool(args.src_audio)) + int(bool(args.src_codes)) + int(bool(args.manifest))
    if source_mode_count != 1:
        raise SystemExit("Provide exactly one of --src-audio, --src-codes, or --manifest.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device)

    if args.mimi:
        mimi_path = Path(args.mimi)
        if not mimi_path.exists():
            raise SystemExit(f"Mimi weights not found: {mimi_path}")
    else:
        mimi_path = Path(hf_hub_download(args.hf_repo, loaders.MIMI_NAME))
    mimi = loaders.get_mimi(mimi_path, device=device, num_codebooks=num_codebooks)
    mimi.eval()

    print(f"Loading tokenizer/model base={base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    expected_vocab = audio_token_offset + (num_codebooks * cardinality)
    current_vocab = model.get_input_embeddings().num_embeddings
    if current_vocab != expected_vocab:
        if current_vocab > expected_vocab:
            raise RuntimeError(
                f"Base model vocab ({current_vocab}) is larger than adapter-expected vocab "
                f"({expected_vocab}). This mapping/base-model pair is inconsistent."
            )
        print(f"Resizing embeddings from {current_vocab} -> {expected_vocab} before loading adapter.")
        model.resize_token_embeddings(expected_vocab)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device)
    model.eval()

    src_desc = ""
    if args.src_audio:
        src_audio_path = Path(args.src_audio)
        if not src_audio_path.exists():
            raise SystemExit(f"Source audio not found: {src_audio_path}")
        wav, sr = _load_audio(src_audio_path)
        wav = _to_mono(wav)
        wav = _resample_if_needed(wav, sr, mimi.sample_rate)
        wav, frames = _pad_to_frame(wav, mimi.frame_size)
        wav_b = wav.unsqueeze(0).to(device=device)
        with torch.no_grad():
            src_codes_batch = mimi.encode(wav_b)
        src_codes = src_codes_batch[0, :num_codebooks, :frames].detach().cpu().long()
        src_desc = str(src_audio_path)

        if args.save_src_codes:
            out_src_codes = Path(args.save_src_codes)
            out_src_codes.parent.mkdir(parents=True, exist_ok=True)
            src_payload = {
                "codes": src_codes.unsqueeze(0).short(),
                "num_codebooks": num_codebooks,
                "cardinality": cardinality,
                "frame_rate": mimi.frame_rate,
                "sample_rate": mimi.sample_rate,
                "source_audio": str(src_audio_path),
            }
            torch.save(src_payload, out_src_codes)
            print(f"Saved source Mimi codes to {out_src_codes}")
    else:
        if args.src_codes:
            src_codes_path = Path(args.src_codes)
        else:
            src_codes_path = _load_source_from_manifest(Path(args.manifest), args.index)
        src_codes = _load_codes(src_codes_path, num_codebooks=num_codebooks, cardinality=cardinality)
        src_desc = str(src_codes_path)

    src_audio_tokens = _pack_audio_codes(src_codes, cardinality=cardinality, audio_token_offset=audio_token_offset)
    if args.max_src_audio_tokens > 0 and src_audio_tokens.numel() > args.max_src_audio_tokens:
        src_audio_tokens = src_audio_tokens[:args.max_src_audio_tokens]
        print(f"Truncated source audio tokens to {src_audio_tokens.numel()} by --max-src-audio-tokens.")
    src_audio_tokens = _trim_to_whole_frames(src_audio_tokens, num_codebooks=num_codebooks)

    max_positions = getattr(model.config, "max_position_embeddings", None)
    fixed_prompt_tokens = (1 if bos_id is not None else 0) + 3 + (1 if task_marker_id is not None else 0)
    # [BOS?] + [TASK?] + src_marker + sep + tgt_marker
    min_new_tokens = num_codebooks  # at least one audio frame
    if isinstance(max_positions, int) and max_positions > 0:
        max_src_for_min = max_positions - fixed_prompt_tokens - min_new_tokens
        if max_src_for_min <= 0:
            raise RuntimeError(
                "Model context is too small for this prompt format "
                f"(model max_position_embeddings={max_positions})."
            )
        if src_audio_tokens.numel() > max_src_for_min:
            src_audio_tokens = src_audio_tokens[:max_src_for_min]
            src_audio_tokens = _trim_to_whole_frames(src_audio_tokens, num_codebooks=num_codebooks)
            print(
                f"Truncated source audio tokens to {src_audio_tokens.numel()} "
                f"to fit model context (max_position_embeddings={max_positions})."
            )

    src_frames_effective = src_audio_tokens.numel() // num_codebooks
    if args.auto_max_new_tokens:
        if args.src_to_tgt_frame_ratio <= 0:
            raise RuntimeError("--src-to-tgt-frame-ratio must be > 0 when using --auto-max-new-tokens.")
        est_tgt_frames = int(round(src_frames_effective * args.src_to_tgt_frame_ratio)) + args.tgt_frame_bias
        est_tgt_frames = max(1, est_tgt_frames)
        max_new_tokens = est_tgt_frames * num_codebooks
        if args.max_new_tokens_cap > 0:
            max_new_tokens = min(max_new_tokens, args.max_new_tokens_cap)
        print(
            f"Auto max_new_tokens from source frames: src_frames={src_frames_effective} "
            f"ratio={args.src_to_tgt_frame_ratio:.3f} bias={args.tgt_frame_bias} "
            f"-> max_new_tokens={max_new_tokens}"
        )
    else:
        if args.max_new_tokens <= 0:
            raise RuntimeError("--max-new-tokens must be > 0 when --auto-max-new-tokens is not set.")
        max_new_tokens = args.max_new_tokens

    if isinstance(max_positions, int) and max_positions > 0:
        remaining_for_new = max_positions - fixed_prompt_tokens - src_audio_tokens.numel()
        if remaining_for_new <= 0:
            raise RuntimeError(
                f"No room left for generation after source prompt. "
                f"max_position_embeddings={max_positions}"
            )
        if max_new_tokens > remaining_for_new:
            print(
                f"Clamping max_new_tokens from {max_new_tokens} to {remaining_for_new} "
                "to fit model context."
            )
            max_new_tokens = remaining_for_new

    prompt = []
    if bos_id is not None:
        prompt.append(bos_id)
    if task_marker_id is not None:
        prompt.append(task_marker_id)
    prompt.append(src_marker_id)
    prompt.extend(src_audio_tokens.tolist())
    prompt.append(sep_marker_id)
    prompt.append(tgt_marker_id)

    input_ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    print(
        f"Generating with prompt_tokens={input_ids.shape[-1]} "
        f"max_new_tokens={max_new_tokens} do_sample={args.do_sample}"
    )
    if task_marker_id is not None:
        if task_token_label:
            print(f"Using task token: {task_token_label} (id={task_marker_id})")
        else:
            print(f"Using task marker id: {task_marker_id}")
    logits_processor = None
    if not args.disable_logit_mask:
        vocab_size = model.get_input_embeddings().num_embeddings
        audio_low = audio_token_offset
        audio_high = audio_token_offset + (num_codebooks * cardinality)
        logits_processor = LogitsProcessorList(
            [AudioOnlyLogitsProcessor(vocab_size, audio_low, audio_high, eos_id)]
        )
        print(
            f"Enabled logits mask: allow audio token range [{audio_low}, {audio_high}) "
            f"and eos_id={eos_id}."
        )

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            eos_token_id=eos_id,
            pad_token_id=(pad_id if pad_id is not None else tokenizer.eos_token_id),
            logits_processor=logits_processor,
        )
    new_tokens = gen[0, input_ids.shape[-1]:].detach().cpu()
    audio_tokens = _extract_audio_tokens(
        new_tokens,
        audio_token_offset=audio_token_offset,
        num_codebooks=num_codebooks,
        cardinality=cardinality,
        eos_id=eos_id,
    )
    tgt_codes = _unpack_audio_tokens(
        audio_tokens,
        audio_token_offset=audio_token_offset,
        num_codebooks=num_codebooks,
        cardinality=cardinality,
    )

    with torch.no_grad():
        wav = mimi.decode(tgt_codes.to(device=device))
    out_wav = wav[0].detach().cpu()

    output_wav = Path(args.output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    _save_audio(output_wav, out_wav, mimi.sample_rate)

    if args.output_codes:
        out_codes = Path(args.output_codes)
        out_codes.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "codes": tgt_codes.cpu().short(),
            "num_codebooks": num_codebooks,
            "cardinality": cardinality,
            "frame_rate": mimi.frame_rate,
            "sample_rate": mimi.sample_rate,
            "source": src_desc,
        }
        torch.save(payload, out_codes)
        print(f"Saved generated codes to {out_codes}")

    print(f"Saved synthesized audio to {output_wav}")


if __name__ == "__main__":
    main()
