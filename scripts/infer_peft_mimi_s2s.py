#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        description="Sample inference for PEFT Mimi S2S model: src Mimi codes -> generated tgt audio."
    )
    parser.add_argument("--adapter-dir", type=str, required=True,
                        help="Path to LoRA adapter dir, e.g. .../checkpoints/lora_mimi_s2s/final")
    parser.add_argument("--mapping", type=str, default="",
                        help="Optional path to mimi_token_mapping.json. Defaults to <adapter-dir>/mimi_token_mapping.json")
    parser.add_argument("--base-model", type=str, default="",
                        help="Optional override for base model. Defaults to mapping['base_model'].")
    parser.add_argument("--src-codes", type=str, default="",
                        help="Path to source Mimi code .pt file.")
    parser.add_argument("--manifest", type=str, default="",
                        help="Optional manifest JSONL to read source path from.")
    parser.add_argument("--index", type=int, default=0,
                        help="Row index in manifest when --manifest is used.")
    parser.add_argument("--output-wav", type=str, required=True)
    parser.add_argument("--output-codes", type=str, default="",
                        help="Optional output path for generated target Mimi codes (.pt).")
    parser.add_argument("--mimi", type=str, default="",
                        help="Path to Mimi weights (.safetensors). If empty, downloads from --hf-repo.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
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
    src_marker_id = int(mapping["src_marker_id"])
    sep_marker_id = int(mapping["sep_marker_id"])
    tgt_marker_id = int(mapping["tgt_marker_id"])
    eos_id = mapping.get("eos_id", None)
    eos_id = int(eos_id) if eos_id is not None else None
    bos_id = mapping.get("bos_id", None)
    bos_id = int(bos_id) if bos_id is not None else None
    pad_id = mapping.get("pad_id", None)
    pad_id = int(pad_id) if pad_id is not None else None

    if args.src_codes:
        src_codes_path = Path(args.src_codes)
    elif args.manifest:
        src_codes_path = _load_source_from_manifest(Path(args.manifest), args.index)
    else:
        raise SystemExit("Provide either --src-codes or --manifest.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device)

    print(f"Loading tokenizer/model base={base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device)
    model.eval()

    src_codes = _load_codes(src_codes_path, num_codebooks=num_codebooks, cardinality=cardinality)
    src_audio_tokens = _pack_audio_codes(src_codes, cardinality=cardinality, audio_token_offset=audio_token_offset)

    prompt = []
    if bos_id is not None:
        prompt.append(bos_id)
    prompt.append(src_marker_id)
    prompt.extend(src_audio_tokens.tolist())
    prompt.append(sep_marker_id)
    prompt.append(tgt_marker_id)

    input_ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    print(
        f"Generating with prompt_tokens={input_ids.shape[-1]} "
        f"max_new_tokens={args.max_new_tokens} do_sample={args.do_sample}"
    )
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            eos_token_id=eos_id,
            pad_token_id=(pad_id if pad_id is not None else tokenizer.eos_token_id),
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

    if args.mimi:
        mimi_path = Path(args.mimi)
        if not mimi_path.exists():
            raise SystemExit(f"Mimi weights not found: {mimi_path}")
    else:
        mimi_path = Path(hf_hub_download(args.hf_repo, loaders.MIMI_NAME))

    mimi = loaders.get_mimi(mimi_path, device=device, num_codebooks=num_codebooks)
    mimi.eval()
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
            "source_src_codes": str(src_codes_path),
        }
        torch.save(payload, out_codes)
        print(f"Saved generated codes to {out_codes}")

    print(f"Saved synthesized audio to {output_wav}")


if __name__ == "__main__":
    main()
