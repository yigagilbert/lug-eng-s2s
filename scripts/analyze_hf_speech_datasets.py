#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Audit one or more Hugging Face speech datasets before Mimi tokenization.

This utility is meant to answer practical dataset questions early:

- Are these mostly sentence-level pairs or long utterances?
- How many hours look usable versus partially silent?
- Do the source and target fields appear complete and aligned?
- Should we trim silence, re-segment long clips, or filter bad rows first?

The script can inspect multiple dataset repos, aggregate split-level summaries,
and emit both JSON and Markdown reports.
"""

import argparse
import io
import json
import math
import os
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, load_dataset


def _split_csv(arg: str) -> list[str]:
    """Split a comma-separated CLI value into trimmed entries."""
    return [x.strip() for x in arg.split(",") if x.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON config file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected top-level JSON object in config: {path}")
    return data


def _try_import_soundfile():
    """Import `soundfile` lazily so we can fall back to torchaudio when absent."""
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        return None
    return sf


def _try_import_torchaudio():
    """Import `torchaudio` lazily for optional decode support."""
    try:
        import torchaudio  # type: ignore
    except Exception:
        return None
    return torchaudio


def _load_audio(path: str | None = None, raw_bytes: bytes | None = None) -> tuple[torch.Tensor, int]:
    """Decode audio from a file path or raw bytes into a `[channels, time]` tensor."""
    sf = _try_import_soundfile()
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
        return torch.from_numpy(wav).t().contiguous(), int(sr)

    if path:
        if sf is not None:
            wav, sr = sf.read(path, always_2d=True)
            return torch.from_numpy(wav).t().contiguous(), int(sr)
        ta = _try_import_torchaudio()
        if ta is None:
            raise RuntimeError(
                "No audio backend found. Install either `soundfile` or `torchaudio`."
            )
        wav, sr = ta.load(path)
        return wav.contiguous(), int(sr)

    raise RuntimeError("Audio object had neither path nor bytes.")


def _audio_to_tensor(audio_obj: Any) -> tuple[torch.Tensor, int]:
    """Normalize Hugging Face audio objects into a float tensor and sample rate."""
    arr = None
    sr = None

    if isinstance(audio_obj, dict):
        arr = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate")
        if arr is None or sr is None:
            path = audio_obj.get("path")
            raw_bytes = audio_obj.get("bytes")
            wav, out_sr = _load_audio(path=path, raw_bytes=raw_bytes)
            return wav.to(dtype=torch.float32).contiguous(), out_sr
    elif hasattr(audio_obj, "get_all_samples"):
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
        if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
            wav = arr.t()
        else:
            wav = arr
    else:
        raise RuntimeError(f"Unsupported audio array shape: {tuple(arr.shape)}")
    return wav.to(dtype=torch.float32).contiguous(), sr


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    """Collapse multi-channel audio to mono using a simple channel mean."""
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _normalize_text(text: Any) -> str:
    """Convert text to a normalized single-line string."""
    return " ".join(str(text or "").strip().split())


def _word_count(text: str) -> int:
    """Count whitespace-delimited words after normalization."""
    if not text:
        return 0
    return len(text.split())


def _quantile(values: list[float], q: float) -> float | None:
    """Compute a linear-interpolated quantile for a non-empty numeric list."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    mix = pos - lo
    return ordered[lo] * (1.0 - mix) + ordered[hi] * mix


def _summarize_numeric(values: list[float]) -> dict[str, float] | None:
    """Summarize a numeric list with mean and several quantiles."""
    if not values:
        return None
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "p05": float(_quantile(values, 0.05) or 0.0),
        "p10": float(_quantile(values, 0.10) or 0.0),
        "p50": float(_quantile(values, 0.50) or 0.0),
        "p90": float(_quantile(values, 0.90) or 0.0),
        "p95": float(_quantile(values, 0.95) or 0.0),
        "p99": float(_quantile(values, 0.99) or 0.0),
        "max": float(max(values)),
    }


def _duration_buckets(durations: list[float]) -> dict[str, dict[str, float]]:
    """Bucket clip durations into ranges useful for segmentation decisions."""
    if not durations:
        return {}
    buckets = {
        "lt_2s": 0,
        "2_to_5s": 0,
        "5_to_10s": 0,
        "10_to_30s": 0,
        "30_to_60s": 0,
        "gt_60s": 0,
    }
    for d in durations:
        if d < 2.0:
            buckets["lt_2s"] += 1
        elif d < 5.0:
            buckets["2_to_5s"] += 1
        elif d < 10.0:
            buckets["5_to_10s"] += 1
        elif d < 30.0:
            buckets["10_to_30s"] += 1
        elif d < 60.0:
            buckets["30_to_60s"] += 1
        else:
            buckets["gt_60s"] += 1
    total = float(len(durations))
    return {
        key: {
            "count": float(count),
            "fraction": float(count / total),
        }
        for key, count in buckets.items()
    }


def _estimate_activity(
    wav: torch.Tensor,
    sr: int,
    window_ms: int,
    min_active_dbfs: float,
    relative_margin_db: float,
) -> dict[str, float | bool]:
    """Estimate speech activity and silence using simple windowed RMS heuristics."""
    mono = _to_mono(wav).squeeze(0).to(dtype=torch.float32)
    duration_sec = float(mono.numel() / max(1, sr))
    if mono.numel() == 0:
        return {
            "duration_sec": 0.0,
            "active_ratio": 0.0,
            "active_sec": 0.0,
            "leading_silence_sec": 0.0,
            "trailing_silence_sec": 0.0,
            "peak_dbfs": -120.0,
            "noise_floor_dbfs": -120.0,
            "silence_sec": 0.0,
            "is_silent": True,
            "is_clipped": False,
        }

    window_size = max(1, int(round(sr * float(window_ms) / 1000.0)))
    peak = float(mono.abs().max().item())
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-8))
    if mono.numel() <= window_size:
        rms = torch.sqrt(torch.mean(mono.square()).clamp_min(1e-12)).unsqueeze(0)
    else:
        usable = (mono.numel() // window_size) * window_size
        trimmed = mono[:usable] if usable > 0 else mono
        if trimmed.numel() == 0:
            trimmed = mono
        windows = trimmed.view(-1, window_size)
        rms = torch.sqrt(windows.square().mean(dim=1).clamp_min(1e-12))
    window_db = (20.0 * torch.log10(rms.clamp_min(1e-8))).tolist()
    noise_floor = float(_quantile(window_db, 0.10) or -120.0)
    ref_db = float(_quantile(window_db, 0.95) or peak_dbfs)
    activity_threshold = max(min_active_dbfs, ref_db - relative_margin_db)
    active_mask = [db >= activity_threshold for db in window_db]

    if any(active_mask):
        first_active = active_mask.index(True)
        last_active = len(active_mask) - 1 - active_mask[::-1].index(True)
        active_count = sum(1 for x in active_mask if x)
        leading_silence_sec = (first_active * window_size) / float(sr)
        trailing_silence_sec = ((len(active_mask) - 1 - last_active) * window_size) / float(sr)
    else:
        active_count = 0
        leading_silence_sec = duration_sec
        trailing_silence_sec = duration_sec

    active_sec = (active_count * window_size) / float(sr)
    active_sec = min(duration_sec, active_sec)
    silence_sec = max(0.0, duration_sec - active_sec)
    active_ratio = 0.0 if duration_sec <= 0 else min(1.0, active_sec / duration_sec)
    is_silent = peak_dbfs < min_active_dbfs or active_count == 0
    is_clipped = peak >= 0.999
    return {
        "duration_sec": duration_sec,
        "active_ratio": active_ratio,
        "active_sec": active_sec,
        "leading_silence_sec": leading_silence_sec,
        "trailing_silence_sec": trailing_silence_sec,
        "peak_dbfs": peak_dbfs,
        "noise_floor_dbfs": noise_floor,
        "silence_sec": silence_sec,
        "is_silent": is_silent,
        "is_clipped": is_clipped,
    }


def _format_hours(seconds: float) -> float:
    """Convert seconds to hours for report readability."""
    return float(seconds / 3600.0)


def _push_top_k(heap: list[tuple[float, dict[str, Any]]], score: float, item: dict[str, Any], k: int) -> None:
    """Keep the top-k items by score in a min-heap."""
    entry = (score, item)
    if len(heap) < k:
        heapq.heappush(heap, entry)
        return
    if score > heap[0][0]:
        heapq.heapreplace(heap, entry)


@dataclass
class TextAccumulator:
    """Collect text completeness and length statistics."""

    values: list[int] = field(default_factory=list)
    chars: list[int] = field(default_factory=list)
    empties: int = 0
    duplicates: int = 0
    seen: set[str] = field(default_factory=set)
    words_per_second: list[float] = field(default_factory=list)

    def add(self, text: str, duration_sec: float | None = None) -> None:
        """Accumulate text lengths, duplicate counts, and speaking-rate proxies."""
        if not text:
            self.empties += 1
            return
        words = _word_count(text)
        self.values.append(words)
        self.chars.append(len(text))
        if text in self.seen:
            self.duplicates += 1
        else:
            self.seen.add(text)
        if duration_sec is not None and duration_sec > 0:
            self.words_per_second.append(words / duration_sec)

    def finalize(self, total_rows: int) -> dict[str, Any] | None:
        """Return summary metrics for the accumulated text column."""
        if total_rows <= 0:
            return None
        return {
            "non_empty_rows": len(self.values),
            "empty_rows": self.empties,
            "empty_rate": float(self.empties / total_rows),
            "duplicate_non_empty_rows": self.duplicates,
            "duplicate_non_empty_rate": (
                float(self.duplicates / max(1, len(self.values)))
            ),
            "words": _summarize_numeric([float(x) for x in self.values]),
            "chars": _summarize_numeric([float(x) for x in self.chars]),
            "words_per_second": _summarize_numeric(self.words_per_second),
        }


@dataclass
class AudioAccumulator:
    """Collect audio duration and activity/silence metrics."""

    durations: list[float] = field(default_factory=list)
    active_ratios: list[float] = field(default_factory=list)
    active_seconds: list[float] = field(default_factory=list)
    leading_silence: list[float] = field(default_factory=list)
    trailing_silence: list[float] = field(default_factory=list)
    peak_dbfs: list[float] = field(default_factory=list)
    noise_floor_dbfs: list[float] = field(default_factory=list)
    silent_count: int = 0
    clipped_count: int = 0

    def add(self, stats: dict[str, float | bool]) -> None:
        """Accumulate metrics computed for one decoded audio sample."""
        self.durations.append(float(stats["duration_sec"]))
        self.active_ratios.append(float(stats["active_ratio"]))
        self.active_seconds.append(float(stats["active_sec"]))
        self.leading_silence.append(float(stats["leading_silence_sec"]))
        self.trailing_silence.append(float(stats["trailing_silence_sec"]))
        self.peak_dbfs.append(float(stats["peak_dbfs"]))
        self.noise_floor_dbfs.append(float(stats["noise_floor_dbfs"]))
        if bool(stats["is_silent"]):
            self.silent_count += 1
        if bool(stats["is_clipped"]):
            self.clipped_count += 1

    def finalize(self) -> dict[str, Any] | None:
        """Return summary metrics for the accumulated audio column."""
        if not self.durations:
            return None
        total_duration = float(sum(self.durations))
        total_active = float(sum(self.active_seconds))
        return {
            "rows": len(self.durations),
            "total_hours": _format_hours(total_duration),
            "estimated_active_hours": _format_hours(total_active),
            "estimated_silence_hours": _format_hours(max(0.0, total_duration - total_active)),
            "speech_coverage_ratio": 0.0 if total_duration <= 0 else float(total_active / total_duration),
            "silent_clip_rate": float(self.silent_count / len(self.durations)),
            "clipped_clip_rate": float(self.clipped_count / len(self.durations)),
            "duration_sec": _summarize_numeric(self.durations),
            "duration_buckets": _duration_buckets(self.durations),
            "activity_ratio": _summarize_numeric(self.active_ratios),
            "leading_silence_sec": _summarize_numeric(self.leading_silence),
            "trailing_silence_sec": _summarize_numeric(self.trailing_silence),
            "peak_dbfs": _summarize_numeric(self.peak_dbfs),
            "noise_floor_dbfs": _summarize_numeric(self.noise_floor_dbfs),
        }


@dataclass
class SplitAccumulator:
    """Collect all split-level dataset diagnostics."""

    top_k_examples: int
    source_audio: AudioAccumulator = field(default_factory=AudioAccumulator)
    target_audio: AudioAccumulator = field(default_factory=AudioAccumulator)
    source_text: TextAccumulator = field(default_factory=TextAccumulator)
    target_text: TextAccumulator = field(default_factory=TextAccumulator)
    src_to_tgt_duration_ratio: list[float] = field(default_factory=list)
    decode_failures: int = 0
    rows: int = 0
    longest_examples: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    quietest_examples: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    def add_example(
        self,
        row_id: str,
        src_duration_sec: float,
        src_active_ratio: float,
        tgt_duration_sec: float | None,
        src_text_words: int,
        tgt_text_words: int,
    ) -> None:
        """Track representative examples for later inspection."""
        item = {
            "id": row_id,
            "src_duration_sec": src_duration_sec,
            "src_active_ratio": src_active_ratio,
            "tgt_duration_sec": tgt_duration_sec,
            "src_text_words": src_text_words,
            "tgt_text_words": tgt_text_words,
        }
        _push_top_k(self.longest_examples, src_duration_sec, item, self.top_k_examples)
        _push_top_k(self.quietest_examples, 1.0 - src_active_ratio, item, self.top_k_examples)


def _structure_guess(duration_stats: dict[str, Any] | None, buckets: dict[str, Any]) -> str:
    """Heuristically describe whether a split looks sentence-level or long-form."""
    if not duration_stats:
        return "unknown"
    p50 = float(duration_stats["p50"])
    p95 = float(duration_stats["p95"])
    gt_60 = float(buckets.get("gt_60s", {}).get("fraction", 0.0))
    five_to_thirty = float(buckets.get("5_to_10s", {}).get("fraction", 0.0)) + float(
        buckets.get("10_to_30s", {}).get("fraction", 0.0)
    )
    if p50 <= 15.0 and p95 <= 30.0 and five_to_thirty >= 0.50:
        return "mostly_sentence_level_short_clips"
    if p50 <= 20.0 and p95 <= 60.0 and gt_60 <= 0.10:
        return "mixed_short_utterances"
    if p50 <= 30.0 and p95 > 60.0:
        return "short_utterances_with_long_tail"
    if p50 > 30.0 or p95 > 90.0:
        return "long_utterances_or_long_form_segments"
    return "mixed_structure"


def _cleaning_recommendations(split_report: dict[str, Any]) -> list[str]:
    """Turn split-level metrics into concrete cleaning suggestions."""
    recs: list[str] = []
    src_audio = split_report.get("source_audio")
    tgt_audio = split_report.get("target_audio")
    src_text = split_report.get("source_text")
    tgt_text = split_report.get("target_text")
    pairing = split_report.get("pairing")
    structure = split_report.get("structure_guess", "unknown")

    if not src_audio:
        return ["No source audio was analyzed; verify the source audio column name."]

    src_duration = src_audio.get("duration_sec") or {}
    src_lead = src_audio.get("leading_silence_sec") or {}
    src_trail = src_audio.get("trailing_silence_sec") or {}
    src_activity = src_audio.get("activity_ratio") or {}
    if structure in {"short_utterances_with_long_tail", "long_utterances_or_long_form_segments"}:
        recs.append(
            "The split contains substantial long-form audio. Consider re-segmenting into sentence-level or "
            "short utterance pairs before Mimi tokenization."
        )
    if float(src_duration.get("p95", 0.0)) > 45.0:
        recs.append(
            "The source p95 duration is above 45 seconds. Long clips may hurt packing efficiency and should be "
            "reviewed for segmentation."
        )
    if float(src_audio.get("speech_coverage_ratio", 1.0)) < 0.75:
        recs.append(
            "Estimated speech coverage is below 75%. Trim silence or run VAD to avoid spending token budget on "
            "non-speech."
        )
    if float(src_lead.get("p50", 0.0)) > 0.25 or float(src_trail.get("p50", 0.0)) > 0.25:
        recs.append(
            "Median leading or trailing silence is above 250 ms. Trimming clip boundaries would likely improve "
            "usable hours."
        )
    if float(src_audio.get("silent_clip_rate", 0.0)) > 0.01:
        recs.append(
            "More than 1% of clips look silent or near-silent. Inspect those rows before building Mimi tokens."
        )
    if float(src_audio.get("clipped_clip_rate", 0.0)) > 0.01:
        recs.append(
            "More than 1% of clips appear clipped. Review recording quality and consider filtering damaged audio."
        )
    if src_text and float(src_text.get("empty_rate", 0.0)) > 0.01:
        recs.append("Source text has missing values above 1%. Verify transcript completeness.")
    if tgt_text and float(tgt_text.get("empty_rate", 0.0)) > 0.01:
        recs.append("Target text has missing values above 1%. Verify translation completeness.")
    if tgt_text:
        wps = (tgt_text.get("words_per_second") or {})
        if wps and (float(wps.get("p50", 2.0)) < 0.6 or float(wps.get("p95", 2.0)) > 6.5):
            recs.append(
                "Target text speaking-rate estimates look unusual for many rows. Spot-check alignment or text quality."
            )
    if tgt_audio and pairing:
        ratio = pairing.get("tgt_to_src_duration_ratio")
        if ratio and (
            float(ratio.get("p95", 1.0)) > 2.5 or float(ratio.get("p05", 1.0)) < 0.4
        ):
            recs.append(
                "Source/target duration ratios have a wide tail. This may indicate inconsistent segmentation or "
                "alignment across language pairs."
            )
    if int(split_report.get("decode_failures", 0)) > 0:
        recs.append("Some rows failed to decode. Review those files before trusting the total hours figure.")
    if not recs:
        recs.append(
            "The split looks broadly usable for Mimi tokenization. Spot-check the longest and lowest-activity clips "
            "before launching a full build."
        )
    return recs


def _finalize_split(acc: SplitAccumulator) -> dict[str, Any]:
    """Finalize one split accumulator into a JSON-serializable report."""
    source_audio = acc.source_audio.finalize()
    target_audio = acc.target_audio.finalize()
    structure = "unknown"
    if source_audio is not None:
        structure = _structure_guess(
            source_audio.get("duration_sec"),
            source_audio.get("duration_buckets", {}),
        )
    pairing = None
    if acc.src_to_tgt_duration_ratio:
        pairing = {
            "tgt_to_src_duration_ratio": _summarize_numeric(acc.src_to_tgt_duration_ratio),
        }

    longest_examples = [
        item for _, item in sorted(acc.longest_examples, key=lambda x: x[0], reverse=True)
    ]
    quietest_examples = [
        item for _, item in sorted(acc.quietest_examples, key=lambda x: x[0], reverse=True)
    ]

    report = {
        "rows": acc.rows,
        "decode_failures": acc.decode_failures,
        "structure_guess": structure,
        "source_audio": source_audio,
        "target_audio": target_audio if target_audio and target_audio.get("rows", 0) > 0 else None,
        "source_text": acc.source_text.finalize(acc.rows),
        "target_text": acc.target_text.finalize(acc.rows),
        "pairing": pairing,
        "examples": {
            "longest_source_clips": longest_examples,
            "lowest_activity_source_clips": quietest_examples,
        },
    }
    report["cleaning_recommendations"] = _cleaning_recommendations(report)
    return report


def _coalesce_split_list(value: Any) -> list[str] | str:
    """Normalize a split config field into either a list or the sentinel `all`."""
    if value is None:
        return "all"
    if isinstance(value, str):
        if value.strip() == "all":
            return "all"
        return _split_csv(value)
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    raise RuntimeError("Dataset `splits` must be a string, list, or null.")


def _normalize_dataset_entry(entry: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    """Normalize one dataset config entry using global CLI defaults."""
    if isinstance(entry, str):
        raw = entry.strip()
        if not raw:
            raise RuntimeError("Empty dataset entry.")
        if "::" in raw:
            repo_id, config_name = raw.split("::", 1)
            return {
                "repo_id": repo_id.strip(),
                "config": config_name.strip() or None,
                **defaults,
            }
        return {"repo_id": raw, "config": None, **defaults}
    if isinstance(entry, dict):
        merged = dict(defaults)
        merged.update(entry)
        if not merged.get("repo_id"):
            raise RuntimeError("Each dataset entry must define `repo_id`.")
        return merged
    raise RuntimeError("Dataset entries must be strings or objects.")


def _resolve_dataset_entries(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve dataset entries from config JSON or repeated CLI arguments."""
    default_entry = {
        "splits": _coalesce_split_list(args.splits),
        "id_col": args.id_col,
        "src_audio_col": args.src_audio_col,
        "tgt_audio_col": (args.tgt_audio_col or None),
        "src_text_col": (args.src_text_col or None),
        "tgt_text_col": (args.tgt_text_col or None),
        "max_samples_per_split": args.max_samples_per_split,
    }

    config_entries = config.get("datasets", [])
    if config_entries:
        if not isinstance(config_entries, list):
            raise RuntimeError("Config field `datasets` must be a list.")
        return [_normalize_dataset_entry(entry, default_entry) for entry in config_entries]

    if not args.dataset:
        raise SystemExit(
            "Provide at least one --dataset entry or a JSON config with a `datasets` list."
        )
    return [_normalize_dataset_entry(entry, default_entry) for entry in args.dataset]


def _resolve_hf_token(args: argparse.Namespace, config: dict[str, Any]) -> str | None:
    """Resolve the Hugging Face token from CLI first, then from config env var."""
    if args.hf_token:
        return args.hf_token
    token_env = str(config.get("hf_token_env", "HF_TOKEN"))
    token = os.environ.get(token_env)
    return token if token else None


def _iter_split_names(
    repo_id: str,
    config_name: str | None,
    splits: list[str] | str,
    token: str | None,
) -> list[str]:
    """Resolve the list of splits to analyze for one dataset repo."""
    if splits == "all":
        ds_dict = load_dataset(repo_id, config_name, token=token)
        return list(ds_dict.keys())
    return list(splits)


def _render_bucket_line(buckets: dict[str, Any]) -> str:
    """Render duration buckets in one compact Markdown line."""
    order = ["lt_2s", "2_to_5s", "5_to_10s", "10_to_30s", "30_to_60s", "gt_60s"]
    parts = []
    for key in order:
        entry = buckets.get(key)
        if not entry:
            continue
        parts.append(f"`{key}` {entry['fraction'] * 100.0:.1f}%")
    return ", ".join(parts)


def _fmt(value: float | None, digits: int = 2) -> str:
    """Format an optional float consistently for Markdown output."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the JSON report into a concise Markdown summary."""
    lines = [
        "# Dataset Audit Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Datasets analyzed: `{len(report['datasets'])}`",
        "",
    ]
    for dataset in report["datasets"]:
        title = dataset["repo_id"]
        if dataset.get("config"):
            title += f"::{dataset['config']}"
        lines.extend([f"## {title}", ""])
        for split_name, split_report in dataset["splits"].items():
            lines.extend([f"### Split `{split_name}`", ""])
            lines.append(f"- Rows analyzed: `{split_report['rows']}`")
            lines.append(f"- Decode failures: `{split_report['decode_failures']}`")
            lines.append(f"- Structure guess: `{split_report['structure_guess']}`")

            src_audio = split_report.get("source_audio")
            if src_audio:
                dur = src_audio.get("duration_sec") or {}
                act = src_audio.get("activity_ratio") or {}
                lead = src_audio.get("leading_silence_sec") or {}
                trail = src_audio.get("trailing_silence_sec") or {}
                lines.append(
                    f"- Source audio hours: total `{_fmt(src_audio.get('total_hours'))}`, "
                    f"active `{_fmt(src_audio.get('estimated_active_hours'))}`, "
                    f"coverage `{_fmt(src_audio.get('speech_coverage_ratio', 0.0) * 100.0)}%`"
                )
                lines.append(
                    f"- Source duration sec: median `{_fmt(dur.get('p50'))}`, "
                    f"p95 `{_fmt(dur.get('p95'))}`, max `{_fmt(dur.get('max'))}`"
                )
                lines.append(
                    f"- Source duration buckets: {_render_bucket_line(src_audio.get('duration_buckets', {}))}"
                )
                lines.append(
                    f"- Source activity ratio: median `{_fmt(act.get('p50'))}`, "
                    f"mean `{_fmt(act.get('mean'))}`, silent clip rate "
                    f"`{_fmt(src_audio.get('silent_clip_rate', 0.0) * 100.0)}%`"
                )
                lines.append(
                    f"- Source boundary silence sec: leading median `{_fmt(lead.get('p50'))}`, "
                    f"trailing median `{_fmt(trail.get('p50'))}`"
                )

            tgt_audio = split_report.get("target_audio")
            if tgt_audio:
                dur = tgt_audio.get("duration_sec") or {}
                lines.append(
                    f"- Target audio hours: total `{_fmt(tgt_audio.get('total_hours'))}`, "
                    f"active `{_fmt(tgt_audio.get('estimated_active_hours'))}`"
                )
                lines.append(
                    f"- Target duration sec: median `{_fmt(dur.get('p50'))}`, "
                    f"p95 `{_fmt(dur.get('p95'))}`, max `{_fmt(dur.get('max'))}`"
                )

            src_text = split_report.get("source_text")
            if src_text:
                src_words = src_text.get("words") or {}
                lines.append(
                    f"- Source text: empty `{_fmt(src_text.get('empty_rate', 0.0) * 100.0)}%`, "
                    f"median words `{_fmt(src_words.get('p50'))}`"
                )

            tgt_text = split_report.get("target_text")
            if tgt_text:
                tgt_words = tgt_text.get("words") or {}
                lines.append(
                    f"- Target text: empty `{_fmt(tgt_text.get('empty_rate', 0.0) * 100.0)}%`, "
                    f"median words `{_fmt(tgt_words.get('p50'))}`"
                )

            pairing = split_report.get("pairing")
            if pairing:
                ratio = pairing.get("tgt_to_src_duration_ratio") or {}
                lines.append(
                    f"- Target/source duration ratio: median `{_fmt(ratio.get('p50'))}`, "
                    f"p95 `{_fmt(ratio.get('p95'))}`"
                )

            lines.append("- Cleaning recommendations:")
            for rec in split_report.get("cleaning_recommendations", []):
                lines.append(f"  - {rec}")

            longest = split_report.get("examples", {}).get("longest_source_clips", [])
            if longest:
                lines.append("- Longest source clips:")
                for item in longest[:3]:
                    lines.append(
                        f"  - `{item['id']}` src `{_fmt(item['src_duration_sec'])}` sec, "
                        f"activity `{_fmt(item['src_active_ratio'] * 100.0)}`%"
                    )
            quietest = split_report.get("examples", {}).get("lowest_activity_source_clips", [])
            if quietest:
                lines.append("- Lowest-activity source clips:")
                for item in quietest[:3]:
                    lines.append(
                        f"  - `{item['id']}` src `{_fmt(item['src_duration_sec'])}` sec, "
                        f"activity `{_fmt(item['src_active_ratio'] * 100.0)}`%"
                    )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _analyze_dataset_split(
    dataset_entry: dict[str, Any],
    split_name: str,
    token: str | None,
    cache_dir: str | None,
    window_ms: int,
    min_active_dbfs: float,
    relative_margin_db: float,
    top_k_examples: int,
) -> dict[str, Any]:
    """Analyze one dataset split and return aggregated diagnostics."""
    repo_id = str(dataset_entry["repo_id"])
    config_name = dataset_entry.get("config")
    src_audio_col = dataset_entry.get("src_audio_col")
    tgt_audio_col = dataset_entry.get("tgt_audio_col")
    src_text_col = dataset_entry.get("src_text_col")
    tgt_text_col = dataset_entry.get("tgt_text_col")
    id_col = str(dataset_entry.get("id_col", "id"))
    max_samples = int(dataset_entry.get("max_samples_per_split", 0) or 0)

    ds = load_dataset(
        repo_id,
        config_name,
        split=split_name,
        token=token,
        cache_dir=cache_dir,
    )
    if src_audio_col:
        ds = ds.cast_column(str(src_audio_col), Audio(decode=False))
    if tgt_audio_col:
        ds = ds.cast_column(str(tgt_audio_col), Audio(decode=False))

    acc = SplitAccumulator(top_k_examples=top_k_examples)
    for idx, row in enumerate(ds):
        if max_samples > 0 and idx >= max_samples:
            break
        row_id = str(row.get(id_col, f"{split_name}-{idx:09d}"))
        acc.rows += 1
        try:
            src_wav, src_sr = _audio_to_tensor(row[str(src_audio_col)])
            src_stats = _estimate_activity(
                src_wav,
                src_sr,
                window_ms=window_ms,
                min_active_dbfs=min_active_dbfs,
                relative_margin_db=relative_margin_db,
            )
            acc.source_audio.add(src_stats)

            tgt_duration = None
            if tgt_audio_col:
                tgt_wav, tgt_sr = _audio_to_tensor(row[str(tgt_audio_col)])
                tgt_stats = _estimate_activity(
                    tgt_wav,
                    tgt_sr,
                    window_ms=window_ms,
                    min_active_dbfs=min_active_dbfs,
                    relative_margin_db=relative_margin_db,
                )
                acc.target_audio.add(tgt_stats)
                tgt_duration = float(tgt_stats["duration_sec"])
                src_duration = float(src_stats["duration_sec"])
                if src_duration > 0:
                    acc.src_to_tgt_duration_ratio.append(tgt_duration / src_duration)
        except Exception:
            acc.decode_failures += 1
            continue

        src_text = _normalize_text(row.get(str(src_text_col), "")) if src_text_col else ""
        tgt_text = _normalize_text(row.get(str(tgt_text_col), "")) if tgt_text_col else ""
        src_duration = float(src_stats["duration_sec"])
        acc.source_text.add(src_text, duration_sec=src_duration if src_text else None)
        target_duration_for_text = tgt_duration if tgt_duration is not None else src_duration
        acc.target_text.add(
            tgt_text,
            duration_sec=target_duration_for_text if tgt_text else None,
        )
        acc.add_example(
            row_id=row_id,
            src_duration_sec=src_duration,
            src_active_ratio=float(src_stats["active_ratio"]),
            tgt_duration_sec=tgt_duration,
            src_text_words=_word_count(src_text),
            tgt_text_words=_word_count(tgt_text),
        )
    return _finalize_split(acc)


def main() -> None:
    """Run the multi-repo dataset audit workflow and emit JSON/Markdown reports."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one or more Hugging Face speech datasets for duration structure, "
            "silence/activity, text coverage, and likely pre-tokenization cleaning needs."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON config with a `datasets` list.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset repo id to analyze. Repeat for multiple repos. "
            "You can also write `repo_id::config_name`."
        ),
    )
    parser.add_argument("--hf-token", type=str, default="", help="Optional HF token.")
    parser.add_argument("--cache-dir", type=str, default="", help="Optional Hugging Face cache dir.")
    parser.add_argument("--splits", type=str, default="all", help="Comma-separated splits, or `all`.")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--src-audio-col", type=str, default="audio_lug")
    parser.add_argument("--tgt-audio-col", type=str, default="audio_eng")
    parser.add_argument("--src-text-col", type=str, default="text_lug")
    parser.add_argument("--tgt-text-col", type=str, default="text_eng")
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="If > 0, cap analysis to this many rows per split.",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=30,
        help="Window size in milliseconds for the activity estimator.",
    )
    parser.add_argument(
        "--min-active-dbfs",
        type=float,
        default=-45.0,
        help="Absolute floor in dBFS below which audio is treated as silence.",
    )
    parser.add_argument(
        "--relative-margin-db",
        type=float,
        default=25.0,
        help="Relative margin from high-energy windows used to estimate activity.",
    )
    parser.add_argument(
        "--top-k-examples",
        type=int,
        default=5,
        help="How many longest and lowest-activity examples to keep per split.",
    )
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save JSON output.")
    parser.add_argument("--output-md", type=str, default="", help="Optional path to save Markdown output.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = _load_json(config_path) if config_path is not None else {}
    token = _resolve_hf_token(args, config)
    dataset_entries = _resolve_dataset_entries(args, config)
    cache_dir = args.cache_dir if args.cache_dir else None

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
    }

    for entry in dataset_entries:
        repo_id = str(entry["repo_id"])
        config_name = entry.get("config")
        splits = _coalesce_split_list(entry.get("splits"))
        split_names = _iter_split_names(repo_id, config_name, splits, token)
        dataset_report = {
            "repo_id": repo_id,
            "config": config_name,
            "splits": {},
        }
        for split_name in split_names:
            print(f"Analyzing {repo_id} split={split_name}...")
            split_report = _analyze_dataset_split(
                dataset_entry=entry,
                split_name=split_name,
                token=token,
                cache_dir=cache_dir,
                window_ms=args.window_ms,
                min_active_dbfs=args.min_active_dbfs,
                relative_margin_db=args.relative_margin_db,
                top_k_examples=args.top_k_examples,
            )
            dataset_report["splits"][split_name] = split_report
        report["datasets"].append(dataset_report)

    markdown = _render_markdown(report)
    print(markdown, end="")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        print(f"Saved Markdown report to {output_md}")


if __name__ == "__main__":
    main()
