# Dataset Audit Utility

Use [`scripts/analyze_hf_speech_datasets.py`](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/analyze_hf_speech_datasets.py)
to inspect one or more Hugging Face speech datasets before building Mimi tokens.

It helps answer questions like:

- Are the rows mostly sentence-level pairs or longer utterances?
- How many hours look usable once silence is discounted?
- Do source and target texts look complete?
- Should we trim silence, segment long clips, or inspect alignment before tokenization?

## Example Config

See
[`configs/dataset_audit.example.json`](/Users/sunbird/Documents/Workshop/s2s/moshi/configs/dataset_audit.example.json).

For your two private Luganda-English repos on an H100, use
[`configs/dataset_audit.h100.private.json`](/Users/sunbird/Documents/Workshop/s2s/moshi/configs/dataset_audit.h100.private.json).

## Run With A Config File

```bash
python3 /Users/sunbird/Documents/Workshop/s2s/moshi/scripts/analyze_hf_speech_datasets.py \
  --config /Users/sunbird/Documents/Workshop/s2s/moshi/configs/dataset_audit.example.json \
  --output-json /Users/sunbird/Documents/Workshop/s2s/moshi/artifacts/dataset_audit/report.json \
  --output-md /Users/sunbird/Documents/Workshop/s2s/moshi/artifacts/dataset_audit/report.md
```

## H100-Tuned Run

```bash
export HF_TOKEN=hf_your_private_token
export HF_HOME=/tmp/hf
mkdir -p /tmp/hf /Users/sunbird/Documents/Workshop/s2s/moshi/artifacts/dataset_audit

python3 /Users/sunbird/Documents/Workshop/s2s/moshi/scripts/analyze_hf_speech_datasets.py \
  --config /Users/sunbird/Documents/Workshop/s2s/moshi/configs/dataset_audit.h100.private.json \
  --device cuda \
  --cache-dir /tmp/hf \
  --batch-size 256 \
  --preprocess-workers 32 \
  --prefetch-batches 16 \
  --window-ms 30 \
  --min-active-dbfs -45 \
  --relative-margin-db 25 \
  --top-k-examples 10 \
  --output-json /Users/sunbird/Documents/Workshop/s2s/moshi/artifacts/dataset_audit/h100_private_report.json \
  --output-md /Users/sunbird/Documents/Workshop/s2s/moshi/artifacts/dataset_audit/h100_private_report.md
```

If you hit GPU memory pressure on unusually long clips, reduce `--batch-size` to `128` or `64`.
If GPU utilization is low, increase `--preprocess-workers` first, then `--prefetch-batches`.

## Run Against Multiple Repos Directly

```bash
python3 /Users/sunbird/Documents/Workshop/s2s/moshi/scripts/analyze_hf_speech_datasets.py \
  --dataset your-namespace/repo-one \
  --dataset your-namespace/repo-two::default \
  --src-audio-col audio_lug \
  --tgt-audio-col audio_eng \
  --src-text-col text_lug \
  --tgt-text-col text_eng \
  --splits train,validation
```

## What The Report Includes

- Duration totals and percentiles
- Duration buckets such as `<2s`, `5-30s`, and `>60s`
- Estimated active speech coverage versus silence
- Leading and trailing silence estimates
- Silent and clipped clip rates
- Empty text rates and median text lengths
- Source/target duration ratio summaries
- Cleaning recommendations and example rows to inspect

## Heuristic Notes

- The activity estimate uses windowed RMS energy, so it is a practical screening tool rather than a strict VAD.
- The `structure_guess` is heuristic and mainly intended to flag whether segmentation is probably needed before Mimi tokenization.
- If you have very large datasets, use `max_samples_per_split` first for a quick pass, then rerun on the full data once the columns and splits look right.

## Throughput Notes

- GPU analysis only accelerates the batched activity estimation stage. Audio decoding can still be the bottleneck, so `--preprocess-workers` and a fast `--cache-dir` matter a lot.
- On H100, start with `--batch-size 256`, `--preprocess-workers 32`, and `--prefetch-batches 16`.
- If your source files are very long, reduce `--batch-size` before changing the activity parameters.
