# S2TT Mimi + PEFT Workflow

This guide covers the speech-to-text translation pipeline added in this repo,
with Luganda as source and English as target:

1. Build or reuse Mimi-tokenized Luganda-source datasets.
2. Train a PEFT (LoRA) model on Luganda audio input and English text output.
3. Run inference from Luganda audio to generated English text.

For a function-by-function script reference, see
[`README_PEFT_DATASET_REFERENCE.md`](/Users/sunbird/Documents/Workshop/s2s/moshi/README_PEFT_DATASET_REFERENCE.md).

## Reused Data Format

The existing dataset builder already writes everything this pipeline needs:

- `src_codes`: Mimi tokens for the Luganda speech input
- `src_text`: source transcript
- `tgt_text`: English text target

That means you can reuse the same tokenized dataset repo or local directory produced by
[`scripts/build_mimi_hf_dataset.py`](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/build_mimi_hf_dataset.py)
without generating a new format just for text targets.

## Scripts

1. Training:
[`scripts/train_peft_mimi_s2tt.py`](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/train_peft_mimi_s2tt.py)
2. Inference:
[`scripts/infer_peft_mimi_s2tt.py`](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/infer_peft_mimi_s2tt.py)
3. Example config:
[`configs/mimi_s2tt_train.example.json`](/Users/sunbird/Documents/Workshop/s2s/moshi/configs/mimi_s2tt_train.example.json)

## 1) Train LoRA Model

Example config:

```json
{
  "hf_token_env": "HF_TOKEN",
  "data": {
    "snapshot_dir": "data/hf_mimi_snapshots",
    "repos": [
      "yigagilbert/salt-s2s-lug-eng-studio-dataset_mimi_token_version"
    ]
  },
  "training": {
    "output_dir": "checkpoints/lora_mimi_s2tt_llama",
    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
    "task_token": "<TASK_S2TT_LG_EN>",
    "target_text_key": "tgt_text",
    "max_seq_len": 4096,
    "truncate_policy": "src_first"
  }
}
```

Run training:

```bash
python3 /Users/sunbird/Documents/Workshop/s2s/moshi/scripts/train_peft_mimi_s2tt.py \
  --config /Users/sunbird/Documents/Workshop/s2s/moshi/configs/mimi_s2tt_train.example.json
```

Notes:

1. The trainer keeps the same Luganda Mimi audio token prefix format as the S2S trainer.
2. Only the target side changes: it now supervises the model on `tgt_text` tokens plus EOS.
3. `truncate_policy=src_first` is a good default for this task because it protects more source audio context before trimming the English text target.

## 2) Inference

Generate English text directly from Luganda audio:

```bash
python3 /Users/sunbird/Documents/Workshop/s2s/moshi/scripts/infer_peft_mimi_s2tt.py \
  --adapter-dir /Users/sunbird/Documents/Workshop/s2s/moshi/checkpoints/lora_mimi_s2tt_llama/final \
  --src-audio /Users/sunbird/Documents/Workshop/s2s/moshi/data/sample_lug.wav \
  --output-text /Users/sunbird/Documents/Workshop/s2s/moshi/data/infer/sample_lug_to_eng.txt \
  --max-new-tokens 128
```

Behavior:

1. The script encodes the input waveform with Mimi.
2. It builds the same prompt layout used during training.
3. Generation is masked to the base text vocabulary so the model stays in text-token space instead of drifting into added audio-token ids.

## Outputs

Training writes:

- `checkpoints/lora_mimi_s2tt_llama/checkpoint-*`
- `checkpoints/lora_mimi_s2tt_llama/final`
- `checkpoints/lora_mimi_s2tt_llama/final/mimi_token_mapping.json`

Inference writes:

- printed English text in the console
- optional text file via `--output-text`

## Practical Notes

1. This is a simple offline training pipeline, not a streaming decoder.
2. It is best suited to paired Luganda-speech and English-text examples like the SALT dataset already present in this repo.
3. If you want later, we can extend the same path with evaluation metrics like BLEU, chrF, or SacreBLEU over the validation manifest.
