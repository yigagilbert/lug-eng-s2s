# S2S Mimi + PEFT Workflow

This guide covers the full pipeline used in this repo with Luganda as source and
English as target:

1. Build Mimi-tokenized data from private Hugging Face datasets.
2. Train a PEFT (LoRA) speech-to-speech model.
3. Run inference from Luganda audio to generated English audio.

For a function-by-function script reference, see
[`README_PEFT_DATASET_REFERENCE.md`](/Users/sunbird/Documents/Workshop/s2s/moshi/README_PEFT_DATASET_REFERENCE.md).

## Scripts

1. Data build:
`/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/build_mimi_hf_dataset.py`
2. Training:
`/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/train_peft_mimi_s2s.py`
3. Inference:
`/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/infer_peft_mimi_s2s.py`

## Environment

Install dependencies:

```bash
pip install -r /workspace/lug-eng-s2s/moshi/requirements.txt
```

Set Hugging Face token for private datasets/models:

```bash
export HF_TOKEN=hf_xxx
```

## 1) Build Tokenized Data

Tokenize dataset 1:

```bash
python3 /workspace/lug-eng-s2s/scripts/build_mimi_hf_dataset.py \
  --dataset yigagilbert/salt-s2s-lug-eng-studio-dataset \
  --splits all \
  --device auto \
  --batch-size 8 \
  --num-codebooks 8 \
  --hf-token $HF_TOKEN
```

This now creates and uploads a dataset repo named:

```text
yigagilbert/salt-s2s-lug-eng-studio-dataset_mimi_token_version
```

Tokenize dataset 2 (example placeholder):

```bash
python3 /workspace/lug-eng-s2s/scripts/build_mimi_hf_dataset.py \
  --dataset <your-second-dataset-id> \
  --splits all \
  --device cuda \
  --batch-size 8 \
  --num-codebooks 8 \
  --hf-token $HF_TOKEN
```

Override the generated repo name if needed:

```bash
python3 /workspace/lug-eng-s2s/scripts/build_mimi_hf_dataset.py \
  --dataset <your-dataset-id> \
  --repo-id <namespace>/<custom-repo-name> \
  --splits all \
  --device cuda \
  --batch-size 8 \
  --num-codebooks 8 \
  --hf-token $HF_TOKEN
```

Keep the tokenized dataset only on local disk:

```bash
python3 /workspace/lug-eng-s2s/scripts/build_mimi_hf_dataset.py \
  --dataset <your-dataset-id> \
  --splits all \
  --device auto \
  --batch-size 8 \
  --num-codebooks 8 \
  --output /workspace/lug-eng-s2s/data/local_mimi_tokens/<dataset-name> \
  --skip-upload \
  --hf-token $HF_TOKEN
```

### Tokenized Data Outputs

For each generated Hugging Face dataset repo:

1. Split manifests:
`dataset.train.jsonl`, `dataset.validation.jsonl`, `dataset.test.jsonl`
2. Mimi source codes:
`codes/lug/<split>/*.lug.pt`
3. Mimi target codes:
`codes/eng/<split>/*.eng.pt`
4. Build stats:
`stats.json`
5. Build metadata:
`build_config.json`

## 2) Train LoRA Model

Create a training config that lists all tokenized dataset repos you want to mix.
An example is included at:

`/Users/sunbird/Documents/Workshop/s2s/moshi/configs/mimi_s2s_train.example.json`

Example:

```bash
{
  "hf_token_env": "HF_TOKEN",
  "data": {
    "snapshot_dir": "data/hf_mimi_snapshots",
    "repos": [
      "yigagilbert/salt-s2s-lug-eng-studio-dataset_mimi_token_version",
      "<your-second-dataset-id>_mimi_token_version"
    ]
  },
  "training": {
    "output_dir": "checkpoints/lora_mimi_s2s_llama",
    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
    "task_token": "<TASK_S2ST_LG_EN>",
    "max_seq_len": 4096,
    "context_overflow": "clamp",
    "truncate_policy": "tail",
    "max_steps": 2000,
    "lr": 0.0001
  }
}
```

Then start training with one command:

```bash
python3 /workspace/lug-eng-s2s/scripts/train_peft_mimi_s2s.py \
  --config /workspace/lug-eng-s2s/configs/mimi_s2s_train.example.json
```

You can also train from a local tokenized dataset directory instead of a HF repo:

```bash
{
  "hf_token_env": "HF_TOKEN",
  "data": {
    "repos": [
      {
        "local_path": "data/local_mimi_tokens/your-dataset-name"
      }
    ]
  },
  "training": {
    "output_dir": "checkpoints/lora_mimi_s2s_llama",
    "base_model": "meta-llama/Llama-3.2-1B-Instruct"
  }
}
```

Behavior:

1. The trainer downloads or refreshes each dataset repo in `data.repos`.
2. It automatically picks up `dataset.train.jsonl` and `dataset.validation.jsonl`.
3. It combines manifests across all listed repos and starts training.

If needed, the older direct-manifest CLI still works, but the config flow is now the recommended path.

### Throughput Notes

1. `build_mimi_hf_dataset.py` now supports `--device auto`, which chooses CUDA when available and otherwise falls back to CPU.
2. For tokenization throughput, the primary knob is `--batch-size`; increase it until GPU memory is close to full without OOM.
3. For training throughput, raise `batch_size`, `num_workers`, and use `--persistent-workers` when your dataloader can keep up.
4. Full 100% GPU utilization is not guaranteed if audio decode or dataset IO is the bottleneck, but these settings help keep the GPU fed.

### Training Outputs

1. Periodic checkpoints:
`/workspace/lug-eng-s2s/checkpoints/lora_mimi_s2s_llama/checkpoint-*`
2. Final adapter:
`/workspace/lug-eng-s2s/checkpoints/lora_mimi_s2s_llama/final`
3. Key metadata:
`/workspace/lug-eng-s2s/checkpoints/lora_mimi_s2s_llama/final/mimi_token_mapping.json`

## 3) Inference

Luganda audio to generated English audio:

```bash
python3 /workspace/lug-eng-s2s/scripts/infer_peft_mimi_s2s.py \
  --adapter-dir /workspace/lug-eng-s2s/checkpoints/lora_mimi_s2s_llama/final \
  --src-audio /workspace/lug-eng-s2s/data/sample_lug.wav \
  --output-wav /workspace/lug-eng-s2s/data/infer/sample_lug_to_eng.wav \
  --output-codes /workspace/lug-eng-s2s/data/infer/sample_lug_to_eng.pt \
  --save-src-codes /workspace/lug-eng-s2s/data/infer/sample_lug_src_codes.pt \
  --auto-max-new-tokens \
  --src-to-tgt-frame-ratio 1.05 \
  --tgt-frame-bias 2 \
  --max-new-tokens-cap 1600 \
  --do-sample --temperature 0.8 --top-p 0.95
```

Notes:

1. Logit masking to valid audio/EOS tokens is enabled by default.
2. Keep adapter and `mimi_token_mapping.json` from the same run.
3. You can disable masking with `--disable-logit-mask` for debugging only.

## Common Issues

1. `size mismatch ... embed_tokens ...` during inference:
Use the updated inference script that resizes embeddings before loading adapter.
2. `torchvision::nms does not exist`:
Uninstall mismatched torchvision for this text/audio pipeline:
`python3 -m pip uninstall -y torchvision`
3. Private HF access errors:
Ensure `HF_TOKEN` is set and has dataset/model access.
