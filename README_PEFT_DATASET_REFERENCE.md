# Dataset + PEFT Script Reference

This document is the code-level reference for the dataset build and PEFT scripts
used in this repo for:

- STS: Luganda speech -> English speech
- STT: Luganda speech -> English text

Use this together with:

- [README_S2S_WORKFLOW.md](/Users/sunbird/Documents/Workshop/s2s/moshi/README_S2S_WORKFLOW.md)
- [README_S2TT_WORKFLOW.md](/Users/sunbird/Documents/Workshop/s2s/moshi/README_S2TT_WORKFLOW.md)

## Script Map

- [scripts/build_mimi_hf_dataset.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/build_mimi_hf_dataset.py)
Purpose: Build the shared Mimi-token dataset format consumed by both STS and STT training.
- [scripts/train_peft_mimi_s2s.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/train_peft_mimi_s2s.py)
Purpose: Train a LoRA adapter for Luganda-speech to English-speech.
- [scripts/infer_peft_mimi_s2s.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/infer_peft_mimi_s2s.py)
Purpose: Run offline STS inference and decode generated Mimi codes back to audio.
- [scripts/train_peft_mimi_s2tt.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/train_peft_mimi_s2tt.py)
Purpose: Train a LoRA adapter for Luganda-speech to English-text.
- [scripts/infer_peft_mimi_s2tt.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/infer_peft_mimi_s2tt.py)
Purpose: Run offline STT inference and decode generated tokens back to text.

## Shared Data Flow

1. `build_mimi_hf_dataset.py` reads paired Luganda/English audio plus aligned text from a Hugging Face dataset.
2. It normalizes the audio, encodes both sides with Mimi, and writes:
   - `dataset.<split>.jsonl`
   - `codes/lug/<split>/*.lug.pt`
   - `codes/eng/<split>/*.eng.pt`
3. `train_peft_mimi_s2s.py` uses `src_codes` and `tgt_codes`.
4. `train_peft_mimi_s2tt.py` uses `src_codes` and `tgt_text`.
5. The inference scripts reuse the saved `mimi_token_mapping.json` to reconstruct the prompt format used during training.

## Dataset Builder Reference

Script: [scripts/build_mimi_hf_dataset.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/build_mimi_hf_dataset.py)

- `_try_import_soundfile`
Role: Lazily import `soundfile` so the script can decode audio when available.
- `_try_import_torchaudio`
Role: Lazily import `torchaudio` for optional decode and resample support.
- `_to_mono`
Role: Convert multi-channel audio to mono before Mimi encoding.
- `_resample_if_needed`
Role: Resample source audio to Mimi's target sample rate.
- `_pad_to_frame`
Role: Pad waveforms so every example has a whole number of Mimi frames.
- `_load_audio`
Role: Decode audio from a file path or raw bytes into a torch tensor.
- `_audio_to_tensor`
Role: Normalize different Hugging Face audio object formats into one tensor layout.
- `_save_codes`
Role: Save Mimi code tensors and their metadata to `.pt` files.
- `_encode_batch`
Role: Batch and encode waveforms with Mimi on CPU or GPU.
- `_sanitize_key`
Role: Build a filesystem-safe sample id for code filenames.
- `_split_list`
Role: Parse comma-separated split names from CLI input.
- `_resolve_device`
Role: Resolve `auto`, `cpu`, or `cuda` into a torch device.
- `_configure_torch_backend`
Role: Turn on CUDA backend optimizations for faster Mimi encoding.
- `_default_repo_id`
Role: Derive the default Hugging Face dataset repo name for the tokenized export.
- `_write_dataset_card`
Role: Write the generated dataset `README.md`.
- `_prepare_item`
Role: Perform per-row CPU preprocessing before batched Mimi encoding.
- `_flush_pending`
Role: Encode the pending batch, write code files, and append manifest rows.
- `main`
Role: Orchestrate the full dataset build and optional Hugging Face upload.

## STS Training Reference

Script: [scripts/train_peft_mimi_s2s.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/train_peft_mimi_s2s.py)

- `_split_csv`
Role: Parse comma-separated manifest and module lists.
- `_resolve_dtype`
Role: Pick the dtype used to load and run the base language model.
- `_configure_torch_backend`
Role: Enable CUDA settings that improve training speed.
- `_set_seed`
Role: Seed Python and torch randomness for reproducibility.
- `_load_jsonl`
Role: Load manifest rows from JSONL files.
- `_load_json`
Role: Load the top-level training config JSON.
- `_resolve_path`
Role: Resolve config-relative paths into absolute paths.
- `_safe_repo_dir_name`
Role: Turn a repo id into a safe local snapshot directory name.
- `_truncate_pair`
Role: Truncate source and target token streams while protecting target coverage.
- `_take_tokens`
Role: Keep head or tail tokens from a sequence based on policy.
- `_truncate_to_frame_boundary`
Role: Ensure packed audio tokens end on a full Mimi frame.
- `_truncate_pair_with_policy`
Role: Apply the user-selected truncation strategy.
- `PackedIds`
Role: Simple container for model `input_ids` and `labels`.
- `MimiS2SDataset`
Role: Convert manifest rows into causal-LM examples for speech-to-speech training.
- `MimiS2SDataset._resolve_path`
Role: Resolve source and target code file paths.
- `MimiS2SDataset._load_codes`
Role: Load Mimi code tensors and validate metadata.
- `MimiS2SDataset._pack_audio_codes`
Role: Flatten per-codebook Mimi indices into one shared audio-token stream.
- `MimiS2SDataset._build_sequence`
Role: Build the actual prompt and supervised target layout used for STS training.
- `MimiS2SDataset.__getitem__`
Role: Read one example from disk and return model-ready tensors.
- `CausalCollator`
Role: Pad variable-length examples into batched tensors.
- `_move_to_device`
Role: Move one collated batch onto CPU or GPU.
- `_build_target_modules`
Role: Decide which transformer modules receive LoRA adapters.
- `evaluate`
Role: Compute validation loss over a dataloader.
- `save_artifacts`
Role: Save the adapter, tokenizer, config, and Mimi token mapping.
- `_apply_config_overrides`
Role: Merge config-file values into CLI args.
- `_resolve_hf_token`
Role: Resolve the Hugging Face token from args or environment.
- `_normalize_repo_entry`
Role: Normalize repo config entries into dict form.
- `_resolve_manifest_paths`
Role: Resolve train and validation manifests inside one dataset snapshot.
- `_resolve_repo_manifests`
Role: Resolve all training/validation manifests from local paths or HF repos.
- `main`
Role: Run the full STS PEFT training lifecycle.

## STS Inference Reference

Script: [scripts/infer_peft_mimi_s2s.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/infer_peft_mimi_s2s.py)

- `_try_import_soundfile`
Role: Lazily import `soundfile` for output and input audio handling.
- `_try_import_torchaudio`
Role: Lazily import `torchaudio` as a fallback audio backend.
- `_save_audio`
Role: Save the decoded English waveform to disk.
- `_load_audio`
Role: Load source Luganda audio from disk.
- `_to_mono`
Role: Convert source audio to mono.
- `_resample_if_needed`
Role: Resample source audio to Mimi's sample rate.
- `_pad_to_frame`
Role: Pad source audio to whole Mimi frames.
- `_resolve_dtype`
Role: Select model inference dtype.
- `_load_mapping`
Role: Load the saved training-time token mapping.
- `_load_codes`
Role: Load source Mimi codes from disk instead of raw audio.
- `_pack_audio_codes`
Role: Convert source Mimi codes into flat prompt tokens.
- `_extract_audio_tokens`
Role: Keep only valid generated audio tokens up to EOS.
- `_unpack_audio_tokens`
Role: Turn flat generated audio tokens back into Mimi code tensors.
- `_trim_to_whole_frames`
Role: Ensure source packed tokens represent whole frames only.
- `AudioOnlyLogitsProcessor`
Role: Constrain generation to audio tokens and EOS.
- `_load_source_from_manifest`
Role: Fetch a source code path from a manifest row by index.
- `main`
Role: Run offline STS inference from prompt construction through waveform save.

## STT Training Reference

Script: [scripts/train_peft_mimi_s2tt.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/train_peft_mimi_s2tt.py)

- `_split_csv`
Role: Parse comma-separated manifest and module lists.
- `_resolve_dtype`
Role: Pick the dtype used to load and run the base language model.
- `_configure_torch_backend`
Role: Enable CUDA settings that improve training speed.
- `_set_seed`
Role: Seed Python and torch randomness for reproducibility.
- `_load_jsonl`
Role: Load manifest rows from JSONL files.
- `_load_json`
Role: Load the top-level training config JSON.
- `_resolve_path`
Role: Resolve config-relative paths into absolute paths.
- `_safe_repo_dir_name`
Role: Turn a repo id into a safe local snapshot directory name.
- `_take_tokens`
Role: Keep head or tail tokens from a sequence based on policy.
- `_truncate_pair`
Role: Truncate source audio and target text while protecting target coverage.
- `_truncate_to_frame_boundary`
Role: Ensure packed source audio ends on a full Mimi frame.
- `_truncate_pair_with_policy`
Role: Apply the user-selected truncation strategy.
- `PackedIds`
Role: Simple container for model `input_ids` and `labels`.
- `MimiS2TTDataset`
Role: Convert manifest rows into causal-LM examples for speech-to-text translation training.
- `MimiS2TTDataset._resolve_path`
Role: Resolve source code file paths.
- `MimiS2TTDataset._load_codes`
Role: Load Mimi source codes and validate metadata.
- `MimiS2TTDataset._pack_audio_codes`
Role: Flatten per-codebook Mimi indices into one shared audio-token stream.
- `MimiS2TTDataset._tokenize_target_text`
Role: Normalize and tokenize English target text.
- `MimiS2TTDataset._build_sequence`
Role: Build the prompt and supervised text target layout used for STT training.
- `MimiS2TTDataset.__getitem__`
Role: Read one example from disk and return model-ready tensors.
- `CausalCollator`
Role: Pad variable-length examples into batched tensors.
- `_move_to_device`
Role: Move one collated batch onto CPU or GPU.
- `_build_target_modules`
Role: Decide which transformer modules receive LoRA adapters.
- `evaluate`
Role: Compute validation loss over a dataloader.
- `save_artifacts`
Role: Save the adapter, tokenizer, config, and Mimi token mapping.
- `_apply_config_overrides`
Role: Merge config-file values into CLI args.
- `_resolve_hf_token`
Role: Resolve the Hugging Face token from args or environment.
- `_normalize_repo_entry`
Role: Normalize repo config entries into dict form.
- `_resolve_manifest_paths`
Role: Resolve train and validation manifests inside one dataset snapshot.
- `_resolve_repo_manifests`
Role: Resolve all training/validation manifests from local paths or HF repos.
- `main`
Role: Run the full STT PEFT training lifecycle.

## STT Inference Reference

Script: [scripts/infer_peft_mimi_s2tt.py](/Users/sunbird/Documents/Workshop/s2s/moshi/scripts/infer_peft_mimi_s2tt.py)

- `_try_import_soundfile`
Role: Lazily import `soundfile` for input audio handling.
- `_try_import_torchaudio`
Role: Lazily import `torchaudio` as a fallback audio backend.
- `_load_audio`
Role: Load source Luganda audio from disk.
- `_to_mono`
Role: Convert source audio to mono.
- `_resample_if_needed`
Role: Resample source audio to Mimi's sample rate.
- `_pad_to_frame`
Role: Pad source audio to whole Mimi frames.
- `_resolve_dtype`
Role: Select model inference dtype.
- `_load_mapping`
Role: Load the saved training-time token mapping.
- `_load_codes`
Role: Load source Mimi codes from disk instead of raw audio.
- `_pack_audio_codes`
Role: Convert source Mimi codes into flat prompt tokens.
- `_trim_to_whole_frames`
Role: Ensure source packed tokens represent whole frames only.
- `_extract_text_tokens`
Role: Keep only valid generated text tokens up to EOS.
- `TextOnlyLogitsProcessor`
Role: Constrain generation to base text-vocabulary tokens and EOS.
- `_load_source_from_manifest`
Role: Fetch a source code path from a manifest row by index.
- `main`
Role: Run offline STT inference from prompt construction through decoded text output.

## Which Script To Use

- If you are preparing data for either task:
Use `build_mimi_hf_dataset.py`.
- If you want audio output at inference time:
Use the STS train and infer scripts.
- If you want text output at inference time:
Use the STT train and infer scripts.
- If your tokenized dataset already exists:
Skip the builder and point the trainers at the local directory or HF dataset repo.
