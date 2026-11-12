# Changelog

## 1.1.0 (unreleased)

First public release of the engine, for the StartLux-Decision v1.1 models (Instant and AutoThink) and the earlier v1
models.

- Decision server with the `/v1/systemone` request format (also served as `/v1/decisions`), typed questions
  (`choice`, `noul`, `score`) and a probability for every option.
- AutoThink path: confidence gate, thought generation with a loop check, second read-out and mixture, with request
  controls `think_mode`, `gate`, `think_budget` and `mix`.
- `allow_none`: an opt-in "None of the above" option for choice questions, with `none_probability` in the answer.
- Default gate 0.6, so that more requests are answered in one pass. The published numbers used 0.9 (`--gate 0.9`).
- Engines: vLLM 0.19.1 and 0.31 (with a plugin for prompt-bidirectional Instant models), transformers, llama.cpp
  (GGUF, with patches for Instant models) and MLX (mlx-vlm).
- The `vllm` extra installs vLLM 0.31.0, which fixes the security advisories against 0.19.1. The engine still runs
  on 0.19.1, the version of the published numbers.
- Images in the evidence and prompts up to 262,144 tokens.
- Production serving: API keys, TLS, early 429 responses, an access log and an optional request log.
- Tools: `launch-vllm`, `bench`, `eval` and `quantize` (FP8, GGUF and MLX builds).
- Choice options given as a list, or under `options`, are knocked out like `{option: description}`. Before, a list of
  more than 26 options, or of 26 when the extra "None of the above" option was added (`none_check` is on by default),
  was refused with a 422.
- `finetune/finetune_lora.py`: a target key that is not an option id now raises with the file, line, question and key
  instead of being read as zero, so a typo cannot train on the wrong label. The training log also counts the two skip
  causes separately (over 26 options, too long).
- `finetune/check_data.py`: a single-file data checker for the people who prepare fine-tuning data (standard library
  + Pillow; no repository, no model folder, no training dependency). It reports blocking and observation findings with
  stable rule ids and file:line, renders the prompt the model will read, writes the normalized record set, checks
  train/dev overlap, converts the dev split to eval data, and emits a handoff report; `--tokenizer <model folder>` adds
  exact prompt lengths during acceptance. The contract it enforces is `docs/zh/finetune_data.md` (Chinese, for the data
  provider).
- `finetune/check_data.py` also reports advisory image-quality observations (`W05`): actual transparency, non-RGB pixel
  modes, EXIF orientation, JPEG quality estimated from the quantization tables below 90, data-URI MIME that disagrees
  with the decoded format, and originals below `--min-pixels`. They never change the exit code. `data/decision_sample/`
  ships fixtures with expected rule ids for the image rules (`out/image_cases.jsonl` and
  `out/image_quality_cases.jsonl`, each with an `_expected.json`).
