# Fine-tuning a v1.1 Instant model

`finetune/finetune_lora.py` adapts a StartLux-Decision **v1.1 Instant** folder to your own typed decisions with LoRA.
It renders every question exactly as the server does, trains only LoRA weights on the text decoder, and writes a merged
folder that the same engines serve.

This script is for Instant (prompt-bidirectional) folders, including `StartLux-Decision-v1.1-0.8B-Instant`. The first
release (v1) has its own causal trainer in [`v1/finetune/finetune_lora.py`](../v1/finetune/finetune_lora.py); the two
script trees are intentionally separate.

## Environment

- Linux, one NVIDIA GPU with CUDA. A 0.8B model needs about 1.6 GB for the bf16 weights; 16 GB of GPU memory is
  enough for the defaults, 24 GB is comfortable. bf16 training needs Ampere or later.
- Python 3.10 to 3.12.
- `transformers` **5.8.x** (the bidirectional prompt attention is written against that release), `peft >= 0.20`, and
  `flash-linear-attention` + `causal-conv1d` on CUDA. The `finetune` extra also installs Pillow and torchvision for
  image inputs.
- About 10 GB of disk for the source folder, the merged copy, the adapter and logs.

```bash
git clone https://github.com/StartLuxLabs/StartLux-Decision.git
cd StartLux-Decision
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install the PyTorch wheel that matches your driver first. The default PyPI wheel can be built for a newer CUDA
# than the driver supports, in which case torch.cuda.is_available() is False. For a CUDA 12.x driver, for example:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[finetune]'
# causal-conv1d builds against the CUDA toolkit and PyTorch; if pip starts compiling it:
# pip install causal-conv1d --no-build-isolation
```

The tested stack is torch 2.11.0+cu128, transformers 5.8.1, peft 0.21.2, flash-linear-attention 0.5.2,
causal-conv1d 1.7.0, torchvision 0.26.0+cu128 and Pillow 12.3.0 on an A40; the v1 release used the same torch and
transformers generation on an H200.
`pip install -e '.[finetune]'` will not replace an already installed compatible
torch; if PyTorch is absent it installs the latest PyPI build, which may not match your driver. The kernels matter:
without flash-linear-attention and causal-conv1d, transformers silently falls back to a plain PyTorch path that is
more than ten times slower. Check the model folder's fast path before training:

```bash
python - <<'PY'
import importlib
from transformers import AutoConfig

path = "models/StartLux-Decision-v1.1-0.8B-Instant"
cfg = AutoConfig.from_pretrained(path)
modeling = importlib.import_module(f"transformers.models.{cfg.model_type}.modeling_{cfg.model_type}")
print("fast kernels:", getattr(modeling, "is_fast_path_available", False))
PY
```

It should print `True`. The trainer also requires the text decoder's attention implementation to be `sdpa`; transformers 5.8
uses it by default, and `bidirectional.enable` stops with the actual value if the model was loaded differently.

Download the model:

```bash
mkdir -p models
hf download startlux-models/StartLux-Decision-v1.1-0.8B-Instant \
  --local-dir models/StartLux-Decision-v1.1-0.8B-Instant
```

## Data

The normative contract, written for the people who prepare the data, is
[zh/finetune_data.md](zh/finetune_data.md) (Chinese only, by design); verify a file with
`python3 finetune/check_data.py data/train.jsonl` before training. The summary below is the short version.

One JSON object per line, in the same shape as a request plus the answers you want:

```json
{"state": {"ticket": "Refund still missing after two weeks, order 5521", "customer_tier": "pro"},
 "questions": {
   "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                         "technical": "App bugs and outages"}},
   "urgent": {"type": "noul", "instructions": "Does the customer ask for this to be solved today?"},
   "severity": {"type": "score", "instructions": "How badly is the customer affected?",
                "criteria": ["cosmetic", "annoying", "blocks the customer"]}
 },
 "targets": {"team": {"billing": 1.0}, "urgent": {"false": 1.0}, "severity": {"1": 0.7, "2": 0.3}}}
```

Target keys follow the question type: the criteria keys for `choice`, `"true"` / `"false"` for yes/no, and `"0"` to
`"n-1"` for a score, lowest level first. A single label is a one-hot target; annotator vote shares can be given as
they are. Questions without a target are skipped, and so are `choice` lists with more than 26 options. Keep 10% to 20%
of the data aside as a dev set.

### Images

A record may carry images in the same form as a request: a record-level `images` list of base64 strings or data URIs,
and `<image>` marks in the state that say where each image goes. Without marks the images come first, as in
`images.place_images`. Only base64/data URIs are accepted; local file paths are not.

```json
{"state": "Photo taken at delivery: <image>",
 "images": ["data:image/png;base64,iVBORw0KGgo..."],
 "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}},
 "targets": {"damaged": {"true": 1.0}}}
```

Each image is resized within the serving path's `--min-pixels` (65,536) and `--max-pixels` (1,048,576) bounds and
takes one token per 32 x 32 pixels. Training resamples each image once with PIL and the processor then sees that
target size, so pixel values need not be bit-identical to serving, but the size bounds and token counts are. The
effective prompt length includes those image tokens; a question whose prompt is over `--max-length` is skipped. A
record may hold up to 16 images, and all questions of the record read the same images.

The model folder must have a vision tower (the script exits otherwise). The tower stays frozen and LoRA is still
applied to the text decoder only. Records with images run one at a time (batch size 1); text-only records keep the
length-sorted batching. If a large image dataset does not fit in memory, lower `--max-pixels` first.

## Train

```bash
python finetune/finetune_lora.py \
  --model models/StartLux-Decision-v1.1-0.8B-Instant \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out runs/0.8b-lora \
  --calibrate
```

Defaults: rank 16, alpha 32, dropout 0.05, learning rate 1e-4 with 5% warm-up and cosine decay, 2 epochs,
micro-batches of up to 16,384 padded tokens and 4 micro-batches per optimizer step, prompts up to 8,192 tokens,
images resized within `--min-pixels 65536` and `--max-pixels 1048576`. The script prints dev loss and accuracy before
training, after each epoch and at the end. If you run out of memory, lower `--max-tokens` (for example 8,192),
`--max-length`, or `--max-pixels` first.

Only LoRA adapters on the linear layers of the text decoder are trained. The script saves:

```
runs/0.8b-lora/adapter/    # the LoRA adapter
runs/0.8b-lora/merged/     # the merged full model, ready to serve
```

`--no-merge` saves only the adapter. A merged folder keeps the source folder's `config.json`,
`decision_config.json` and other JSON configs, so the Instant prompt-attention setting travels with it.

## Calibrate

`--calibrate` uses the final dev read-outs and fits one temperature per question type (choice, yes/no, score); dev
records with images are included. A temperature never changes which option wins; it only sharpens or flattens the
probabilities. Types with fewer than 200 dev questions get the pooled temperature. The fitted values are written to
`runs/0.8b-lora/merged/decision_config.json`.

If you do not pass `--calibrate`, the temperatures are copied unchanged from the source model. Calibration needs
`--dev` and a merged model, so it cannot be combined with `--no-merge`.

Watch two things, as with v1: if the model is right on nearly every dev question, the fit pushes the temperature to
the bottom of its range and the probabilities become very sharp on new data; use a larger or harder dev set. A
temperature fitted on one source can over- or under-correct on another, so check it on a second held-out set if you
have one.

## Serve and evaluate

The transformers backend reads Instant folders with the bidirectional attention they were trained with:

```bash
startlux-decide serve \
  --model runs/0.8b-lora/merged \
  --backend transformers \
  --port 8090
```

For production, use the vLLM path; `serve --launch-vllm` sets the prompt-bidirectional plugin for Instant folders:

```bash
startlux-decide serve \
  --model runs/0.8b-lora/merged \
  --launch-vllm --gpus 0 \
  --port 8090
```

The server takes 8 images per request by default; pass `--max-images 16` when your records carry more.

Do not evaluate the fine-tuned folder through a causal path. The v1 calibrate script and the v1 inference engine read
causally; they are the wrong reference for an Instant model. Compare the original and fine-tuned folders with the same
server and the same request data:

```bash
startlux-decide eval --url http://127.0.0.1:8090 \
  --data data/dev_requests.jsonl --out preds/finetuned.jsonl
```

If other decisions got worse, use fewer epochs or a lower learning rate. The training loss only sees your labelled
questions; it says nothing about the decisions you did not train on.
