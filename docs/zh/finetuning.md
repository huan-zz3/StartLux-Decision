> 译自 [../finetuning.md](../finetuning.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# 微调 v1.1 Instant 模型

`finetune/finetune_lora.py` 用 LoRA 把 StartLux-Decision **v1.1 Instant** 目录适配到你自己的带类型的决策（typed decision）上。
它渲染每个问题的方式与服务（serving）端完全一致，只在文本解码器上训练 LoRA 权重，并写出一个合并后的模型目录
（merged），供同一批引擎（engine）服务。

这个脚本面向 Instant（提示（prompt）双向）目录，包括 `StartLux-Decision-v1.1-0.8B-Instant`。首个
发布版（v1）在 [`v1/finetune/finetune_lora.py`](../../v1/finetune/finetune_lora.py) 里有自己的因果训练器；两套
脚本树是有意分开的。

## 环境

- Linux，一块带 CUDA 的 NVIDIA GPU。0.8B 模型的 bf16 权重约占 1.6 GB；16 GB 显存
  足够跑默认配置，24 GB 更宽裕。bf16 训练需要 Ampere 或更新的架构。
- Python 3.10 到 3.12。
- `transformers` **5.8.x**（双向提示注意力是针对该版本写的）、`peft >= 0.20`，以及在 CUDA 上的
  `flash-linear-attention` + `causal-conv1d`。`finetune` extra 还会安装 Pillow 和 torchvision，用于
  图片输入。
- 约 10 GB 磁盘，用于源目录、合并后的副本、适配器（adapter）和日志。

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

经过测试的组合是 torch 2.11.0+cu128、transformers 5.8.1、peft 0.21.2、flash-linear-attention 0.5.2、
causal-conv1d 1.7.0、torchvision 0.26.0+cu128 和 Pillow 12.3.0，硬件为 A40；v1 发布版在 H200 上使用了同一代 torch 与
transformers。
`pip install -e '.[finetune]'` 不会替换已安装且兼容的
torch；如果 PyTorch 不存在，它会安装最新的 PyPI 构建，而这未必匹配你的驱动。内核很关键：
缺少 flash-linear-attention 和 causal-conv1d 时，transformers 会静默回退到纯 PyTorch 路径，慢
十倍以上。训练前先检查模型目录的快路径：

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

它应当打印 `True`。训练器还要求文本解码器的注意力实现为 `sdpa`；transformers 5.8
默认使用它，如果模型以别的方式加载，`bidirectional.enable` 会带着实际值中止。

下载模型：

```bash
mkdir -p models
hf download startlux-models/StartLux-Decision-v1.1-0.8B-Instant \
  --local-dir models/StartLux-Decision-v1.1-0.8B-Instant
```

## 数据

每行一个 JSON 对象，形状与请求相同，再加上你想要的答案：

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

target 的键跟随问题类型：`choice` 用 criteria 的键，是/否问题（noul）用 `"true"` / `"false"`，量表题（score）用
`"0"` 到 `"n-1"`，最低档在前。单个标签就是独热（one-hot）target；标注者的投票占比（vote share）可以按原样
给出。没有 target 的问题会被跳过，选项超过 26 个的 `choice` 列表同样跳过。留出 10% 到 20%
的数据作为验证集（dev）。

### 图片

一条记录可以按与请求相同的形式携带图片：记录级的 `images` 列表，元素为 base64 字符串或 data URI，
以及 state 中的 `<image>` 标记，用来说明每张图片放在哪里。没有标记时图片排在最前面，与
`images.place_images` 一致。只接受 base64/data URI；不接受本地文件路径。

```json
{"state": "Photo taken at delivery: <image>",
 "images": ["data:image/png;base64,iVBORw0KGgo..."],
 "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}},
 "targets": {"damaged": {"true": 1.0}}}
```

每张图片会在服务（serving）路径的 `--min-pixels`（65,536）与 `--max-pixels`（1,048,576）边界内缩放，
每 32 x 32 像素占一个 token。训练时用 PIL 对每张图片重采样一次，随后处理器看到的就是该
目标尺寸，所以像素值不必与服务端逐位相同，但尺寸边界与 token 数必须一致。有效提示（prompt）长度
包含这些图片 token；提示超过 `--max-length` 的问题会被跳过。一条记录最多可含 16 张图片，记录中
所有问题读取同一批图片。

模型目录必须带视觉塔（vision tower），否则脚本会退出。视觉塔保持冻结（frozen），LoRA 仍只
施加在文本解码器上。含图片的记录逐条运行（批大小为 1）；纯文本记录保留按长度排序的批处理。
如果大批图片数据集放不进内存，先调低 `--max-pixels`。

## 训练

```bash
python finetune/finetune_lora.py \
  --model models/StartLux-Decision-v1.1-0.8B-Instant \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out runs/0.8b-lora \
  --calibrate
```

默认值：秩（rank）16、alpha 32、dropout 0.05、学习率 1e-4 配合 5% 预热（warm-up）与余弦衰减、2 个轮次（epoch）、
每微批次（micro-batch）最多 16,384 个填充 token 且每次优化器步进 4 个微批次、提示最多 8,192 token、
图片在 `--min-pixels 65536` 与 `--max-pixels 1048576` 内缩放。脚本会在训练前、每个轮次之后
以及结束时打印验证集（dev）loss 与准确率。如果显存不足，先调低 `--max-tokens`（例如 8,192）、
`--max-length` 或 `--max-pixels`。

只有文本解码器线性层上的 LoRA 适配器（adapter）会被训练。脚本会保存：

```
runs/0.8b-lora/adapter/    # the LoRA adapter
runs/0.8b-lora/merged/     # the merged full model, ready to serve
```

`--no-merge` 只保存适配器（adapter）。合并后的模型目录（merged）会保留源目录的 `config.json`、
`decision_config.json` 及其他 JSON 配置，因此 Instant 的提示（prompt）注意力设置也随之一同带过去。

## 校准

`--calibrate` 使用最终的验证集（dev）读出（read-out），为每种问题类型（choice、是/否问题（noul）、量表题（score））各拟合一个
温度；带图片的验证记录也包含在内。温度从不会改变哪个选项获胜；它只会让
概率更尖锐或更平坦。验证问题少于 200 道的类型使用池化后的温度。拟合出的值会写入
`runs/0.8b-lora/merged/decision_config.json`。

如果不传 `--calibrate`，温度会从源模型原样复制。校准需要
`--dev` 和合并后的模型（merged），因此不能与 `--no-merge` 组合使用。

和 v1 一样，要盯两件事：如果模型在几乎每道验证问题上都答对，拟合会把温度推到
其取值范围的底端，概率在新数据上会变得非常尖锐；请改用更大或更难的验证集。在某个源上
拟合出的温度，在另一个源上可能过度纠正或纠正不足，所以如果有第二份留出（held-out）集，就在它上面检查一下。

## 服务与评估

transformers 后端（backend）会按 Instant 目录训练时所用的双向注意力来读取它们：

```bash
startlux-decide serve \
  --model runs/0.8b-lora/merged \
  --backend transformers \
  --port 8090
```

生产环境请使用 vLLM 路径；`serve --launch-vllm` 会为 Instant 目录设置提示（prompt）双向插件：

```bash
startlux-decide serve \
  --model runs/0.8b-lora/merged \
  --launch-vllm --gpus 0 \
  --port 8090
```

服务端默认每次请求接收 8 张图片；当记录携带更多图片时传入 `--max-images 16`。

不要通过因果（causal）路径评估微调后的目录。v1 的校准脚本和 v1 推理引擎以因果方式读取；
对 Instant 模型来说它们是错误的参照。用同一个服务端和同一份请求数据比较原始目录与微调后的目录：

```bash
startlux-decide eval --url http://127.0.0.1:8090 \
  --data data/dev_requests.jsonl --out preds/finetuned.jsonl
```

如果其他决策变差了，改用更少的轮次（epoch）或更低的学习率。训练损失只看到你标注过的
问题；它对你没有训练过的那些决策一无所知。
