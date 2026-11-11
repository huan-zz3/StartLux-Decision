# 微调上手指南：StartLux-Decision v1.1 0.8B Instant

这份文档按实际操作的顺序走一遍：先证明管线能跑，再用你自己的数据做出可对比的数字。参考细节（每个字段、每个默认参数的含义）在 [finetuning.md](finetuning.md)，这里不重复，只给顺序、判据和失败时的下一步。

结论先说：**在本仓库里跑通一次微调不需要额外配置**，模型、依赖、GPU 都已经就位；真正决定结果的是数据，第 4 节是本文最长的一节。

## 0. 你会得到什么，花多少时间

| 阶段 | 做什么 | 在这台机器上的实测耗时 |
|---|---|---|
| 自检 | 确认 CUDA、快路径核、模型目录 | 1 分钟 |
| 冒烟 | 20 条样例数据训练 1 轮，产出合并模型 | 训练 74 秒（单张 A40） |
| 基线 | 起服务，用样例测出改前数字 | 约 1 分钟 |
| 造数据 | 标注你自己的问题 | 取决于数据量，见第 4 节 |
| 训练 | 真实数据训练 | 与数据量成正比 |
| 对比 | 同一份数据测改前改后 | 约 1 分钟 |

产出是 `runs/<名字>/merged/` 这样一个目录，里面是完整的模型，`startlux-decide serve --model <该目录>` 直接就能服务。

## 1. 自检：三条命令

**先挑一张空闲的卡。** 默认会用 `cuda:0`，而它常常被别的进程占着。训练前先看：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

选 `memory.used` 小的那张，后面所有命令都加前缀。本文示例用 5 号卡：

```bash
export CUDA_VISIBLE_DEVICES=5
```

**确认 torch 真的能用 GPU。** 驱动显示的 CUDA 版本（例如 12.2）和 torch 编进去的版本（例如 12.8）可以不同，能跑就行，要看的是这一行输出：

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望 `True NVIDIA A40`。若是 `False`，说明装的是和驱动不匹配的 torch wheel，照 [finetuning.md](finetuning.md) 的“环境”一节重装。

**确认快路径核生效。** 没有 `flash-linear-attention` 和 `causal-conv1d` 时，transformers 会静默退到慢十倍的普通实现，不报错：

```bash
.venv/bin/python - <<'PY'
import importlib
from transformers import AutoConfig
path = "models/StartLux-Decision-v1.1-0.8B-Instant"
cfg = AutoConfig.from_pretrained(path)
m = importlib.import_module(f"transformers.models.{cfg.model_type}.modeling_{cfg.model_type}")
print("fast kernels:", getattr(m, "is_fast_path_available", False))
PY
```

必须打印 `fast kernels: True`。本仓库的 `.venv` 已经装好完整组合（torch 2.11.0+cu128、transformers 5.8.1、peft 0.21.2、flash-linear-attention 0.5.2、causal-conv1d 1.7.0），模型也已经在 `models/StartLux-Decision-v1.1-0.8B-Instant`。

## 2. 冒烟：用 20 条样例跑通

仓库带了一份 20 行的样例数据 `examples/finetune/sample.jsonl`，模拟电商客服工单分派，覆盖三种题型和一条图片记录。它的用途只有一个：**证明整条管线通**。

```bash
.venv/bin/python finetune/finetune_lora.py \
  --model models/StartLux-Decision-v1.1-0.8B-Instant \
  --train examples/finetune/sample.jsonl \
  --dev examples/finetune/sample.jsonl \
  --out /tmp/opencode/run-smoke \
  --epochs 1 \
  --calibrate
```

注意 `--dev` 和 `--train` 是同一个文件。这在冒烟时是有意为之，因为此时任何数字都没有意义；真实训练时绝不可以这样，见第 4 节。

实测输出（A40，74 秒）：

```
41 training questions (0 skipped: too long or over 26 options), 41 dev questions
trainable params: 10,822,656 || all params: 763,215,680 || trainable%: 1.4180
{"optimizer_steps": 1, "micro_batches_per_epoch": 2, "before": {"dev_loss": 0.5011, "dev_accuracy": 0.8537}}
{"epoch": 1, "dev_loss": 0.5011, "dev_accuracy": 0.8537}
{"step": 1, "loss": 0.0307, "lr": 0.0, "epoch": 1, "elapsed_s": 73}
{"done": {"optimizer_steps": 1, "elapsed_s": 74, "dev_loss": 0.3432, "dev_accuracy": 0.9268}}
adapter written to /tmp/opencode/run-smoke/adapter
merged model written to /tmp/opencode/run-smoke/merged - serve it with: startlux-decide serve --model ...
```

**跑通的判据**（四条全中才算通）：

1. 第一行显示训练题数大于 0。如果这里就显示 `0 training questions`，往后都没意义，直接看第 7 节。
2. 出现 `trainable%: 1.4180` 这一行。这是 LoRA 挂上了文本解码器的证据。
3. 出现 `{"done": ...}` 且 `elapsed_s` 是实数。
4. `runs/<名字>/merged` 目录存在，体积与源模型目录接近。

最后一段是校准输出，每个题型给出 `temperature` 和校准前后的 ECE / NLL。样例每个题型都不足 200 题，所以全部落到 `pooled_temperature`。

## 3. 基线：先测改前，再测改后

**为什么不能跳过基线。** 微调的目的是“在我的问题上更准”，没有改前数字，就只能凭感觉说“好像好了一点”。而且基线必须用同一份数据、同一个引擎、同一套服务参数。

第一步，把带 `targets` 的训练数据转成 `eval` 要的 `gold` 格式：

```bash
.venv/bin/python finetune/to_eval_data.py examples/finetune/sample.jsonl -o /tmp/opencode/sample_eval.jsonl
# wrote 20 requests (0 questions skipped, 0 records dropped) to /tmp/opencode/sample_eval.jsonl
```

转换规则只有一条值得记住：每题取 `targets` 里概率最高的选项。并列最高会直接报错，不会替你挑一个；`score` 题输出整数等级，因为评测脚本不接受字符串数字；无 `targets` 的题会被跳过并计入统计。

第二步，起原模型的服务，测基线：

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/startlux-decide serve \
  --model models/StartLux-Decision-v1.1-0.8B-Instant \
  --backend transformers --port 18091 &

.venv/bin/startlux-decide eval --url http://127.0.0.1:18091 \
  --data /tmp/opencode/sample_eval.jsonl --out /tmp/opencode/preds-base.jsonl
```

实测（样例数据，20 请求 41 个问题）：

```json
{"requests": 20, "failed_requests": 0, "questions": 41,
 "accuracy": 85.37, "ece": 12.00, "brier": 0.234,
 "by_type": {"choice": {"questions": 19, "accuracy": 89.47},
             "noul": {"questions": 13, "accuracy": 84.62},
             "score": {"questions": 9, "accuracy": 77.78}}}
```

`accuracy` 是问题级准确率；`ece` 是置信度校准误差，越低说明模型对自己有多确定说得越准；`brier` 同时惩罚“答错”和“答对但不够自信”。`by_type` 常常是最有用的一栏：总分不涨但某一类明显涨，也是有效结果。

## 4. 造你自己的数据

这是唯一无法由脚本代劳的部分，也是最影响结果的部分。格式的完整定义在 [finetuning.md](finetuning.md#数据)，这里只讲怎么下手。

### 4.1 先决定问题问什么

一条记录就是一次请求的形状：一个 `state`（要判断的东西），加上一组 `questions`（你要的答案）。**一个 state 可以挂多个问题**，它们共享同一段上下文，训练和推理都是这样。

| 题型 | 用在哪 | target 的写法 |
|---|---|---|
| `choice` | 互斥分类：分派给哪个团队、属于哪个类目 | `{"billing": 1.0}`，键必须是 `criteria` 的键 |
| `noul` | 二值判断：是否紧急、是否损坏 | `{"true": 1.0}` 或 `{"false": 1.0}` |
| `score` | 有序等级：严重程度、情绪强度 | `{"0": 1.0}` 到 `{"n-1": 1.0}`，0 是最低级 |

选型原则：能用 `choice` 就不要用 `score`，有序等级要求模型学到顺序，数据少时很难；二值问题用 `noul`，不要做成两个选项的 `choice`。

### 4.2 要多少条

仓库没有给出数据量下限，也不会给，因为它取决于问题的难度和选项数量。以下是经验起点，不是实测结论：

- **能看出趋势**：每个题型 200 到 500 个有标注的问题。低于这个量，`eval` 的置信区间会宽到无法判断好坏（第 6 节的实测就是例子：只翻转 3 个问题，95% 区间下界就是 0）。
- **每个选项都要有足够样本**：`choice` 里某选项只出现两三次，模型学不会它，评测也测不出它。
- **验证集单独留**：10% 到 20%，从同一批数据里随机切，不要按时间切（除非你的业务确实随时间漂移）。验证集不能出现在训练集里，冒烟时那种 `--dev` 与 `--train` 同文件的用法只用于验证管线。

### 4.3 标注时最容易犯的五个错

1. **target 的键和 `criteria` 不一致。** `choice` 的 target 键必须是 `criteria` 里的键，写错会被当成非法 target 报错（在 `finetune_lora.py` 里）或被静默跳过（在转换器里被跳过计数）。
2. **`score` 的 target 写成中文标签。** target 键是 `"0"`、`"1"`、`"2"`，等级名称写在 `criteria` 列表里。
3. **图片写成本地路径。** 只接受 base64 或 data URI，路径会在加载时报错。图片在 `state` 里用 `<image>` 标记位置，不加标记则图片排在文本最前。
4. **同一个 state 的多个问题标注矛盾。** 例如同一张工单既标成“紧急”又标成“严重度最低”，模型会学到噪声。标注完抽查一批多问题记录。
5. **软标签被当成四舍五入。** `{"billing": 0.7, "shipping": 0.3}` 表示标注者分歧，不是“以 0.7 为准”。如果下游只关心谁赢，直接给 one-hot，别混用。

### 4.4 会静默消失的数据

训练端仍然是“跳过而不报错”：选项超过 26 个的 `choice`、渲染后超过 `--max-length`（默认 8,192 token）的问题
都会被跳过，日志里两种原因分开计数；没有 `targets` 的问题在训练端被跳过，在检查器里是阻断错误。
所以数据交出去之前先跑检查器：

```bash
conda install pillow          # 只有这一个非标准库依赖
python3 finetune/check_data.py data/train.jsonl --report data/train.report.json --against data/dev.jsonl
python3 finetune/check_data.py data/train.jsonl --render 5
```

缺 `targets`、选项超过 26 个在自检里就是阻断错误；超长在带 `--tokenizer` 的验收运行里阻断（图片侧离线也判）。
完整的字段规则、规则编号和交接流程见 [finetune_data.md](finetune_data.md)。

### 4.5 用模型预标注来省力

如果已有大量未标注文本，常见做法是先用一个强模型预标注，再人工抽查修正。这样做可以，但要知道代价：预标注里的系统性偏差会被你的微调模型完整继承，而且 `eval` 测不出来（验证集也是这么标的）。至少保留一部分完全人工标注的验证集，用它来判断。

## 5. 训练

真实数据上通常不用改参数，先按默认跑：

```bash
.venv/bin/python finetune/finetune_lora.py \
  --model models/StartLux-Decision-v1.1-0.8B-Instant \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out runs/0.8b-lora \
  --calibrate
```

默认值：秩 16、alpha 32、dropout 0.05、学习率 1e-4（5% 预热后余弦衰减）、2 轮、每个优化步 4 个微批次、每个微批次最多 16,384 个填充 token、提示最长 8,192 token。只训练文本解码器上的 LoRA，可训练参数约占 1.4%。

**什么时候改参数**（按优先级）：

| 现象 | 先改 | 说明 |
|---|---|---|
| 显存不足 | `--max-tokens 8192` | 再不行降 `--max-length` 或 `--max-pixels` |
| 验证集 loss 先降后升 | `--epochs 1` | 过拟合，数据量小时尤其常见 |
| 几乎所有题都答对，概率过于尖锐 | 换更难更大的验证集 | 校准温度会顶到下界，见第 6 节 |
| 学不动，验证集准确率不涨 | `--lr 2e-4` 或 `--rank 32` | 一次只改一个 |

`--calibrate` 用验证集的读出结果，为每个题型拟合一个温度。温度只改变概率的尖锐程度，不改变哪个选项赢：如果准确率没变而 `ece` 下降，说明校准起了作用。

## 6. 前后对比与判读

用**同一个**服务参数起微调后的模型，跑同一份评测数据：

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/startlux-decide serve \
  --model /tmp/opencode/run-smoke/merged --backend transformers --port 18092 &

.venv/bin/startlux-decide eval --url http://127.0.0.1:18092 \
  --data /tmp/opencode/sample_eval.jsonl --out /tmp/opencode/preds-ft.jsonl

.venv/bin/startlux-decide eval --compare /tmp/opencode/preds-base.jsonl /tmp/opencode/preds-ft.jsonl
```

样例上的实测对比（**这是训练集等于验证集的演示，数字不能当作效果证据**，只为展示输出形状）：

```json
{"paired_questions": 41, "agreement": 92.68,
 "accuracy_a": 85.37, "accuracy_b": 92.68, "accuracy_diff_b_minus_a": 7.32,
 "accuracy_diff_ci95": [0.0, 15.38],
 "both_correct": 35, "only_a_correct": 0, "only_b_correct": 3,
 "mean_max_abs_dp": 0.0896, "max_abs_dp": 0.509}
```

怎么读：

- `accuracy_diff_b_minus_a` 是提升幅度，`accuracy_diff_ci95` 是配对 bootstrap 的 95% 区间。**区间跨过 0 就是“还不能说变好了”**，这里下界正好 0，只因为翻转了 3 个问题。这就是第 4.2 节强调数据量的原因。
- `only_a_correct` 比 `only_b_correct` 更值得警惕。前者是“原来对、现在错”，如果它不为 0，说明微调伤害了原有能力。
- `agreement` 高而两个准确率都低，说明模型很稳定地错在同一处，问题在数据而不在训练。
- `mean_max_abs_dp` 是每个问题里概率变化最大的那个选项的平均变化量。它接近 0 说明模型行为几乎没被改动，可能训练没生效或学习率太小。

**什么情况算成功**：目标题型准确率提升、`only_a_correct` 为 0 或很少、`ece` 不上升。三者同时满足再谈上线。

**什么情况该退回去**：验证集 loss 上升、`only_a_correct` 明显、`ece` 大幅上升（模型变得过度自信）。先减轮次、再降学习率，最后考虑加数据。

## 7. 常见失败对照

| 现象 | 原因 | 处理 |
|---|---|---|
| 第一行 `0 training questions` | 数据被全部跳过：没 `targets`、选项超 26 个、或提示超长 | 看 `skipped` 计数，逐条排查；临时加 `--max-length 16384` 验证是否是长度问题 |
| `... is not an Instant (prompt-bidirectional) folder` | 用了 v1 的因果模型目录 | Instant 用本脚本，v1 用 `v1/finetune/finetune_lora.py` |
| `fine-tuning needs torch` | 在没装 `[finetune]` extra 的解释器里跑 | 用 `.venv/bin/python`，不要用系统 python |
| CUDA out of memory | 微批次太大或其他进程占卡 | 降 `--max-tokens`，换空闲卡 |
| `fast kernels: False` | 缺 flash-linear-attention 或 causal-conv1d | 按 [finetuning.md](finetuning.md#环境) 重装，能训练但慢十倍 |
| 服务读出的结果和训练日志的 dev 准确率不一致 | 服务走了因果路径，或引擎版本不对 | Instant 目录必须用支持提示双向的引擎；transformers 后端会自己检查，vLLM 要 `--launch-vllm` 或插件开关 |
| `--calibrate needs --dev` / `drop --no-merge` | 校准需要验证集和合并后的模型 | 去掉 `--no-merge`，补上 `--dev` |

## 8. 跑通之后

- 换生产引擎：`startlux-decide serve --model runs/0.8b-lora/merged --launch-vllm --gpus 5 --port 8090`，Instant 目录会自动带上前提双向插件。
- 压缩体积：见 [quantization.md](quantization.md)，FP8 / GGUF / MLX 三种路线。
- 服务上线：见 [serving_publicly.md](serving_publicly.md)。
- 想改本仓库的微调脚本：先跑 `python3 -m unittest -q tests.test_finetune tests.test_to_eval_data`，仓库没有 lint 或 formatter 配置。

## 9. 术语对照

| 中文 | 英文 | 一句话 |
|---|---|---|
| 读出 | read-out | 一次前向传播，从选项字母的 logprob 得到概率，不生成文本 |
| 提示 | prompt | 每条问题渲染出来的完整输入，含整个 state |
| 双向提示注意力 | bidirectional prompt attention | Instant 模型的特性，提示内后文可见；按因果方式服务会得到不同的模型 |
| 门控 | gate | AutoThink 模型决定“直接答”还是“先想”的阈值 |
| 思考 | thought | AutoThink 模型生成的推理轨迹，Instant 没有 |
| 淘汰分组 | knock-out | 选项超过 26 个时先分组初筛，再对入围者做最后一轮 |
| 校准温度 | calibration temperature | 按题型缩放概率的系数，不改变谁赢 |
| 适配器 | adapter | LoRA 权重，几十 MB，可单独保存 |
| 合并后的模型 | merged | 适配器合并回全量权重后的目录，可直接服务 |
| 验证集 | dev set | 留出用于测 loss、准确率与拟合温度的数据 |
