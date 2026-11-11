> 译自 [README.md](README.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# StartLux-Decision

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.png">
  <img src="docs/assets/banner_light.png" alt="StartLux-Decision" width="100%">
</picture>

<p align="center">
  <a href="https://huggingface.co/startlux-models">Hugging Face</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2610.17309">Technical report</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://decision.startlux.com/">Blog</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="docs/zh/api.md">API reference</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://bamboonode.dev/">Free unlimited use</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.17309"><img src="https://img.shields.io/badge/arXiv-2610.17309-8A5CD6" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-Apache--2.0-8A5CD6" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.9%2B-8A5CD6" alt="Python"></a>
  <a href="#engines"><img src="https://img.shields.io/badge/backends-vLLM%20%7C%20llama.cpp%20%7C%20MLX-8A5CD6" alt="Backends"></a>
</p>

StartLux-Decision 是一系列开放的决策模型。给定状态（state）（文本、JSON 或图像）和一组带类型的问题，模型会为每个问题的每个选项返回一个概率：在所列选项中选择一个的选择题（choice）、一个是/否判断（noul），或量表上的一个等级（score）。Instant 模型以单次前向传播作答，对提示（prompt）施加双向注意力。AutoThink 模型在最高概率达到门控（gate）时一次作答，默认门控为 0.6，否则生成一小段推理轨迹，并再次读出答案。

本仓库包含这两种变体的推理引擎与 HTTP 服务端。API 兼容 System One `/v1/systemone`。

<p align="center"><img src="docs/assets/score_vs_size.png" width="860" alt="Decision Index 0.3 public score against model size"></p>

## Demos

<table align="center">
  <tr>
    <td width="50%" align="center"><a href="v1/media/rescue_harvey_27b.mp4"><img src="v1/media/rescue_harvey_27b.gif" alt="Coast Guard helicopter rescue"></a><br><sub>直升机救援：在飓风 Harvey 中，以每秒一帧的速度判断海岸警卫队吊运救援的每个阶段</sub></td>
    <td width="50%" align="center"><a href="v1/media/chess_27b_scotch_vs_jev.mp4"><img src="v1/media/chess_27b_scotch_vs_jev.gif" alt="Chess against Jev 1.13"></a><br><sub>国际象棋：执白，在 Scotch 开局第 19 步将死 Jev 1.13，双方都不做搜索</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><a href="v1/media/mario_1-1_27b.mp4"><img src="v1/media/mario_1-1_27b.gif" alt="Super Mario Bros."></a><br><sub>超级马力欧兄弟：通关 World 1-1</sub></td>
    <td width="50%" align="center"><a href="v1/media/sc2_hard_27b.mp4"><img src="v1/media/sc2_hard_27b.gif" alt="StarCraft II"></a><br><sub>星际争霸 II：为一个人族 bot 做战略决策，并击败内置的 Hard AI</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><a href="v1/media/launch_artemis_27b.mp4"><img src="v1/media/launch_artemis_27b.gif" alt="Artemis I launch"></a><br><sub>火箭发射：以每秒一帧跟踪 Artemis I 夜间发射，判断点火升空，并告诉摄像机何时上仰</sub></td>
    <td width="50%" align="center"><a href="v1/media/jevball_27b.mp4"><img src="v1/media/jevball_27b.gif" alt="JevBall"></a><br><sub>JevBall：为一支足球队实时做决策</sub></td>
  </tr>
</table>

在国际象棋和游戏的对局中，每一步都是一个带类型问题的请求，由首个发布版 StartLux-Decision-v1-27B 一次作答。救援与发射视频每秒发送一帧，附带九个问题，发送给 [bamboonode.dev](https://bamboonode.dev/) 上的 27B AutoThink 模型，该模型仅在其最高概率低于 0.6 门控时才进行思考。这些素材来自美国海岸警卫队和 NASA（公有领域，不构成任何背书），被救人员的面部已做模糊处理。点击预览图可观看视频。[v1/README.md](v1/README.md#demos) 描述了每个 harness，计算机操作与国际象棋的 harness 位于 [v1/demos](v1/demos)。

## Release notes

- 2026-10-16：StartLux-Decision v1.1 27B Instant 在 Hugging Face 的 [Decision Index 0.3.2](https://huggingface.co/spaces/multimodalart/jev-decision-index) 排行榜上，于 174 个条目中排名第一，Full 分数为 69.12。它在三个部分也各自第一：公开 benchmark（72.88）、同一技能的私有测试（64.82），以及来自新领域的私有任务（61.41）。
- 2026-10：技术报告已发布在 arXiv，编号为 [2610.17309](https://arxiv.org/abs/2610.17309)。

## Models

| StartLux-Decision v1.1 | Decision Index 0.3 | DecisionPivot | 其他格式 |
|---|---|---|---|
| [27B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink) | **73.37** | **86.9** | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink-MLX-4bit) |
| [27B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant) | 72.83 | 75.4 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-MLX-4bit) |
| [9B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink) | **70.45** | **59.9** | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink-MLX-4bit) |
| [9B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant) | 69.96 | 47.5 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant-MLX-4bit) |
| [4B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink) | **68.21** | **59.9** | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-MLX-4bit) |
| [4B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant) | 67.88 | 41.5 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant-MLX-4bit) |
| [2B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant) | 59.14 | 21.1 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant-MLX-4bit) |
| [0.8B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant) | 53.75 | 8.1 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant-MLX-4bit) |

Decision Index 0.3 指公开子集，使用公开工具包评估。DecisionPivot 报告 7,800 个程序化生成的样例上的成对准确率，这些样例按对比对排列，每对仅在一个决定性细节上不同。只有两个答案都正确时，一对才计为正确。AutoThink 的结果使用 0.9 的门控。

在各个规模上，Instant 模型在 Decision Index 0.3 上比同规模已评估的最强决策模型高出 7.4 到 16.9 分；AutoThink 相对 Instant 在 DecisionPivot 成对准确率上再增加 11.5 到 18.4 分。AutoThink 的推理使用 RLCD² 训练，即面向校准的审慎决策的强化学习。完整评估见技术报告。该引擎也服务 [v1 模型](https://huggingface.co/collections/startlux-models/startlux-decision-v1-6abba92b301b573fa154d493)（0.8B 到 27B 以及 35B-A3B），它们一次作答。

在一张 H200 上使用 vLLM、且只有一个在途请求时，从 0.8B 到 27B 的 Instant 模型响应的中位延迟为 25 到 52 ms。通过本服务端，AutoThink 模型在有把握时只需一次 38 到 59 ms 的读出。当它没有把握时，会写一段思考（thought）并再次读出答案，写得越多，耗时越长。

## Quick start

若想在无 GPU 的情况下试用模型，请使用 [bamboonode.dev](https://bamboonode.dev/)。它以相同的请求格式（`POST /v1/decisions`）托管 27B AutoThink 模型，免费且无使用限制，只需要一个 Bamboo API key。

在 NVIDIA GPU 上，[StartLux-Prism](https://github.com/StartLuxLabs/StartLux-Prism) 是我们为这些模型编写的推理引擎。它直接从已发布的 GGUF 文件服务 v1 与 v1.1 AutoThink（4B、9B 和 27B），位于相同的 `/v1/systemone` API 之后；尚不支持 v1.1 Instant 模型。在 Linux 上使用 Python 3.12 与 CUDA 13 驱动安装：

```bash
git clone https://github.com/StartLuxLabs/StartLux-Prism.git
cd StartLux-Prism
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # PyTorch 2.14.1 for CUDA 13, Triton, nvcc 13.0
source scripts/cuda_env.sh          # nvcc from pip as CUDA_HOME (in every new shell)
```

下载 GGUF 文件及对应发布版的 tokenizer 与 config，然后启动服务端：

```bash
M=~/models/StartLux-Decision-v1.1-27B-AutoThink-GGUF
hf download startlux-models/StartLux-Decision-v1.1-27B-AutoThink-GGUF \
  StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf --local-dir $M
hf download startlux-models/StartLux-Decision-v1.1-27B-AutoThink --exclude "*.safetensors" --local-dir $M
python -m startlux_prism.server --gguf $M/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf --model-dir $M --port 8090
```

当服务端打印 `Ready` 时即就绪；首次启动会编译 CUDA 内核，并一次性测量 int8 激活范围。它按原样回答下面的请求。v1.1 模型默认开启 `allow_none`（`--allow-none off` 与本包一致），服务端没有 API key 也没有 TLS，因此请将其置于你自己的访问控制之后。[StartLux-Prism 的 README](https://github.com/StartLuxLabs/StartLux-Prism#readme) 涵盖 CUDA 12 驱动、多 GPU 以及所有 flag。

若要自行运行这些模型，请将本包安装到与 vLLM 相同的 Python 环境。它带有 Instant 模型所需的 vLLM 插件。

```bash
git clone https://github.com/StartLuxLabs/StartLux-Decision.git
cd StartLux-Decision
pip install -e '.[vllm]'          # vLLM 0.31.0; the published numbers were measured on 0.19.1, which is also supported
```

下载模型，并为每张 GPU 启动一个 vLLM 服务端，把决策服务端放在它们前面：

```bash
hf download startlux-models/StartLux-Decision-v1.1-4B-AutoThink --local-dir models/StartLux-Decision-v1.1-4B-AutoThink
startlux-decide serve --model models/StartLux-Decision-v1.1-4B-AutoThink --launch-vllm --gpus 0,1,2,3 --port 8090
```

提一个问题：

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
    "urgent":   {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]}
  }
}'
```

27B AutoThink 模型的回答（数字已四舍五入）：

```json
{"answers": {
  "team":     {"type": "choice", "choice": "billing", "confidence": 0.992,
               "probabilities": {"billing": 0.995, "shipping": 0.000, "technical": 0.005},
               "none_probability": 0.002},
  "urgent":   {"type": "noul", "noul": 0.870},
  "severity": {"type": "score", "score": 1.858, "confidence": 0.787,
               "legend": {"0": "cosmetic", "1": "annoying", "2": "blocks the customer"},
               "probabilities": {"0": 0.005, "1": 0.133, "2": 0.863}}},
 "usage": {"input_tokens": 290, "output_tokens": 144}, "model": "startlux-decision-27b-autothink"}
```

`noul` 是回答“是”的概率。`score` 是期望等级，并给出每个等级的概率。`none_probability` 是所列选项都不合适的概率（见 Controls）。

[examples/](examples) 包含同一请求的 Python 版本、一个带图片的请求，以及一个小型批量客户端。

## Controls

请求可以引导 AutoThink 模型的推理，并要求更多细节。所有字段均为可选。

| 字段 | 默认值 | 作用 |
|---|---|---|
| `think_mode` | `auto` | `auto` 在第一次读出的最高概率低于 `gate` 时进行推理；`never` 一次作答；`always` 先推理 |
| `gate` | 0.6 | `auto` 据此进行推理的置信度阈值（已发布结果使用 0.9） |
| `think_budget` | 4096 | 思考的最大长度，以 token 计 |
| `none_check` | true | 每个选择题（choice）答案都会报告 `none_probability`，即没有所列选项合适的概率；`choice` 仍为所列选项之一 |
| `allow_none` | false（`serve --allow-none`：true） | 当“None of the above”是最可能的答案时，允许 `choice` 为 null |
| `return_details`, `return_thought` | false | 每个答案的读出与计时，以及思考文本 |
| `images` | 无 | 用作证据的 base64 字符串或 data URI |

默认门控 0.6 能让更多请求一次作答。门控 0.9 更常思考，在需要规则或算术的决策上更准确。对于本引擎上的 27B AutoThink 模型，把门控从 0.9 调到 0.6，会使 Decision Index 0.3 从 73.22 降到 72.85，DecisionPivot 成对准确率从 88.1% 降到 68.4%。Instant 模型始终一次作答。

我们建议把 0.2 及以上的 `none_probability` 视为“没有所列选项合适”。取 0.2 时，它能抓住所列选项全部错误的问题中的三分之二（66.3%），同时把 2.2% 确有正确答案的问题标出来。更低的阈值会标出更多好问题，更高的阈值会漏掉更多坏问题（[为什么取 0.2](docs/zh/none_probability.md#why-02)）。[docs/zh/api.md](docs/zh/api.md) 描述了完整的请求与响应格式、置信度定义、details 字段以及错误。

## Engines

| 引擎 | Instant | AutoThink | 图像 | 典型用途 |
|---|---|---|---|---|
| vLLM（默认） | 是，需使用本包中的插件 | 是 | vLLM 0.31 | 服务（serving），以及已发布数字 |
| [StartLux-Prism](https://github.com/StartLuxLabs/StartLux-Prism) | 尚不支持 | 是，4B、9B 和 27B | 配合 `mmproj` 文件 | NVIDIA GPU 上的 GGUF 文件，延迟最低 |
| llama.cpp | 是，标记为非因果的 GGUF 构建 | 是 | 配合 `mmproj` 文件 | 没有 vLLM 的机器 |
| MLX | 是，mlx-vlm 构建 | 是 | 是 | Apple Silicon |
| transformers | 是 | 是 | 是 | 参考校验 |

FP8 与 GGUF Q8_0 基本保留 BF16 的决策性能，而 Q4_K_M 最多可能损失 4.3 分（27B AutoThink 在 DecisionPivot 上）。[docs/zh/deployment.md](docs/zh/deployment.md) 列出了需要开启和关闭的引擎设置及其原因，[docs/zh/quantization.md](docs/zh/quantization.md) 介绍量化构建。

## Serving in production

`startlux-decide serve` 可以直接面向互联网，支持 API key、HTTPS、负载下提前返回 429、访问日志以及可选的请求日志。[docs/zh/serving_publicly.md](docs/zh/serving_publicly.md) 给出完整命令和一份检查清单。其他命令用于衡量部署：`bench` 测延迟，`eval` 在带标注的请求上测准确率与校准，`quantize` 生成 FP8、GGUF 与 MLX 构建。

## Fine-tuning

`finetune/finetune_lora.py` 用 LoRA 在文本或图像记录上，把 v1.1 Instant 模型目录适配到你自己的带类型的决策上，然后把适配器（adapter）合并进一个可由相同引擎服务的目录。它按照 Instant 模型的训练方式双向读取提示（prompt），并可在验证集（dev）上拟合各类型的温度。[docs/zh/finetuning.md](docs/zh/finetuning.md) 包含环境、数据格式与命令；首个发布版的因果训练器保留在 `v1/finetune/`。

## Documentation

| 文档 | 内容 |
|---|---|
| [docs/zh/api.md](docs/zh/api.md) | 请求与响应格式、控制项、置信度、`none_probability`、错误 |
| [docs/zh/finetuning.md](docs/zh/finetuning.md) | 在你自己的决策上对 v1.1 Instant 模型做 LoRA 微调 |
| [docs/zh/none_probability.md](docs/zh/none_probability.md) | `none_probability` 的含义、测量方式、为什么取 0.2 |
| [docs/zh/deployment.md](docs/zh/deployment.md) | 引擎设置、图像与长输入、与 v1 模型的兼容性 |
| [docs/zh/engines.md](docs/zh/engines.md) | 每种引擎如何驱动，以及测了什么 |
| [docs/zh/vllm_bidirectional.md](docs/zh/vllm_bidirectional.md) | 用于提示（prompt）双向注意力的 vLLM 插件 |
| [docs/zh/quantization.md](docs/zh/quantization.md) | FP8、GGUF 与 MLX 构建 |
| [docs/zh/serving_publicly.md](docs/zh/serving_publicly.md) | key、TLS、限流、日志 |
| [docs/zh/benchmarks.md](docs/zh/benchmarks.md) | 准确率与延迟如何测量 |

## Tests

```bash
python -m unittest -v
```

测试只需要标准库；缺少 aiohttp、torch 或 transformers 时，会跳过需要它们的部分。将 `STARTLUX_TEST_MODEL_DIR` 设为一个模型目录，即可增加针对其真实 tokenizer 的检查。

## The first release

StartLux-Decision-v1（0.8B 到 27B，以及一个 35B-A3B 的专家混合模型）一次作答每个问题。其推理代码、评估脚本、结果、对局日志与演示 harness 都在 [v1/](v1)，该发布时点的仓库打了 `v1` 标签。本仓库中的引擎也服务 v1 模型。

## Citation

Xialie Zhuang<sup>1,2</sup>, Zunhai Su<sup>1,2</sup>, Jia Pan<sup>2</sup>, Ngai Wong<sup>2</sup>, Guanhua Chen<sup>3,\*</sup>, Chuan-Wei Kuo<sup>1,\*</sup>

<sup>1</sup>StartLux&nbsp;&nbsp;<sup>2</sup>The University of Hong Kong&nbsp;&nbsp;<sup>3</sup>Southern University of Science and Technology&nbsp;&nbsp;<sup>\*</sup>Corresponding authors

```bibtex
@misc{startlux2026decision,
  title         = {StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions},
  author        = {Zhuang, Xialie and Su, Zunhai and Pan, Jia and Wong, Ngai and Chen, Guanhua and Kuo, Chuan-Wei},
  year          = {2026},
  eprint        = {2610.17309},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.17309}
}
```

## License

本仓库中的代码以 [Apache License 2.0](LICENSE) 发布。模型权重以 CC BY-NC 4.0 发布，仅限非商业使用。商业授权请联系 StartLux。
