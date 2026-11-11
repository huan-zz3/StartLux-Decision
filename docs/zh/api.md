> 译自 [../api.md](../api.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# API

服务端（server）使用 TypeSafe 的 `/v1/systemone` 格式，因此 TypeSafe 客户端可以直接对接。`POST /v1/decisions` 是同一
API 在 OpenRouter 名下的版本。请求中的每个问题都作为独立任务运行：某个问题在思考（thought）时不会拖住
其他问题，并且可以同时有多个请求在处理。公开提供服务（API keys、HTTPS、提前返回 429、OpenRouter models
文档、访问日志）：[serving_publicly.md](serving_publicly.md)。

## POST /v1/systemone（以及 /v1/decisions）

### 请求

```json
{
  "state": "any text, or any JSON value",
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": null}},
    "urgent":   {"type": "noul", "instructions": "Answer today?"},
    "severity": {"type": "score", "instructions": "How bad?", "criteria": ["minor", "major", "critical"]}
  },
  "images": ["data:image/jpeg;base64,/9j/4AAQ..."],
  "think_mode": "auto", "gate": 0.6, "think_budget": 4096, "mix": 0.75, "thought_generation": "auto",
  "allow_none": false, "none_check": true, "return_details": false, "return_thought": false
}
```

| 问题类型 | 判定标准（criteria） | 回答 |
|---|---|---|
| `choice` | `{选项 id: 描述或 null}`，或选项 id 的列表 | `choice`、`confidence`、覆盖每个选项的 `probabilities` |
| `noul`（是/否问题（noul），也接受 `bool`） | 可选的 `{"true": 描述, "false": 描述}` | `noul`：答案为"是"的概率 |
| `score` | 2 到 10 个等级的列表，由低到高排序，或使用数字键的 `{key: level}`（按键升序排序） | `score`（按概率加权的平均等级）、`confidence`、`legend`、以 `"0"` 到 `"n-1"` 为键的 `probabilities` |

选择题（choice）与量表题（score）必须提供 `instructions`；是/否问题（noul）缺少它时会问 "Which answer fits the
evidence?"。JSON 类型的状态（state）会渲染为紧凑 JSON；元素数量为 8 个或更多的数组会给每个元素带上 `_index`。

图片（可选，与 v1 相同）：`images` 是 base64 字符串或 data URI 的列表。在状态（state）中，`<image>` 标记每张
图片的插入位置，每张图片一个标记，按顺序对应（JSON 状态可以把这些标记放在它的字符串里）。没有标记时，
图片排在前面（有多张时，`Image 1:`、`Image 2:` ... 各自单独成行），随后是一个空行和状态（state）。每张图片
保持宽高比，缩放到最多 1,048,576 像素、至少 65,536 像素，且两边都是 32 像素的整数倍，并按每 32 × 32 像素
一个 token 计（一张 1024 × 1024 的照片是 1,024 个 token）。请求中的每个问题都会读取每张图片，`usage.input_tokens`
会把图片 token 计入。默认每个请求最多 8 张图片（`serve --max-images`、`--max-pixels`；`--no-images` 只提供纯文本
服务）。图片需要模型目录带有视觉塔（vision tower），且引擎能读取它们：vLLM 0.31 或更高版本、transformers、
llama-server 带 mmproj 文件时的 llama.cpp（[engines.md](engines.md#llamacpp)），或使用 mlx-vlm 模型目录的
MLX。其他引擎会返回 422。

长状态：提示（prompt）可以长到引擎的上下文上限。在 vLLM 0.31 上，AutoThink 与 v1 模型目录是 262,144 个 token
（这些模型的原生上下文），Instant 模型目录是 131,072 个 token，除非服务端启动时给了更低的 `--max-model-len`。
更长的提示会得到 422，并附带引擎给出的原因。AutoThink 与 v1 模型目录对同一请求中各问题共享的长状态只读
一次；Instant 模型目录会为每个问题重新读取，因为它对每个提示做双向（bidirectional）读取。

选择题列表（choice list）：一次读出（read-out）最多读取 26 个选项（字母 A 到 Z）。选项更多的列表会做淘汰
分组（knock-out）：大小相近、每组最多 25 个选项的分组各用一次读出处理，每组最好的 3 个进入最后一轮（符合
条件的超过 26 个时，取总体最好的 26 个），未进入最后一轮的选项按其所在分组那一轮的比例分享 0.001 的概率。
最后一轮是普通问题（它可以思考）。只有一个选项的选择题不经过模型直接作答（概率为 1）。

"以上皆非"（"None of the above"，仅限选择题（choice））。`serve` 默认会检查每个选择题是否缺少答案
（`none_check`）：该问题会在最后多出一个选项 "None of the above"，答案会带上 `none_probability`，
而 `choice` 仍然是列出的选项中最可能的那个。在 51,506 个基准问题上，这没有改变准确率（开启思考时，有它
和无它都是 84.66）。在请求里写 `"none_check": false`，或用 `serve --no-none-check`，可以关掉它。设置
`"allow_none": true`（需主动开启）时，选择题也会在最后多出一个选项 "None of the above"，像其他选项一样
展示给模型（思考也能看到它，并可以选择它）；读出覆盖 n + 1 个选项。此时答案里，当该选项是最可能的选项时
`choice` 为 null（概率相同时归给列出的选项），`probabilities` 只覆盖你的选项，并重新归一化到和为 1
（忽略新字段的客户端看到的是普通分布），还有 `none_probability`，即额外选项在归一化之前的概率。`confidence`（置信度）
照旧由重新归一化后的概率计算，因此它描述的是你的选项之间的分散程度，而 `none_probability` 才是要据此行动的
信号。`allow_none` 打开时，只有一个选项的选择题也会经过模型读取（两个选项），而不是直接作答。在淘汰分组的
列表里，额外选项只在最后一轮加入；26 个选项的列表也会做淘汰分组（第 26 个字母就是额外选项）。你自己命名为
"None of the above" 的选项会得到 422。`allow_none` 关闭时（除非服务端以 `--allow-none` 运行，否则这是默认
值），什么都不变：提示、答案和分数与没有该控制项时完全相同。`allow_none` 也可以设在单个问题上
（`{"type": "choice", ..., "allow_none": true}`），它会覆盖请求级设置；`null` 表示没有给出。其他问题类型
忽略它。

读取 `none_probability`：它是报警分数，不是校准（calibration）过的概率，我们建议的阈值是 0.2。在 DecisionPivot
与 Decision Index 0.3 的 51,506 个选择题上（27B AutoThink，开启思考），每个问题按发布时的样子问一次，再在
去掉正确选项后问一次，阈值 0.2 抓住了 66.3% 没有正确选项的问题，并把 2.2% 的已发布问题标记出来
（AUROC 0.920）。[none_probability.md](none_probability.md) 里有其他阈值、为什么是 0.2、哪些任务上它不可靠，
以及它如何与思考相互作用。单靠置信度区分这两类问题要差得多，因为有正确选项的难题读出来也低。额外
选项还给了思考一条出路：没有它时，没有合适选项的问题常常让思考原地打转直到预算耗尽，因为模型仍然必须
挑一个列出的选项。

`confidence` 遵循 TypeSafe：选择题为 (p_max − 1/n) / (1 − 1/n)，截断到 [0, 1]；量表题为 1 减去最可能等级到
各等级按概率加权的等级距离，除以从中位等级量起的均匀分布下的同一距离，下限为 0。是/否问题没有
置信度；|2p − 1| 是等价的量。

### 控制项

| 字段 | 取值 | 作用 |
|---|---|---|
| `think_mode` | `auto` / `never` / `always`（也接受 `hybrid` / `plain` / `think`） | `auto`：第一次读出的最高概率低于 `gate` 时思考 |
| `gate` | [0, 1] 内的数字 | `auto` 的阈值 |
| `think_budget` | 整数 ≥ 1 | 思考 token 的上限；更大的值会被服务端上限截断（`serve --max-think-budget`，默认值：默认预算） |
| `mix` | [0, 1] 内的数字 | final = (1 − mix) · 第一次读出 + mix · 思考后的读出；0 表示忽略思考，1 表示只用思考后的读出 |
| `thought_generation` | `auto` / `stream` / `chunked` | 思考的生成方式。`stream`：一次流式请求；循环检测器在每收到 1,024 个 token 时检查一次思考，一旦发现循环就结束请求。`chunked`：每次请求 1,024 个 token，每次重新发送提示和到那时为止的思考，中间做循环检测。`auto`：引擎支持流式思考（vLLM 0.31 或更高版本）时用 `stream`，否则用 `chunked`。在其他引擎上用 `stream` 会得到 422 |
| `allow_none` | bool，默认 false（`serve --allow-none` 时为 true） | 选择题在最后多出一个选项 "None of the above"（见上文）；单个问题自己的 `allow_none` 会覆盖它；`noul` 与 `score` 问题忽略它；非布尔值会得到 422 |
| `none_check` | bool，`serve` 中默认 true | 选择题在内部多出一个选项并报告 `none_probability`；除非设置了 `allow_none`，`choice` 仍是列出的选项；非布尔值会得到 422 |
| `return_details` | bool | 给每个答案加上 `details`，并给响应加上 `timing_ms` |
| `return_thought` | bool | 同时返回思考文本；隐含 `return_details` |

字段缺失或为 null 时取服务端默认值。服务端默认值先来自模型目录（内置的 AutoThink 默认值，或它的
`decision_config.json` 里的 `auto_think` 块；v1 模型目录一次读出作答，即 `never`），然后来自 `serve` 的选项。
Instant 模型从不思考：`think_mode` 为 `auto` 或 `always` 会得到 422；`gate`、`mix` 和 `think_budget` 被接受但
不起作用。思考还需要引擎上下文里有空间：预算会被削减为上下文长度减去提示、再减去 64 个 token，若剩下
32 个 token 或更少，该问题就用第一次读出作答。

长思考：跑到预算的思考是请求中最慢的部分。在 A800 上以 BF16 运行时，一个思考的生成速度约为
27 token/s，所以 4,096 个 token 大约要 150 秒。这类思考的 `details.thought.ended` 是 `budget`。在上面那
10,000 个问题上，4% 的问题跑到了预算，用掉了全部思考 token 的三分之一以上。其中多数是难题，长思考在那里
仍然划算（38% 答对，而只靠第一次读出是 22%）。没有合适选项的问题更容易跑到预算（5.9%，打开 `allow_none`
后是 3.9%），而在那里长思考毫无收益。有三件事能让这类问题上的思考保持简短。只问适用于当前步骤的问题
（已经决定滚动的 agent 不应同时问点哪个链接）。在情况可能落在列表之外时，加上"不可见"或"其他"之类的选项。
在列表可能不完整时打开 `allow_none`。调低 `think_budget` 会让每个难题都付代价：在 DecideBench 上，27B
AutoThink 模型在 4,096 个 token 时得 88.6，512 时得 80.4，256 时得 68.4，而一次读出作答（`never`）大约是
70。

### 响应

```json
{
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.81, "probabilities": {"billing": 0.905, "shipping": 0.095}},
    "urgent": {"type": "noul", "noul": 0.64},
    "severity": {"type": "score", "score": 1.3, "confidence": 0.19, "legend": {"0": "minor", "1": "major", "2": "critical"},
                 "probabilities": {"0": 0.12, "1": 0.46, "2": 0.42}}
  },
  "usage": {"input_tokens": 290, "output_tokens": 0},
  "model": "StartLux-Decision-4B"
}
```

设置 `allow_none` 时，选择题的答案还会带上 `none_probability`，并且 `choice` 可能为 null：

```json
"team": {"type": "choice", "choice": null, "confidence": 0.33, "probabilities": {"billing": 0.667, "shipping": 0.333},
         "none_probability": 0.7}
```

`usage.input_tokens` 统计的是每次单遍提示读取的 token（包括淘汰分组的所有轮次）；
`usage.output_tokens` 统计思考 token。

### 详情与耗时

设置 `return_details` 时，每个答案都带上：

```json
"details": {
  "first_pass":    {"probabilities": {"billing": 0.55, "shipping": 0.45}, "top": 0.55, "ms": 21.4},
  "thought":       {"used": true, "tokens": 812, "ended": "closed", "ms": 2140.7, "generation": "stream",
                    "text": "Let me work through ..."},
  "after_thought": {"probabilities": {"billing": 0.97, "shipping": 0.03}, "ms": 24.9},
  "gate": 0.6,
  "mix": 0.75,
  "total_ms": 2188.3
}
```

| 字段 | 含义 |
|---|---|
| `first_pass.probabilities` | 第一次读出，键与答案一致（淘汰分组后：覆盖每个选项）；设置 `allow_none` 时还带键 `None of the above`，并在所有选项上按读出的原值求和为 1（未重新归一化） |
| `first_pass.top` | 在最后一轮所读选项上的最高概率：与门控（gate）比较的值 |
| `first_pass.ms` | 第一次读出的引擎调用耗时（淘汰分组的各轮与最后的读出） |
| `thought.used`、`thought.tokens` | 该问题是否思考过，以及生成的思考 token 数 |
| `thought.ended` | `closed`（模型自己结束了思考）、`loop`（循环检测器结束了它）、`budget`（被截断：到达预算或上下文；此时读出会看到一行 "I have to decide now with what I have worked out so far."）、`skipped`（没有思考） |
| `thought.ms` | 从发出思考请求（第一个分块）到收到它的结束（最后一个分块，含波次等待） |
| `thought.generation` | `stream` 或 `chunked`（见控制项）；仅在该问题思考过时存在 |
| `thought.text` | 设置 `return_thought` 时：完整思考，含强制的开头；其后的读出只看到最后 4,000 个字符 |
| `after_thought` | 思考后的读出（已重新调温），及其引擎调用耗时；没有思考时为 null |
| `gate` | 第一次读出所比较的门控（gate）；未查询它时为 null（`never`、`always`、Instant、单选项列表） |
| `mix` | 实际应用的混合权重；没有思考时为 null |
| `total_ms` | 该问题的整条管线（pipeline） |

响应还会带上 `timing_ms`：`wall`（请求在服务端内的耗时，从校验到给出答案）、`first_pass_sum` 与
`after_thought_sum`（各问题读出耗时之和；问题并发运行，所以这些值可能超过 `wall`），以及 `thought_span`
（从第一个思考的第一个分块到最后一个思考的最后一个分块；没有思考时为 0）。

### 错误

每个响应都带有 `x-request-id`（若客户端发来合法值就用它，最长 128 个 `A-Z a-z 0-9 . _ : -` 字符），并在
`x-typesafe-request-id` 里给出相同的值。错误是 JSON：

```json
{"error": "gate must be a number from 0 to 1, not 2", "detail": [{"loc": ["body", "gate"], "msg": "...", "type": "value_error"}]}
```

| 状态码 | 何时 |
|---|---|
| 401 | 服务端配置了 API keys，而请求没有带其中之一对应的 `Authorization: Bearer <key>`（`/health` 之外的所有路由） |
| 429 | 服务端已达到 `--max-inflight` 个决策请求；立即返回并带 `Retry-After: 1` |
| 422 | 请求体不是 JSON 或不是对象；某个控制项取值不合法（请求级或选择题上的 `allow_none` 不是 true 或 false 也算）；某个问题无法渲染（缺少 instructions、11 个 score 等级、重复的 id 等）；Instant 模型被要求思考；`images` 无法读取、图片数量超过上限、`<image>` 标记与图片不匹配，或在只读文本的引擎/模型目录上提供了图片；引擎拒绝某个问题（提示超过其上下文） |
| 404, 405 | 未知路径或方法 |
| 502 | 引擎失败或无法连接（每次调用试四次） |

401 或 422 是最终结果；429 或 502 可以重试。请求里的 `model` 字段会被接受并忽略；响应中的 `model` 是服务端的
名字。

## 模型目录

模型目录存放权重、tokenizer、`config.json` 和 `decision_config.json`。包会读取：

| `decision_config.json` 键 | 含义 |
|---|---|
| `temperature_by_type` | 每种问题类型的读出温度（`choice`、`noul`、`score`；缺失的类型用 1.0） |
| `letter_token_ids` | A 到 Z 的 token id，会与 tokenizer 核对 |
| `max_options_per_pass` | 26（其他值会被拒绝） |
| `wide_choice` | `{"group": 25, "keep": 3, "residual": 0.001}`：选项超过 26 个的选择题列表的淘汰分组 |
| `prompt_attention` | Instant 模型目录中为 `"bidirectional"`；必须与 `config.json` 里的 `jev_prompt_attention` 一致 |
| `release_date` | 由 `/v1/models` 报告（可选） |
| `auto_think` | 可选，仅因果（causal）模型目录：覆盖下面的 AutoThink 默认值 |

目录的类型由 `config.json` 决定：`"jev_prompt_attention": "bidirectional"` 是 Instant 模型目录，其他都按
AutoThink 提供。只为一次读出训练的因果（causal）模型目录可以带上 `"auto_think": {"think_mode": "never"}`
（或以 `--think-mode never` 启动）。

| `auto_think` 键 | 默认值 | 含义 |
|---|---|---|
| `think_mode` | `"auto"` | 请求字段的默认值 |
| `gate` | 0.6 | 请求字段的默认值（发表的数字用的是 0.9） |
| `mix` | 0.75 | 请求字段的默认值 |
| `budget` | 4096 | `think_budget` 的默认值 |
| `prefix` | `"Let me work through this carefully, step by step.\n"` | 每个思考的强制开头 |
| `tail_chars` | 4000 | 思考后的读出只看到最后 N 个字符，并标记为 `[...]`；0 表示全部 |
| `chunk_tokens` | 1024 | 思考按每 N 个 token 生成一次，中间做循环检测；0 表示一次调用、不做循环检测 |
| `think_temperature` | `{"choice": 1.0, "noul": 1.0, "score": 1.0}` | 思考后的读出会从类型温度重新调到这个温度；未列出的类型不重新调温 |
| `sampling` | `{"temperature": 0.6, "top_p": 0.95, "top_k": 20}` | 思考的采样 |

未知的键和超出范围的值会在服务端启动时被拒绝。

## GET /v1/models

`{"models": [{"name", "description", "release_date"}]}`。`release_date` 取自 `decision_config.json`
（`release_date`），否则取该文件的日期。

## GET /health

状态、模型名与类型（`instant`、`auto_think` 或 `v1`）、服务端是否读取图片、提示注意力（prompt attention）、
引擎及其设置、默认控制项、请求可要求的最大思考预算，以及用于空间检查的上下文长度。带 API keys 的服务端
对没有合法 key 的请求只返回状态、模型名和类型，以便其引擎的地址和设置保持私密。

## Python

```python
import asyncio
from startlux_decide.backends.vllm import VLLMBackend
from startlux_decide.modelcfg import load_model_folder, load_tokenizer
from startlux_decide.pipeline import Decider

async def main():
    folder = load_model_folder("/models/StartLux-Decision-4B")
    engine = VLLMBackend(["http://127.0.0.1:8001"], load_tokenizer(folder.path))
    await engine.start()
    decider = Decider(folder, engine)
    result = await decider.decide(state, questions, decider.controls({"return_details": True}))
    print(result.answers, result.usage, result.timing_ms)
    await engine.close()

asyncio.run(main())
```
