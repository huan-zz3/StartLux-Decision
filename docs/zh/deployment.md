> 译自 [../deployment.md](../deployment.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# 部署

服务器与引擎如何处理图片、长提示（prompt）以及两个模型发布版，还有哪些引擎设置要打开或关闭。下面每项设置都经过测量；[engines.md](./engines.md) 和 [quantization.md](./quantization.md) 给出原因与数字。

## 部署设置

`launch-vllm` 读取 vLLM 版本并挑选下面的参数；在 vLLM 0.31 上，这些设置就是参考部署所用的设置（前缀缓存（prefix caching）与分块预填充（chunked prefill）开启，每步 8,192 个批（batch）token）。

| 引擎 | 设置 | 开启或关闭 | 原因 |
|---|---|---|---|
| vLLM 0.19.1 | `--no-enable-prefix-caching` | 关闭前缀缓存 | 在 0.19.1 上，它会在不报错的情况下把这些混合模型 2.6% 的读出移动超过 0.05 |
| vLLM 0.19.1 | `--no-enable-chunked-prefill`，`--max-num-batched-tokens` 等于 `--max-model-len` | 关闭分块预填充 | 在 0.19.1 上，被拆分到多个调度器步骤的提示会被读错 |
| vLLM 0.31 | `--enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192` | 对 AutoThink 和 v1 开启 | 在 0.31 上，前缀缓存只有在分块预填充开启时才能在这些模型上生效；两者都开启时，27B AutoThink 模型在 823 个问题中有 99.88% 的答案与 0.19.1 上相同；思考（thought）不再重读它的提示 |
| vLLM 0.31 | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | 当文件夹包含 MTP 权重时开启；`--no-mtp` 将其关闭 | MTP 层每一步提议 3 个 token，因此思考生成得更快；在我们对 4B、9B 和 27B 的检查中，思考保持了其长度分布 |
| vLLM 0.31 | 把思考作为一个流式请求（`--thought-generation`，默认 `auto`） | 开启 | 不会出现某个分块重发提示与截至当时的思考的情况；循环检测器仍每 1,024 个 token 检查一次思考，并在那里终止陷入循环的思考 |
| vLLM，Instant | 关闭前缀缓存与分块预填充，不使用 MTP，适用于每个版本 | 必需 | Instant 插件需要每个提示在一步内整体预填充 |
| vLLM | `--logprobs-mode processed_logprobs --max-logprobs 26` | 开启 | 读出恰好返回选项字母 |
| vLLM | `--generation-config vllm` | 开启 | 让文件夹的 top-k 与 top-p 不作用于读出 |
| vLLM | 提示双向插件（`launch-vllm` 会设置 `STARTLUX_PREFIX_BIDIR=1`），vLLM 0.19.1 或 0.31.0 | 对 Instant 开启 | Instant 模型是在提示双向读取的条件下训练的；没有它，它们会像一个不同的模型那样作答 |
| vLLM | 思考波次（`--waves` / `--no-waves`；默认：对早于 0.31 的服务器开启） | 在 vLLM 0.19.1 上开启 | 在 0.19.1 上，加入正在运行的解码批的思考可能变成循环；0.31 修复了这一点 |
| vLLM | `--quantization fp8` | 可选，仅 27B | 在 27B 上差异在 0.2 分以内；会让 4B Instant 模型损失 1.3 个 DecideBench 分，并把决策转向思考 |
| vLLM | `--kv-cache-dtype fp8` | 关闭 | 插件会拒绝，而且它会把决策转向思考 |
| vLLM 0.31 | `--limit-mm-per-prompt '{"image":8,"video":0}' --mm-processor-kwargs ... --trust-request-chat-template` | 对带视觉塔的文件夹开启；`--no-images` 将其关闭 | 图片调用会带上客户端透传的对话模板（chat template），服务器会在本包允许的范围内缩放图片；Instant 服务器还会得到 `--skip-mm-profiling`（用 vLLM 对 256 张全尺寸图片做 profile 时，4B Instant 服务器无法启动） |
| vLLM | `--max-model-len` | 在 0.31 上取文件夹原生的 262,144，Instant 为 131,072；在 0.19.1 上为 65,536 | Instant 的提示要在一步内读完，而在 0.31 上这一步在 196,000（27B）和 256,000 个 token（4B）时让 vLLM 崩溃；超过上限的提示会得到 422 |
| llama.cpp | 用 [`patches/llama.cpp/0002`](../../patches/llama.cpp) 转换的 GGUF（或 `--override-kv qwen35.attention.causal=bool:false`） | 对 Instant 开启 | 把提示标记为双向；按因果方式读时，4B Instant 模型会损失 27 个 DecideBench 分 |
| llama.cpp | 原版 llama-server：`--parallel 1 --ctx-checkpoints 0 -b 16384 -ub 16384`；或用 [`patches/llama.cpp/0001`](../../patches/llama.cpp) 搭配任意 `--parallel` | 对 Instant 必需 | 否则提示会被成批处理或被拆分，这会让 4B Instant 模型损失 7.7 个 DecideBench 分 |
| llama.cpp | 提示缓存（`cache_prompt`） | 关闭（后端（backend）发送 false） | 复用前缀会破坏双向读取 |
| llama.cpp | GPU 采样（`-bs`） | 关闭 | 它让读出在对数概率上最多变化 0.40 |
| llama.cpp | 量化 | 推荐 Q8_0 | Q4_K_M 在 2B 到 9B 上会损失 1.6 到 3.8 个 DecideBench 分 |

## 图片与长输入

请求可以把图片作为其证据（evidence）的一部分携带，而提示可以长到 262,144 个 token（256K），即模型的原生上下文。这是 v1 的请求格式。

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Photo taken at delivery: <image>",
  "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}},
  "images": ["data:image/jpeg;base64,'"$(base64 < parcel.jpg | tr -d '\n')"'"]
}'
```

- `images` 是 base64 字符串或 data URI 的列表。状态（state）中的 `<image>` 标记每张图片放在哪里，每个标记对应一张图片，按顺序。没有标记时，图片排在前面（有多张时写成 `Image 1:`、`Image 2:` ...），后面跟着状态。
- 每张图片会被缩放到最多 1,048,576 像素，保持其宽高比（`--max-pixels`），并按每 32 × 32 像素一个 token 计入：一张 1024 × 1024 的照片是 1,024 个 token。请求中的每个问题读同一批图片。每个请求最多 8 张图片（`--max-images`）；`--no-images` 只服务文本。
- 图片在 vLLM 0.31 或更新版本、transformers 引擎、llama.cpp（llama-server 需用模型的 `mmproj-*-F16.gguf` 文件启动；没有它，带图片的请求会得到 422）和 MLX（mlx-vlm 文件夹）上都可用，适用于 v1 与 v1.1 文件夹（它们都保留视觉塔）。
- 在 vLLM 0.31 上对照 transformers 引擎、用同样的 12 个图片问题做过检查：在 4B 与 27B AutoThink、4B 与 27B Instant 以及 v1 4B 的 60 次读出中有 59 次首选选项相同，每个模型的最大概率差为 0.04 到 0.09。同样的问题去掉图片后差异最多 0.012，而两个引擎的图片处理器给出完全相同的像素：图片差异来自两个引擎的视觉路径（在不使用 Instant 插件的因果文件夹中也是如此）。因此带图片的读出接近但不等于参考结果。

长输入：

- 在 vLLM 0.31 上，服务器接受最长到文件夹原生 262,144 个 token 的提示，Instant 服务器最长到 131,072（`launch-vllm` 的默认值；在 0.19.1 上如以前一样是 65,536）。`--max-model-len` 设置服务（serving）上限（在 80 GB GPU 上 65,536 是个好选择）；更长的提示会得到 422，并附上引擎给出的原因。
- AutoThink 与 v1 文件夹每个请求只读一次长状态：本包把各问题共享的开头读入 vLLM 的前缀缓存，把请求中的每个问题发给那台服务器，之后每个问题只读自己那部分。思考也不会重读任何内容。
- Instant 文件夹把每个提示整体、在一步内读完，因为它们的提示注意力是双向的：带 n 个问题的请求会把它的状态读 n 次，而一步必须容纳整个提示。在 vLLM 0.31 上，这一步在 256,000 个 token 的提示（4B）和 196,000 个 token 的提示（27B）时让服务器崩溃，这就是 Instant 服务器停在 131,072 个 token 的原因。
- 长提示的代价，每台服务器一张 H200，答案在状态的最初几行，此前没有任何缓存：

  | 模型 | 32,000 个 token | 128,000 | 256,000 | 同一状态再来一次 |
  |---|---|---|---|---|
  | 4B AutoThink | 0.65 s | 3.8 s | 11.0 s | 0.14 / 0.45 / 0.96 s |
  | 27B AutoThink | | 15.9 s | 42.0 s | 0.46 / 1.04 s |
  | 4B Instant | 0.78 s | 7.3 s | | 每个问题都会重读 |
  | 27B Instant | | 24.7 s | | 每个问题都会重读 |

  每个答案都是对的。对同一状态提第二个问题，给因果模型增加 0.2 到 1.3 s，而 4B Instant 模型的时间几乎翻倍（32,000 个 token 时 1.5 s，128,000 时 12.7 s）。在完整上下文下，4B AutoThink 服务器的缓存可容纳 13.7 个 262,144 token 的提示，27B 为 4.2（[engines.md](./engines.md) 有细节）。

## 兼容性

引擎为两个发布版提供决策服务，支持图片与长输入：

| 发布版 | 文件夹 | 如何作答 |
|---|---|---|
| v1.1 | Instant 与 AutoThink 文件夹（0.8B 到 27B） | 如上 |
| v1 | `decision_config.json` 带 `"format": "StartLux-Decision-v1"`（0.8B、2B、4B、9B、27B 和 35B-A3B） | 默认一遍通过，提示与 v1.1 相同。请求仍可把 `think_mode` 设为 `auto` 或 `always`，但 v1 没有针对思考训练 |

图片在两个发布版上均可用于 vLLM 0.31、transformers、llama.cpp（需 mmproj 文件）和 MLX（mlx-vlm 文件夹），256K 上下文在 vLLM 0.31 和 transformers 上可用。由本引擎服务的 v1 4B 文件夹读取了 12 个图片问题，首选选项与 transformers 引擎相同，并在 4.1 s 内正确回答了一个 128,000 token 的请求。

## 公布的数字

公布的数字是在 vLLM 0.19.1 上用那里的默认值测量的：`launch-vllm` 的 vLLM 参数、每个文件夹的读出温度（temperature），以及 AutoThink 的门控（gate）0.9、预算 4,096、每块 1,024 个 token 的思考分块并在其间运行循环检测、在思考之后读取其最后 4,000 个字符、思考温度 1.0 和混合系数 0.75。在 vLLM 0.31 上，默认值改变的是思考的产生方式，而非决策规则：思考是一个流式请求，前缀缓存开启，MTP 每一步提议 3 个 token（见“部署设置”）。引擎默认的门控现在是 0.6，所以在 vLLM 上要用 `--gate 0.9` 提供服务，并对基准测试的请求运行 `startlux-decide eval`；[benchmarks.md](./benchmarks.md) 描述准确率与延迟（latency）的测量，[engines.md](./engines.md) 列出本包与那些运行的代码在何处不同（答案的置信度字段、用量计数和边界情况；从不涉及概率）。
