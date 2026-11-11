> 译自 [../engines.md](../engines.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# 引擎

每个引擎都实现两个原语（`startlux_decide/backends/base.py`）：

* `readout(prompt, letter_ids)`：一次前向，读出选项字母作为下一个 token 的对数概率（logprob）；
* `generate(prompt, max_tokens, sampling, stop, seed)`：一次采样续写，用于 AutoThink 模型的思考（thought）。

能安全流式输出思考的引擎（vLLM 0.31 或更高版本）还实现 `generate_stream`，即把同一段续写作为一次流式请求；提前关闭流会在引擎侧结束该请求。

提示（prompt）渲染、温度、门控（gate）、思考循环、混合、淘汰分组（knock-out）列表和答案格式都是本包自己的代码，对所有引擎相同。

## vLLM（默认）

vLLM 0.19.1（发布运行时所用的版本）或 0.31.0，每块 GPU 一个 OpenAI 兼容服务端。两个版本都用 Instant 插件验证过（[vllm_bidirectional.md](vllm_bidirectional.md#vllm-versions)）。`startlux-decide launch-vllm`（或 `serve --launch-vllm`）会读取运行 vLLM 的那个 Python 的 vLLM 版本，并用以下参数启动服务端：

```
all:      --gpu-memory-utilization 0.88 --logprobs-mode processed_logprobs --max-logprobs 26
          --generation-config vllm --served-model-name m
          --max-model-len: 0.31 or later 262144 (the folder's max_position_embeddings), Instant folders 131072;
                           0.19.1 65536; --max-model-len N sets it
0.19.1, and Instant folders on every version:
          --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens <max-model-len>
AutoThink and v1 folders on 0.31 or later:
          --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192
          --speculative-config '{"method":"mtp","num_speculative_tokens":3}'   (folders with MTP weights; --no-mtp: off)
images, on 0.31 or later for folders with a vision tower (--no-images: text only):
          --limit-mm-per-prompt '{"image":8,"video":0}' --trust-request-chat-template
          --mm-processor-kwargs '{"size":{"longest_edge":1048576,"shortest_edge":65536}}'
          (Instant folders also --skip-mm-profiling)
text only (0.19.1, --no-images, folders without a vision tower):
          --limit-mm-per-prompt '{"image":0,"video":0}'
```

| 参数 | 原因 |
|---|---|
| `--logprobs-mode processed_logprobs`、`--max-logprobs 26` | 一次读出用 `allowed_token_ids` 请求最多 26 个字母的对数概率；processed 对数概率是在该掩码之后取的，因此返回的正好是这些字母。若用原始对数概率，返回的是整个词表的前 26 个，字母会缺失 |
| `--generation-config vllm` | 否则 vLLM 会把文件夹的 generation_config（top_k、top_p）应用到每个请求（包括读出），top k 之外的字母就得不到概率 |
| `--no-enable-prefix-caching`（0.19.1、Instant） | 在 vLLM 0.19.1 上，前缀缓存会静默改变这些混合模型（Gated DeltaNet 层）的答案：在我们的测试中，它让 2.6% 的普通读出和 16% 的淘汰分组读出变化超过 0.05 |
| `--no-enable-chunked-prefill`（0.19.1、Instant） | 在 0.19.1 上，分块 prefill 也会静默改变长提示的读出，而且 Instant 插件要求每个提示整体完成前缀预填充（prefill） |
| `--max-num-batched-tokens` = `--max-model-len`（0.19.1、Instant） | 整条提示能在一步内装下 |
| `--enable-prefix-caching --enable-chunked-prefill`（0.31 上的 AutoThink 和 v1） | 在 vLLM 0.31 上，前缀缓存只有与分块 prefill 一起才能在这些混合模型上工作。两者都打开时，27B AutoThink 模型在 823 个问题中有 99.88% 与 0.19.1 上答案相同（0.12% 的读出变化超过 0.05），而且一次思考请求不再重读它与首次读出共享的提示 |
| `--max-num-batched-tokens 8192`（0.31 上的 AutoThink 和 v1） | prefill 的分块大小；参考部署所用的设置 |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`（0.31，有 MTP 权重的文件夹） | 多 token 预测：MTP 层每步提出 3 个 token，由模型校验，因此在相同采样分布下思考生成得更快。当文件夹的 config 声明了 `mtp_num_hidden_layers` 且含有 `mtp.*` 张量时，该文件夹就有 MTP 权重；在 4B、9B 和 27B 上，使用 3 个候选 token 时，我们的检查中思考的长度分布保持不变。`--no-mtp` 关闭它。Instant 插件拒绝投机解码 |
| `--limit-mm-per-prompt '{"image":0,"video":0}'`（纯文本） | 不为视觉塔的输入分配显存 |
| `--limit-mm-per-prompt '{"image":8,"video":0}'`、`--mm-processor-kwargs`、`--trust-request-chat-template`（图片） | 每条提示最多 8 张图片（`--max-images`），在本包的边界内缩放（`--max-pixels`）；带图片的调用会携带客户端透传的 chat template（见下文） |
| `--max-model-len` | 在 0.31 上为文件夹的原生上下文，262,144 个 token，Instant 文件夹为 131,072（它们对 196,000 到 256,000 个 token 的提示做一步 prefill 会让 vLLM 0.31 崩溃，见 [vllm_bidirectional.md](vllm_bidirectional.md#known-limits)）；更低的服务（serving）上限（例如在 80 GB GPU 上用 `--max-model-len 65536`）对较短的提示没有代价 |

启动时，vLLM 客户端从 `/v1/models` 读取所服务的模型名和上下文长度，并发送一次覆盖全部 26 个字母的探测读出；缺少读出相关参数的服务端会被拒绝，并给出需要补上的参数。它看不到缓存相关的参数：手动启动的服务端必须使用上面的参数。

**vLLM 上的思考。** 启动时客户端从 `/version` 读取每个服务端的版本。在 vLLM 0.31 或更高版本上，一次思考就是一次流式请求（`thought_generation` 为 `stream`，这是该版本上的默认值）。循环检测器在每新到 1,024 个 token 时查看一次思考，并在预算结束它时再看一次，这些正是分块路径运行检测的位置；陷入循环的思考会像分块路径那样被截断（保留推理部分和循环的开头），其请求被关闭，这在 vLLM 中会结束该请求。使用 MTP 时，流的单个事件最多可携带 4 个 token，因此一次检查可能越过 1,024、2,048 或 3,072 最多 3 个 token。在较旧的 vLLM 上（或 `thought_generation` 为 `chunked` 时），思考按每次 1,024 个 token 生成；每个分块都用各自的 seed 重新发送提示和到此为止的思考，循环检测器在分块之间运行。两者都使用带强制开头的提示、第一个分块的 seed（来自思考提示的哈希；之后每个分块各加 7,919）、温度 0.6、top-p 0.95、top-k 20、停止字符串 `</think>`、预算和 CUT 行。同一个问题的所有调用都发往同一个服务端；问题在各个服务端之间轮询分发。

**波次（Waves，vLLM 0.19.1）。** 在 vLLM 0.19.1 上，当新的生成请求加入一个已在运行的解码批次时，混合模型的状态（state）会被破坏：在我们的测试中，25% 到 35% 的思考变成了乱码循环。因此客户端按波次发送每个服务端的分块思考请求：排队的请求一起发出，只有整波完成后才启动下一波。使用波次后没有思考被破坏（183 个中 0 个）。一个请求最多等待 0.3 秒以便本波凑齐（`--wave-gather`），一波最多 128 个请求（`--wave-max`）。读出只做前缀预填充（prefill），可以随时发送。vLLM 0.31 修复了这种破坏（在 64 个请求并发、不用波次时，183 个思考中 0 个未闭合），因此波次只对早于 0.31 或不报告版本的服务端开启；`--waves` 和 `--no-waves` 可另行决定。在此类服务端上请求 `stream` 会得到 422。

**vLLM 上的图片**（0.31 或更高版本）。提示中含图片的调用走 `/v1/chat/completions` 而非 `/v1/completions`：一条 user 消息，其各个部分是本包渲染后提示的片段，图片（data URI）插在片段之间，并使用一个 chat template，把这些片段紧接着写在一起，并为每张图片写出图像占位符（`<|vision_start|><|image_pad|><|vision_end|>`）。因此 vLLM 读到的正是本包的提示；它的图像处理器在服务端的边界内缩放每张图片（`--mm-processor-kwargs`，启动器将其设为本包的 65,536 到 1,048,576 像素；本包用同一规则计算 token 数），把每张图片的 token 放到其占位符处，并赋予它们多模态位置。服务端需要 `--trust-request-chat-template`（模板随每次调用传入）。当请求带图片时，读出、思考（分块或流式）以及前缀缓存填充都走这条路径。在 Instant 服务端上，插件像处理其他所有提示 token 一样，以双向方式读取图像 token。

**长提示。** 服务端接受最长 `--max-model-len` 的提示（在 vLLM 0.31 上：因果文件夹 262,144 个 token，Instant 文件夹 131,072 个）。更长的提示会被 vLLM 拒绝，请求得到 422 并附带 vLLM 的原因。因果文件夹（AutoThink、v1）以每步 8,192 个 token 把长提示读入前缀缓存（分块 prefill）；当一个请求的各问题共享 4,096 个 token 或更长的开头（即它们的状态）时，本包只读一次该开头（一个单 token 请求），然后把该请求的每个问题发往同一服务端，这样每个问题只读自己的部分，而一次思考不再重读提示，也不再重读到此为止的思考。Instant 文件夹必须一步整体读完每条提示：它们的服务端以 `--max-num-batched-tokens` 等于上下文运行且没有前缀缓存，因此一个含 n 个问题、基于长状态的请求会把该状态读 n 次，而一步必须装下整条提示。

在 vLLM 0.31 上测得，每个服务端一块 H200（一个节点上并排五个服务端），一个答案位于长状态开头几行的决策，且该状态尚未进入任何缓存：

| 模型 | 32,000 tokens | 128,000 | 256,000 | 同一状态上的两个问题 | 再次发送同一请求 |
|---|---|---|---|---|---|
| 4B AutoThink | 0.65 s | 3.8 s | 11.0 s | 0.81 / 4.4 / 12.1 s | 0.14 / 0.45 / 0.96 s |
| 27B AutoThink | | 15.9 s | 42.0 s | 16.4 / 43.3 s | 0.46 / 1.04 s |
| v1 4B | | 4.1 s | | 4.7 s | 0.46 s |
| 4B Instant | 0.78 s | 7.3 s | 崩溃（见上文） | 1.5 / 12.7 s | （无缓存） |
| 27B Instant | | 24.7 s | 在 196,000 时不可用 | | （无缓存） |

每个答案都正确（选择题（choice）的概率为 0.997 或更高）。因果请求的第二个问题的代价是读取它自己的部分；在 256,000 个 token 时，在一个状态已缓存的请求的 0.96 秒中，本包自身的工作（渲染并计算每个问题的提示 token 数）占了大部分。原生上下文下的 KV 缓存（GPU 的 0.88）：4B AutoThink 360 万 token（13.7 条 262,144 的提示），27B AutoThink 109 万（4.2 条提示），v1 4B 311 万；Instant 服务端还要额外为整条提示一步的激活预留空间（4B 在 262,144 时：296 万 token 缓存；27B 在 131,072 时：79 万）。

**单进程。** 决策服务端是一个 Python 进程；提示渲染和 token 计数在其事件循环中运行，状态较短时每个问题约 0.4 ms，8,000 个提示 token 时每个问题约 10 ms（用模型的 tokenizer 在单个 CPU 核上测得）。若要在多块 GPU 上承受高请求率，可运行多个 `serve` 进程，每个进程对应其那一份 vLLM 服务端，放在负载均衡器之后，发布运行时就是这样做的（每块 GPU 一个）。

**vLLM 上的 Instant 文件夹**需要本包中的插件，它让 vLLM 在全注意力层中以双向方式读取提示。它是一个 vLLM 通用插件（入口点 `startlux_prefix_bidir`），当文件夹是提示双向且设置了 `STARTLUX_PREFIX_BIDIR=1` 时生效。启动器会设置该变量，检查 vLLM 的 Python 能看到该插件，并在任一服务端日志未出现 “prefix-bidirectional prompt attention ON” 时拒绝继续。没有插件时，vLLM 会以因果方式读取这样的文件夹，且没有任何提示，回答起来就像另一个模型。vLLM 0.31 默认运行新的 model runner（V2）；插件同时给 V1 runner 和 V2 runner 打补丁，其 runner 未被补丁的 vLLM worker 会在启动时拒绝 Instant 文件夹。细节与测试结果见 [vllm_bidirectional.md](vllm_bidirectional.md)。

## transformers

PyTorch 上的进程内参考实现（`--backend transformers`，extra `[transformers]`，transformers 5.8）。一次前向处理一条提示，一次一个调用：用于校验，不用于服务负载。图片经过文件夹的处理器，使用相同的缩放边界和模型的视觉塔；Instant 文件夹用它训练时的双向提示注意力读取结果（图像特征就位，多模态位置）。Instant 文件夹用它训练时的双向提示注意力读取（`backends/bidirectional.py`，启动时在模型上校验：因果路径等于原生前向，padding 从不触及任何 token，靠后的 token 能到达第一个位置）。思考使用 transformers 的采样器，因此在相同 seed 下与 vLLM 并非逐 token 相同。在 CUDA 上安装 flash-linear-attention 和 causal-conv1d；没有它们，transformers 会在普通 PyTorch 路径上运行线性注意力层，速度慢十倍以上。在没有 GPU 的机器上不能安装这两个包（它们会在 CPU 张量上被调用）。

## llama.cpp

`--backend llamacpp --llamacpp-url URL` 把本包放在 llama-server 前面，用于 GGUF 构建；模型文件夹仍提供 tokenizer 和 `decision_config.json`。一次读出以 `n_predict` 1、`post_sampling_probs`、对 k 个选项字母施加 +100 的 logit bias 以及 `top_k` = k 发送 token id，因此采样器只保留这些字母，它们的概率就是仅在这些字母上做的 softmax（对每个字母施加相同的 bias 不改变它们的相对概率）；其他所有采样器都保持中性。思考使用 vLLM 的采样设置及其采样器顺序（先温度，再 top-k 和 top-p；llama.cpp 的默认是先应用 top-p）。提示缓存关闭。用 llama.cpp b11549 测试过。

**llama.cpp 上的图片。** 用模型的 mmproj 文件（`--mmproj mmproj-...-F16.gguf`，即与 GGUF 在同一仓库中的 F16 文件）和 `--image-min-tokens 64 --image-max-tokens 1024`（本包的像素边界，以 token 计）启动 llama-server。此时 `/props` 会报告视觉模态和服务端的媒体标记，后端（backend）便接受图片（`/health` 中的 `engine.images` 为 true）。带图片的调用发送 `"prompt": {"prompt_string": ..., "multimodal_data": [...]}`：以文本形式给出渲染后的提示，每张图片处放媒体标记，图片则以 base64 PNG 给出。llama.cpp 自己对文本做 tokenize，因为对象形式的提示无法携带 token id（在冒烟测试的每个图片请求上，它的计数都与 Hugging Face tokenizer 相等），并在每张图片前后加上 `<|vision_start|>` 和 `<|vision_end|>`。后端已经用 PIL（双三次）把每张图片缩放到处理器尺寸，因此服务端自己的缩放不起作用，像素就是 vLLM 读到的那份。每次图片调用后，它都会把服务端的 `tokens_evaluated` 与自己算出的计数（文本 token，加上 2，再加上每张图片的 token）比较（`/tokenize` 可区分文本的差异和图片的差异），不一致就抛错，例如服务端以更小的 `--image-max-tokens` 启动并缩放了一张图片。没有 `--mmproj` 时，带图片的请求得到 422。纯文本调用保持原样。

**Instant 文件夹**需要全注意力层对提示做非因果注意力。已发布的 Instant GGUF 带有 `qwen35.attention.causal = false`（由 `patches/llama.cpp/0002-...` 写入），并且提示必须单独、在一个微批次中求值。`STARTLUX_LLAMACPP_BIDIR` 指明服务端如何提供这一点：

| 取值 | 服务端 | 后端做什么 |
|---|---|---|
| `auto`（默认） | 任意 | 启动时读一次 `/props`：报告 patch 0003（`mtmd_prompt_whole_when_non_causal`）的服务端按 `server` 工作，其他按 `request` 工作 |
| `request` | 带 `patches/llama.cpp/0001-...` 的 llama-server，任意 `--parallel` | 每次读出都发送 `prompt_bidirectional` 并校验回显；原生未打补丁的服务端会报错 |
| `server` | 原生 llama-server，带 `--parallel 1 --ctx-checkpoints 0 -b 16384 -ub 16384` | 依赖文件的非因果上下文 |
| `off` | 任意 | 以因果方式读取提示：这是另一个模型，仅用于测量该代价 |

原生 llama-server 不会报告任何关于其上下文的信息，因此 `auto` 会把它当作 `request` 服务端，Instant 文件夹的首次读出会失败，并在消息中点名 `server`：为它设置 `STARTLUX_LLAMACPP_BIDIR=server`。正在使用的模式是 `/health` 中的 `engine.bidir_mode`。

**带图片的 Instant 文件夹。** 一条提示的文本和图像嵌入必须在一个微批次中以非因果注意力读取。使用模式 `server`（对带 patch 0003 的服务端 `auto` 会选它；其他情况请设置 `STARTLUX_LLAMACPP_BIDIR=server`），配合非因果 GGUF 和 `--parallel 1 --ctx-checkpoints 0 -b N -ub N`，N 至少为最长提示（含其图像 token）的长度。按请求的模式（`request`，patch 0001）不覆盖图片，会被 422 拒绝。原生 b11549 会把这样的提示拆成三个微批次（文本、图像、文本）：于是在 32 个图片问题上，4B Instant 的 Q8_0 读出与 BF16 平均相差 0.022（最大 0.22），而使用 `patches/llama.cpp/0003-server-mtmd-prompt-whole-on-non-causal-context.patch` 后相差 0.0055（最大 0.06）。打过补丁的服务端会在 `/props`（`mtmd_prompt_whole_when_non_causal`）中说明这一点；当带图片的 Instant 文件夹在没有该补丁的情况下被服务时，后端会给出警告，并在 `/health` 中报告 `image_prompt: split (stock server)`。

在一块 H200 上对照 vLLM BF16 测得（823 个问题）：BF16 GGUF 的 top-1 一致率为 99.5 到 100%（两次 vLLM 运行：99.9%），Q8_0 为 99.4 到 99.6%，Q4_K_M 为 97.0 到 99.2%。在 DecideBench 上，4B Instant 模型作为 Q8_0 损失 0.6 分，Q4_K_M 损失 3.1 分，使用默认设置的原生服务端损失 7.7 分，提示以因果方式读取时损失 26 分（[benchmarks.md](benchmarks.md)）。llama.cpp 回答短的 Instant 请求比 vLLM 快，长的请求最多慢三倍，因为它把请求中的双向提示逐条求值。

## MLX

`--backend mlx`（extra `[mlx]`，Apple Silicon）在 mlx-vlm 中运行文件夹自带的解码器，并自行构建输入。它可读取 mlx-vlm 文件夹（`mlx_vlm.convert`：文本模型 8 位或 4 位，视觉塔保持 BF16）、BF16 发布文件夹，以及纯文本 MLX 文件夹（此时图片会得到 422）。Instant 文件夹用它训练时的双向提示注意力读取（全注意力层能看到单次输入中的每个 token，线性注意力层保持因果；加载时的自检会确认这一点）。AutoThink 的思考用请求的设置和 seed 采样。图片沿用 vLLM 的流程：文件夹的图像处理器、视觉塔在图像 token 处的特征、处理器的三轴位置。一次只运行一个调用。

## 与发布运行的一致性

黄金测试（`tests/test_golden.py`）在假 vLLM 上并行运行本包的管线（pipeline）与已发布 AutoThink 数字背后的那些运行的逐问题管线，覆盖随机请求和设置：发给 vLLM 的请求（每个读出提示、思考分块、seed 和采样设置）以及概率、选择题（choice）、量表题（score）和思考 token 数都逐比特相同。Instant 读出就是那些运行的单次读出（同一条路径，关闭思考）。有意的差异如下，其中没有一项会改变已作答问题的概率：

| | 发布运行 | 本包 |
|---|---|---|
| 选择题和量表题答案的 `confidence` | 最高概率 | TypeSafe 的定义（见上文） |
| 当判定标准为 `{key: level}` 时量表题的 `legend` | 按给定顺序排列的等级 | 按展示给模型的顺序排列的等级（数字键升序） |
| 只有一个选项的选择题 | 拒绝（422） | 以概率 1 作答 |
| `mix` 0 | 意为“仅思考之后的读出” | 意为“仅首次读出”（公式） |
| 淘汰分组（knock-out）列表的 `usage.input_tokens` | 最后一轮的提示 | 每一轮的提示 |
| 请求控制项 | 只有 `think_mode`，未知取值被忽略 | `think_mode`、`gate`、`think_budget`、`mix`、`thought_generation` 及详情；错误取值给出 422 |
| vLLM 0.31 上的思考 | （未运行） | 一次流式请求（按需走分块路径）；黄金测试跑分块路径，而 `tests/test_stream.py` 表明，在假 vLLM 上每个流事件一个 token 时，流式路径给出的思考与概率和分块路径相同 |
| 格式错误的请求体、空的 `questions` | 400 | 422，FastAPI 风格的 `detail` |
| `cache_mode`、`--rep-pen`、`--stop-loops`、`--loop-version` 1 和 2 | 测试开关 | 未沿用（发布版没有使用其中任何一个） |
| 后端错误 | 拒绝时为 422，否则为无结构的 500 | 拒绝时为 422，重试四次后为 502 |
