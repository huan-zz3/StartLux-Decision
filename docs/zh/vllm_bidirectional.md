> 译自 [../vllm_bidirectional.md](../vllm_bidirectional.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# vLLM 中的提示（prompt）双向注意力

Instant 目录在训练时，是在全注意力层中用双向注意力读取其提示（prompt）的。原版 vLLM 以因果方式读取每个
提示（prompt），因此会把这样的目录当成另一个模型来跑，且不报任何错误。`startlux_decide/vllm_plugin.py` 中的
插件让 vLLM（V1 引擎）按这些目录训练时的方式读取它们。它已在 vLLM 0.19.1 和 0.31.0 上验证（见
[vLLM 版本](#vllm-版本)）。已安装的 vLLM 中没有任何内容被修改：插件在 vLLM 导入这些类时对它们打补丁。

## 掩码

在全注意力层中（每四层有一层），在单个请求内部：

    query q may read key k  <=>  k <= q  or  (q < p and k < p)

Gated DeltaNet 层、其中的短卷积以及 KV 缓存写入，与原始 vLLM 完全一致。

一个请求的提示（prompt）结束位置 p 来自它的 prompt token id：

| 提示（prompt） | p |
|---|---|
| 以空的 think 块 `<think>\n\n</think>\n\n` 结尾（单遍读出） | L：整个输入都是双向的 |
| 其他情况 | 紧跟在第一个 `<\|im_start\|>assistant\n` 之后的 `<think>\n` 后面（系统轮、证据、问题、选项、assistant 头部）。强制前缀、思考（thought）、`</think>` 以及其后的读出都是因果的 |
| 在 `SamplingParams(extra_args={"startlux_prompt_end": p})`，或 OpenAI 请求中的 `"vllm_xargs": {"startlux_prompt_end": p}` | 该 p（0 = 因果，L = 全部双向） |
| 以上都不匹配 | 报错（或 `STARTLUX_PREFIX_BIDIR_UNMATCHED=causal` / `full`） |

在每个模型 570 条测试提示（prompt）上，该规则给出的 p 与 transformers 参考实现相同。思考（thought）内部
出现一个游离的 `<think>` 不会改变 p。

## 工作原理

1. 每一步，在模型运行器构建注意力元数据的位置有一个钩子，它找出本步中提示（prompt）被前缀预填充（prefill）
   的请求：vLLM 模型运行器 V1 中是 `GPUModelRunner._build_attention_metadata`，模型运行器 V2（0.31 的默认值）
   中是模型状态的 `prepare_attn`。它检查每个这样的提示都从位置 0 开始并被整体调度，否则抛错。它把一个小计划
   附加到 FLASH_ATTN 元数据上：批次行 `[start, start + p)` 和提示长度。
2. `FlashAttentionImpl.forward` 先执行原版的因果 FlashAttention 调用。然后只针对计划中的行，对每个请求自身的
   提示 q/k/v 执行一次非因果的 varlen FlashAttention 调用（即 vLLM 已经使用的核（kernel））。k/v 是原版代码
   写入 KV 缓存的那些张量。结果覆盖这些行。
3. 位于 p 或之后的行保留原版的因果结果，这正好就是 q >= p 时的掩码。
4. 后续步骤不需要任何处理。缓存中的提示 K/V 是由双向提示计算出来的，而之后的每个查询都晚于每个提示键。解码
   步骤不受影响：解码批次的完整 CUDA 图与原始 vLLM 相同。带前缀预填充（prefill）的步骤在分段图之间以 eager
   方式运行注意力，修正就在那里执行。
5. 被抢占的请求会连同其已生成的 token 从位置 0 重新前缀预填充（prefill），在一个步骤内完成。同样的计划适用。

额外开销是对提示行做的因果遍历，而它的结果会被丢弃：在我们的测量中增加 2.2% 的实际耗时（wall time）。

## 启用

在运行 vLLM 的 Python 环境中安装本包（它声明了 vLLM 通用插件 `startlux_prefix_bidir`），并设置
`STARTLUX_PREFIX_BIDIR=1`。没有该变量时，插件完全不做任何事。`startlux-decide launch-vllm` 会完成这两步并
检查结果。双向目录所需的引擎设置（插件会在引擎启动时检查，不满足就拒绝运行）：

```bash
vllm serve DIR --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 65536 --max-model-len 65536 ...
```

`--max-num-batched-tokens` 必须不小于 `--max-model-len`，以便每个提示（prompt）都在一个步骤内被整体前缀预填充
（prefill）。注意力后端（backend）必须是 FLASH_ATTN（这里是 vLLM 的默认值）。

vLLM 会在引擎核心进程以及每个 worker 构建之前调用通用插件。插件只注册导入钩子，因此非 vLLM 进程和 API
服务器都不受影响。如果设置了 `VLLM_PLUGINS`，它必须列出 `startlux_prefix_bidir`。进程内替代方案：
`VLLM_ENABLE_V1_MULTIPROCESSING=0`，然后
`from startlux_decide import vllm_plugin; vllm_plugin.install()`。

**检查引擎是否启用了它。** 服务端日志会打印 `prefix-bidirectional prompt attention ON for DIR` 以及前三个请求
的 p；没有这一行时 `launch-vllm` 会拒绝继续。对于进程内的 `vllm.LLM`，除非每个 worker 都报告插件处于活动
状态，否则 `vllm_plugin.require_active(llm, DIR)` 会抛错（`llm.collective_rpc("startlux_prefix_status")`；
原始 vLLM 没有这个方法）。

**关闭时。** 未设置 `STARTLUX_PREFIX_BIDIR` 时不会打任何补丁。设置了该变量但面对因果目录时，打过补丁的方法
只调用原版代码。两种情况下输出都与原始 vLLM 逐位相同（见下文测试 d）。

**调试。** `STARTLUX_PREFIX_BIDIR_CHECK=1` 会以 fp32 和显式掩码重新计算每次层调用中最多
`STARTLUX_PREFIX_BIDIR_CHECK_REQS`（2）个请求的注意力。如果修正结果与其参考的距离超过原版因果核（kernel）
与其自身参考距离的两倍（或在最大输出处超过一个 bf16 ulp），它就会抛错。

插件早期内部副本中的变量名和额外参数键会作为别名被接受。

## vLLM 版本

| vLLM | 模型运行器 | 插件在哪里找到某一步的提示（prompt） |
|---|---|---|
| 0.19.1（发布版本所用） | V1，`vllm.v1.worker.gpu_model_runner` | `GPUModelRunner._build_attention_metadata`，从常驻输入批次和缓存的请求状态中获取 |
| 0.31.0 | V2，`vllm.v1.worker.gpu.model_runner`（默认） | 模型状态的 `prepare_attn`，从该步的 `InputBatch` 获取；提示 id 和采样参数来自调度器输出中的新请求 |
| 0.31.0 且 `VLLM_USE_V2_MODEL_RUNNER=0` | V1 | 与 0.19.1 相同 |

修正本身（`FlashAttentionImpl.forward`，一次非因果的 varlen FlashAttention 调用）在两个版本上是同一份代码。
两者都经过验证（[结果](#vllm-0310)）。其他版本会被拒绝，除非 `STARTLUX_PREFIX_BIDIR_ALLOW_VERSION=1`。

**为什么 0.31 需要移植。** vLLM 0.31 默认构建模型运行器 V2（其日志会显示 "Using V2 Model Runner"）。为
0.19.1 写的插件在豁免版本检查后于该环境运行，却只给 V1 类打了补丁：它记录了 "installed"，但从未记录
"prefix-bidirectional prompt attention ON"，于是 vLLM 以因果方式读取 Instant 4B 且没有报错。与带插件的
0.19.1 相比，它在 823 个问题中的 95.1% 上选择了同一选项，并使 23.7% 的读出结果变化超过 0.05。

**不认识运行器的会被拒绝。** worker 构建好模型运行器之后，插件会检查双向目录得到的是否是它打过补丁的运行器；
否则引擎会在启动时停止，并给出运行器的类名。这也会拒绝 0.19.1 的实验性运行器 V2
（`VLLM_USE_V2_MODEL_RUNNER=1`）。

**0.31 上的核（kernel）预热。** 在开始服务之前，vLLM 0.31 会通过真实模型路径发送合成请求来预热核（提示 id 为
0、1、2、……）。它们不匹配任何提示结束规则。预热运行期间插件不制定计划，因此这些请求按因果方式读取；它们的
输出会被丢弃。状态计数器 `warmup_steps` 统计这类步骤。

**0.31 上的引擎设置。** 相关参数与 0.19.1 相同。在 0.31 中，这些混合模型上的前缀缓存只有配合分块前缀预填充
（chunked prefill）才能工作；插件和以前一样需要两者都关闭。带提示前缀预填充（prefill）的步骤不能重放完整
CUDA 图（FULL CUDA graph）（修正运行在 eager 注意力中）；在默认的 `FULL_AND_PIECEWISE` 下这类步骤以分段方式
运行，而一旦某步被派发到 FULL 图，插件就会抛错。在 0.31 上，即使设置 `STARTLUX_PREFIX_BIDIR_ALLOW_TRITON=1`，
TRITON_ATTN 也会被拒绝：0.31 在构建元数据时会将其 PrefixLM 范围转成一个张量，因此计划无法触及它们。

## 测试结果

### vLLM 0.19.1

测量使用的是该插件移入本包之前的版本（逻辑相同，只改了名字），bf16，每次运行一块 H200。模型：一个提示双向的
思考测试模型（B），一个单遍 Instant 4B 候选模型（I），以及用于噪声基线的因果 4B（C）。transformers 参考实现
是训练代码中的双向前向。token id 只渲染一次，因此两侧读取的 id 完全相同。Δp 是每个条目中某个选项概率的最大
变化。

**噪声基线。** 对因果模型 C 做同样的比较，原始 vLLM 对 transformers：单遍平均 |Δp| 0.0061，最大 0.040，
top-1 一致率 99.0%；带思考（thought）之后的读出：平均 0.0055，top-1 99.0%；贪心 128 token：50% 相同。

| 测试 | 带插件的 vLLM 对 transformers 参考 | 原始 vLLM 对 transformers 参考 |
|---|---|---|
| (a) 单遍读出，模型 B，300 个条目 | 平均 \|Δp\| 0.0065，最大 0.048，top-1 99.3% | 0.097，最大 0.74，top-1 88.3% |
| (a) 单遍读出，模型 I，300 个条目 | 平均 0.0053，最大 0.031，top-1 99.7% | 0.097，最大 0.75，top-1 88.3% |
| (c) 存储的思考（thought）之后的读出，200 条思考 | 平均 0.0060，最大 0.052，top-1 99.5% | 0.067，最大 0.57，top-1 95.0% |
| (b) 贪心思考（thought），128 token，50 条提示（prompt） | 62% 相同；平均在 94 个 token 后首次出现差异；首个 token 的 top-1 98%，平均 \|Δlogprob\| 0.014 | 2% 相同；22 个 token |

对于 (b)：让 transformers 参考实现与自身比较，提示（prompt）逐条解码而非批处理，得到 74% 相同、首个 token
的平均 |Δlogprob| 为 0.014，因此 vLLM 与参考实现之间的差异属于批处理噪声。

| 更多检查 | 结果 |
|---|---|
| (d) 插件关闭，逐位相同 | 原始版 vs 未设置变量即加载插件（模型 B）：相同。原始版 vs 在因果目录上启用插件（模型 C）：相同。每项覆盖 570 个用例（12,315 个浮点数以及每个 token），各跑两次，调度不依赖时序（进程内引擎，无异步调度；使用异步调度时原始 vLLM 自身都不可复现） |
| 检测出的 p vs 显式的 p | 离线逐位相同；经 HTTP 时在批处理噪声范围内相同 |
| fp32 检查（570 个用例，每次层调用 4 个请求，1,000 次检查） | 修正误差最多为原版因果核（kernel）在相同行上自身误差的 2.03 倍（输出约 37 时最大 0.148 对 0.170：bf16 舍入所致）。双向部分对提示行的改变至少为 4.4 |
| 掩码探针（提示对数概率） | 改变最后一个用户轮次的 token，带插件时会使更早位置的对数概率变化 0.67 到 1.86，不带插件时为 0。改变某个思考（thought）的 token 不会影响它之前的位置（恰好为 0），但会影响其后位置至少 2.4 |
| 抢占（小 KV 缓存，6 个请求被重新前缀预填充（prefill）） | 所有 248 次受检查的修正，包括重新前缀预填充，都通过 fp32 检查（比值 ≤ 1.21） |
| 带插件的 `vllm serve` | 单遍平均 \|Δp\| 0.0066，top-1 99.3%；思考（thought）之后的读出 0.0059，99.5%；serve 与离线在批处理噪声范围内相同 |
| 留出集运行，3,021 个条目（模型 B，采样思考（thought），预算 4,096） | 单遍读出：平均 \|Δp\| 0.0069，top-1 99.0%；准确率 72.49%（参考）vs 72.46%（vLLM）。思考之后：74.81% vs 74.94%，差 +0.13 个点，95% CI -0.56 到 +0.86。Brier 0.359 vs 0.357。思考闭合（closed）17.7% vs 18.1%，3,538 vs 3,539 个 token。vLLM 上 GPU 时间低 4.3 倍 |
| 吞吐，一块 H200（思考（thought）生成，预算 4,096，温度 0.6 / top-p 0.95 / top-k 20，前后各做读出） | 每小时 12,728 条思考，含读出（仅生成则 15,744 条），9,920 token/s；transformers 参考运行器约每小时 3,500 条思考：3.6 倍（仅生成 4.5 倍） |
| 额外开销（1,024 × 512 个强制 token，相同引擎设置） | 带插件 54.6 s，原始版 53.4 s（+2.2%） |

### vLLM 0.31.0

测量于 2026 年 10 月 11 日，使用本包，bf16，每个引擎一块 H200，两个版本都用 FlashAttention 3。Instant 发布
目录（除非注明，均为 4B）。离线测试在单进程中运行 `vllm.LLM`，不使用异步调度，使用来自延迟请求的 300 条单遍
提示（prompt）（选项数 2 到 26 的问题）、40 条在写出一段简短思考（thought）之后读取的提示、20 条带显式 p 的
单遍提示，以及 32 个 token 的贪心续写。Δp 是每个问题中某个选项概率的最大变化。

| 测试 | 结果 |
|---|---|
| 0.19.1，移植前的插件与当前插件对比，同一块 GPU 先后运行 | 逐位相同：全部 360 次读出、全部 40 次续写、相同的计划计数器。任一插件的两次运行也逐位相同。（最初的几次运行中，两个引擎在空缓存上同时编译各自的核（kernel），差异处于批处理噪声水平：300 次单遍读出中 146 次逐位相同，平均 \|Δp\| 0.0014，全部首选同一选项；缓存填满后每次重复都逐位相同） |
| 0.31 上的 fp32 检查，每次层调用 4 个请求 | 运行器 V2：216 次检查，修正误差最多为原版核（kernel）在相同行上自身误差的 1.52 倍。运行器 V1：232 次检查，最多 1.52 倍。双向部分对提示行的改变至少为 4.6 |
| 0.31 上的抢占（36 个 528 token 的 KV 块，16 个序列，续写强制为 512 个 token），开启 fp32 检查 | 每个运行器上各有 6 个请求连同其已生成 token 被重新前缀预填充（prefill）；全部 976 次检查通过（V2 上最多 1.75 倍，V1 上 1.49 倍） |
| 0.31 上的显式 p | p = L 逐位得到检测出的单遍读出（10/10）；p = 0 按因果方式读取（平均 \|Δp\| 0.035） |
| 0.31 上的因果目录（AutoThink 4B），插件开启与关闭 | 逐位相同（240 次读出，20 次续写） |
| 0.31 上运行器 V2 对运行器 V1 | 340 次读出全部首选同一选项，其中 329 次逐位相同；续写相同 |
| 0.19.1 对 0.31（运行器 V2），离线 | 340 次读出全部首选同一选项；单遍平均 \|Δp\| 0.0015，最大 0.025，没有超过 0.05 的；40 次续写中 38 次相同 |
| 未知运行器 | 0.19.1 设置 `VLLM_USE_V2_MODEL_RUNNER=1`：引擎在启动时拒绝该 Instant 目录 |

通过服务端（serving）：在带发布参数的 `vllm serve` 前面放 `startlux-decide serve`，300 条延迟请求（823 个
问题）逐条发送，`think_mode` 从不启用。参考：带移植前插件的 vLLM 0.19.1。

| 引擎 | Instant 4B：同一选项、平均 \|Δp\|、超过 0.05 的占比 | Instant 27B |
|---|---|---|
| 0.31.0 + 本插件，运行器 V2 | 99.88%（一个问题），0.0017，0.12% | 100%，0.0019，0.12% |
| 0.31.0 + 本插件，运行器 V1 | 99.88%，0.0017，0.12% | 100%，0.0019，0.12% |
| 0.19.1 + 本插件 | 100%，0.00001，0% | 100%，0.00006，0% |
| 0.31.0 + 移植前的插件（豁免版本检查；按因果读取） | 95.1%，0.050，23.7% | 未运行 |
| 版本之间的基线：因果 27B AutoThink，0.31.0 对 0.19.1，相同参数 | | 100%，0.0023，0.24% |

单遍读出的延迟：通过发布客户端逐条发送 300 条延迟请求，第 2 遍，按 Decision Index 的提示长度构成加权
（短 90.2%，中 9.0%，长 0.79%）；单位为每请求的毫秒，取中位数 / 平均值 / p80；每个服务端一块 H200，所有列
使用同一节点和同一批目录。

| 规模 | 0.19.1 + 插件 | 0.31.0 + 插件 | 0.31.0 不带插件（提示按因果读取，仅计时） |
|---|---|---|---|
| 0.8B | 31 / 45 / 46 | 33 / 46 / 44 | 33 / 45 / 45 |
| 2B | 31 / 48 / 47 | 31 / 46 / 45 | 32 / 46 / 44 |
| 4B | 36 / 60 / 55（重复 36 / 60 / 53） | 39 / 62 / 53（重复 43 / 64 / 56） | 38 / 59 / 51 |
| 9B | 34 / 68 / 53（重复 35 / 79 / 55） | 40 / 82 / 62（重复 40 / 73 / 57） | 39 / 70 / 57 |
| 27B | 60 / 158 / 120 | 62 / 164 / 120 | 62 / 160 / 119 |

vLLM 0.31 对短请求的应答比 0.19.1 晚 1 到 7 ms，对最长请求则早 20 到 31 ms（长请求段中位数：4B 从 541 降到
514 ms，27B 从 2,602 降到 2,582 ms）。不带插件时也是如此，所以这是引擎本身的行为。插件自身在 0.31 上的开销
是：短请求 0 到 4 ms，最长提示上为 4 到 9%（对提示多做的一次非因果注意力）。单个短请求若卡顿一秒，会使加权
平均值变动约 9 ms（两个版本都能观察到），因此应比较中位数和 p80。

## 已知限制

- **仅支持 FLASH_ATTN。** 曾用 vLLM 自带的 PrefixLM 范围实现 TRITON_ATTN 作为交叉检查。其读出与参考一致，但在
  0.19.1 中，带 `USE_MM_PREFIX` 的 `triton_unified_attention` 会把一次启动中的每个序列都放宽到
  `max(causal bound, seq_len)` 个键：与这类前缀预填充（prefill）共享同一次启动的解码行会读到自身末尾之后，
  生成随即崩成 `!!!!`。除非设置 `STARTLUX_PREFIX_BIDIR_ALLOW_TRITON=1`（仅做读出的任务，且仅限 0.19.1；
  在 0.31 上一律拒绝），否则插件会拒绝 TRITON_ATTN。
- **在 H200 上验证**（两个版本都是 FlashAttention 3）。在 Blackwell 上，vLLM 0.31 会选择 FlashAttention 4；
  在那里信任它之前，请先跑 fp32 检查（`STARTLUX_PREFIX_BIDIR_CHECK=1`）。
- **必需的引擎设置。** 关闭分块前缀预填充（chunked prefill）和前缀缓存，
  `max_num_batched_tokens >= max_model_len`。较长的前缀预填充会让同一步的解码停滞。
- **图像**（vLLM 0.31）。图像 token 与其他 token 一样是提示（prompt）token：修正会把它们的行与提示的其余部分
  一起重新计算。在 12 个图像请求下，4B 和 27B Instant 目录的读出分别有 12/12 和 11/12 个问题与 transformers
  参考首选同一选项，最大概率差为 0.037 和 0.072；不使用插件的因果 AutoThink 目录差异同样大（0.073 和
  0.058），而在没有图像时，同样的问题最多相差 0.007（4B Instant）和 0.006（27B Instant）。两个引擎的图像
  处理器给出完全相同的像素，因此差异来自视觉通路的两种实现，而不是插件。在 vLLM 对图像编码器做 profile 时
  （一步能容纳的满尺寸图像数量，在 262,144 个 token 时为 256），4B 服务端无法启动（profile 运行中出现非法
  内存访问）；因此启动器会给带图像的 Instant 服务端传入 `--skip-mm-profiling`。
- **长提示。** 一个提示（prompt）会在一个步骤内被整体前缀预填充（prefill），因此 `max_num_batched_tokens`
  必须能容纳最长的提示，并且 vLLM 会为这样的步骤预留激活。对于提示行数达到 16,384 或更多的步骤，修正会就地
  执行（不复制 query、key、value 或输出；与在 32K 提示上使用聚集调用的概率相同），因为 vLLM 的内存 profile
  是在没有修正的情况下运行的。在 vLLM 0.31 和一块 H200 上，128,000 token 的提示在 4B 和 27B 上都能运行
  （7.3 s 和 24.7 s），但 196,000 token 提示（27B）和 256,000 token 提示（4B）的那一步会让服务端崩溃（非法
  内存访问；4B 还发生了内存不足），而 27B 在 262,144 token 时无法启动（vLLM 自身对这类步骤的 profile 会内存
  不足）。因此 `launch-vllm` 在 0.31 上给 Instant 服务端 131,072 个 token；因果目录会分块读取长提示，可以
  达到 262,144。
- **不支持：** 投机解码（speculative decoding）、解码上下文并行、ubatching（DBO）、FP8 KV 缓存、提示嵌入
  （prompt embeddings）；在 vLLM 0.19.1 上也不支持多模态输入（仅文本：`--limit-mm-per-prompt` 0）。
- **用户文本中出现字面头部。** 如果证据中包含 `<|im_start|>assistant\n<think>\n`，思考型提示的检测会把提示
  在那里结束。此类管线（pipeline）中请显式传入 p。单遍提示不受影响（它们的规则看的是提示末尾）。
- **空思考（thought）。** 没有强制前缀时（`template + "" + "\n</think>\n\n"`），空思考之后的读出与单遍提示是
  相同的文本，因此会完全按双向方式读取。
- **插件未加载。** 设置了 `STARTLUX_PREFIX_BIDIR=1` 但 vLLM 环境中没有本包时，得到的是原始 vLLM，它会按因果
  方式读取该目录且不作任何提示。请使用 `launch-vllm`、`require_active` 或检查日志。
- **版本。** 与 vLLM 0.19.1 和 0.31.0 的内部实现绑定：`FlashAttentionImpl.forward`；对运行器 V1 是
  `GPUModelRunner._build_attention_metadata` 和缓存的请求状态；对运行器 V2 是 `GPUModelRunner.add_requests`、
  `_remove_request`、`load_model`、`execute_model`、模型状态的 `prepare_attn` 以及核（kernel）预热函数。
  其他版本会被拒绝；`STARTLUX_PREFIX_BIDIR_ALLOW_VERSION=1` 可在重新验证后覆盖（跑 fp32 检查，并与已验证
  版本比较读出）。
- **运行器 V2 上的状态计数器。** 被抢占后恢复的请求会作为新请求回到运行器 V2，因此它的提示结束位置会被再次
  找到，并在 `bi_requests` 和提示类型计数器中计两次（仅用于诊断）。
- **提示对数概率。** 对于双向提示，这些不是语言模型概率：某个位置能看到其后的提示 token。
