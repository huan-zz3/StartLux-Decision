> 译自 [../quantization.md](../quantization.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# 量化

`startlux-decide quantize fp8|gguf|mlx` 会打印下面的步骤（也可以用 `--run` 执行它们）。它会检查所需工具是否存在，不下载任何东西。

## vLLM 中的 FP8

无需单独的构建：vLLM 在加载文件夹时把权重（weights）量化为 FP8。

```bash
startlux-decide launch-vllm --model /models/StartLux-Decision-4B --gpus 0 --quantization fp8
```

这会给每台服务器加上 `--quantization fp8`。KV cache 保持 BF16（Instant 插件拒绝 FP8 的 KV cache）。

## 用于 llama.cpp 的 GGUF

适用于 AutoThink 文件夹；如果 llama.cpp 代码树里有转换器补丁 `patches/llama.cpp/0002-convert-qwen35-prompt-bidirectional.patch`（它把 GGUF 标记为非因果），也适用于 Instant 文件夹（该命令会检查它）。若要支持图片，先用 `convert_hf_to_gguf.py MODEL --mmproj --outtype f16` 转换一次视觉编码器，再用 `--mmproj` 把该文件传给 llama-server。使用带有 `llama-quantize` 可执行文件的 llama.cpp 检出：

```bash
startlux-decide quantize gguf --model /models/StartLux-Decision-4B --out /models/4B-gguf --llama-cpp ~/llama.cpp --type Q8_0 --run
llama-server -m /models/4B-gguf/model-Q8_0.gguf -ngl 99 -c 65536 --port 8081
startlux-decide serve --model /models/4B-gguf --backend llamacpp --llamacpp-url http://127.0.0.1:8081
```

第一步用当前的 Python 运行 `convert_hf_to_gguf.py --outtype bf16`（这需要 llama.cpp 的转换依赖），然后运行 `llama-quantize`（除非指定 `--type BF16`），并把 tokenizer 文件、`config.json` 和 `decision_config.json` 复制到 GGUF 文件旁边：本包自己渲染提示并读取字母。llama.cpp 必须足够新，能识别该模型的架构。使用 `--parallel N` 时，llama-server 把 `-c` 分摊到 N 个槽位；本包从 `/props` 读取每个槽位的上下文，用于思考的空间检查。

`quantize gguf` 只处理文本解码器。图片需要模型的视觉编码器单独放在一个 mmproj 文件里，已发布的 GGUF 仓库以 F16 文件形式提供（`mmproj-*-F16.gguf`）；见 [engines.md](./engines.md#llamacpp)。

## MLX

适用于 Apple Silicon 上的 Instant 与 AutoThink 文件夹。文本模型会被量化（仿射（affine），组大小 64），视觉塔保持 BF16，因此文件夹仍能读取图片：

```bash
startlux-decide quantize mlx --model /models/StartLux-Decision-4B --out /models/4B-mlx-8bit --bits 8 --run
startlux-decide serve --model /models/4B-mlx-8bit --backend mlx
```

## Instant 文件夹

Instant（提示双向）文件夹的 GGUF 构建必须用它在训练时使用的双向提示来读：用 `patches/llama.cpp/0002-...` 转换它（已发布的文件已经带有该标记），并按 [engines.md](./engines.md#llamacpp) 所述提供服务。按因果方式读时，该构建会像一个不同的模型那样作答。MLX 引擎以双向方式读取 Instant 文件夹，所以它们的 MLX 构建没有问题。

## 每个构建都要重新测量

量化后的构建是一个不同的模型。决策模型在这件事上比聊天模型更敏感，原因有两方面：

* 答案本身就是概率分布，其校准（calibration）（ECE、Brier）即使在所选选项不变时也可能变化；
* 当第一次读出的最高概率低于 0.9 时，AutoThink 会进行思考。该概率变化几个百分之一就会改变哪些问题会思考，而在一个构建中思考、在另一个构建中不思考的问题，可能以不同的答案、延迟和 token 数收场。

在同一批请求上，把每个构建与 BF16 对比：

```bash
startlux-decide eval --url http://127.0.0.1:8090 --data gold.jsonl --out bf16.jsonl     # BF16 server
startlux-decide eval --url http://127.0.0.1:8091 --data gold.jsonl --out fp8.jsonl      # the build
startlux-decide eval --compare bf16.jsonl fp8.jsonl
```

`--compare` 报告所选答案的一致率、两个构建各自的准确率及其差值（对请求做 95% bootstrap 区间）、门控决策（是否思考）的一致率，以及每个问题上任一选项概率的最大变化。

| 模型 | 构建 | 引擎 | 与 BF16 的一致率 | 准确率 BF16 → 构建（差值的 95% CI） | 思考决策一致 | max \|Δp\|（均值） |
|---|---|---|---|---|---|---|
| 4B Instant | FP8 | vLLM | TODO | TODO | n/a | TODO |
| 4B AutoThink | FP8 | vLLM | TODO | TODO | TODO | TODO |
| 4B AutoThink | GGUF Q8_0 | llama.cpp | TODO | TODO | TODO | TODO |
| 4B AutoThink | GGUF Q4_K_M | llama.cpp | TODO | TODO | TODO | TODO |
| 4B AutoThink | MLX 8-bit | mlx-vlm | TODO | TODO | TODO | TODO |
