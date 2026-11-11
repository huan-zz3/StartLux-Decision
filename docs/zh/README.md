> 译自 [../README.md](../README.md)（commit 9ba65e2），另有新增的中文主线指南。两者不一致时，以英文原版为准。

# 文档

| 文档 | 什么时候读它 |
|---|---|
| [finetune_quickstart.md](finetune_quickstart.md) | 第一次用自己的数据微调：从自检、跑通、做数据、测基线，到训练与前后对比（本文为中文原创，无英文对应） |
| [finetune_data.md](finetune_data.md) | 准备微调数据：字段、题型、目标概率、图片、规则编号与交接流程（本文为中文原创，无英文对应） |
| [finetuning.md](finetuning.md) | 查微调的参考细节：环境、数据字段、参数默认值、校准、部署与评测 |
| [api.md](api.md) | 写客户端：请求与响应格式、控制项、置信度、`none_probability`、错误 |
| [none_probability.md](none_probability.md) | 使用 `none_probability`：它是什么意思、怎么测出来的、阈值为什么是 0.2 |
| [deployment.md](deployment.md) | 跑服务：引擎开关、图片与长输入、v1 模型 |
| [engines.md](engines.md) | 理解每个引擎怎么被驱动，以及实测到了什么 |
| [vllm_bidirectional.md](vllm_bidirectional.md) | 理解服务 Instant 模型的那个 vLLM 双向插件 |
| [quantization.md](quantization.md) | 构建或选择 FP8、GGUF、MLX 版本 |
| [serving_publicly.md](serving_publicly.md) | 把服务放到公网：密钥、TLS、速率限制、日志 |
| [benchmarks.md](benchmarks.md) | 复现准确率与延迟数字 |
| [../../patches/llama.cpp](../../patches/llama.cpp) | 为 Instant 的 GGUF 构建打补丁 |

英文原版见 [../README.md](../README.md)。
