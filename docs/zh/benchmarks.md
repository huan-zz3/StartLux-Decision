> 译自 [../benchmarks.md](../benchmarks.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# Benchmarks

## 准确率与校准：`startlux-decide eval`

输入：JSONL，每行一个请求，带有 gold 标签：

```json
{"id": "r1", "state": "...", "questions": {"team": {"type": "choice", "instructions": "...", "criteria": {...}}},
 "gold": {"team": "billing"}}
```

标签对选择题（choice）是选项 id，对是/否问题（noul）是 `true` / `false`（或 `"yes"` / `"no"`），对量表题（score）是等级索引（0 = 最低）或 criteria 键。没有标签的问题仍会发送，但不计分。

```bash
startlux-decide eval --url http://127.0.0.1:8090 --data gold.jsonl --out run.jsonl --concurrency 16
startlux-decide eval --score run.jsonl                    # score a predictions file again
startlux-decide eval --compare run_a.jsonl run_b.jsonl    # paired comparison
startlux-decide eval ... --controls '{"think_mode": "never"}'   # merged into every request
```

每个请求都带 `return_details` 发送。对每个问题，预测标签是：选择题（choice）的选项、P(yes) ≥ 0.5 时的 yes、或概率最高的等级；其概率即置信度。

| 指标 | 定义 |
|---|---|
| accuracy | 被计分问题中预测标签等于 gold 标签的比例 |
| ECE | 最高标签校准误差：10 个等宽置信度分箱，每个分箱的 \|准确率 − 平均置信度\| 按其占比加权 |
| Brier | 对各选项求和 (p − 1[选项为 gold])²，再取平均（是/否问题计两个选项） |
| thought share | 进行了思考（thought）的问题占比（来自答案的 details） |
| thought tokens | 所有问题的思考（thought）token 平均值，以及进行过思考的问题上的平均值 |

`--compare A B` 按请求 id 与 key（相同的 gold 标签）配对问题，并报告：预测标签的一致率、两者的准确率、差值 B − A 及其 95% bootstrap 区间（对请求重采样 2,000 次，因为一个请求内的各个问题共享证据）、both / only-A / only-B 正确计数、门控（gate）决策的一致率，以及每个问题中任一选项概率的最大变化（均值与最大值）。可用它比较引擎、构建（BF16 对 FP8、GGUF 或 MLX）与设置。

## 延迟：`startlux-decide bench`

```bash
startlux-decide bench --url http://127.0.0.1:8090 --requests requests.jsonl --concurrency 1 --warmup 20 \
    --out latency_summary.csv --rows latency_rows.csv
```

请求文件每行是一个 `/v1/systemone` 请求体（忽略 `id` 与 `gold` 字段）。每个请求都带 `return_details` 发送。按请求统计：

| 列 | 测量方 | 含义 |
|---|---|---|
| `client_ms` | 客户端 | 从发出请求到收到完整答案 |
| `wall_ms` | 服务端 | 请求在服务端内部，从校验到生成答案 |
| `first_pass_sum_ms` | 服务端 | 各问题首轮引擎调用的总和 |
| `thought_span_ms` | 服务端 | 从发送第一个思考（thought）分块到收到最后一个分块，覆盖所有问题（无思考为 0） |
| `after_thought_sum_ms` | 服务端 | 各问题在思考（thought）之后的读出（read-out）之和 |
| `questions`、`thought_questions`、`thought_tokens` | 服务端 | 按请求统计 |

服务端的计时只包住引擎调用本身：提示（prompt）渲染与 tokenization 不在各阶段时间内，而在 `wall_ms` 之内。在 vLLM 上，思考（thought）跨度包含等待一个 wave 的时间（每个分块最多 `--wave-gather`，0.3 s）。汇总 CSV 对成功的请求给出每一列的 n、均值、中位数、p80 与 p95（分位数按次序统计量之间的线性插值计算）。

延迟取决于引擎、GPU、进行思考的问题占比以及负载。要与已发布的数字可比，应在预热之后一次只测一个请求（`--concurrency 1`），使用带发布标志的服务器，且其 GPU 上不运行其他任务，并在延迟旁一并报告带思考的请求占比。

## 复现已发布的数字

用默认配置在 vLLM 上服务该目录（`startlux-decide serve --model DIR --launch-vllm --gpus ...`），并发送该 benchmark 的请求。默认配置即发布运行所用的设置（README，“Published numbers”）；对 AutoThink，混合系数 0.75 就是已发布数字所用的值（它被施加于每个问题已存储的第一次与第二次读出）。思考（thought）是采样的，因此重跑 AutoThink benchmark 会在采样噪声范围内与发布运行不同；Instant 的读出在批处理的数值误差范围内是确定性的。
