> 译自 [../serving_publicly.md](../serving_publicly.md)（commit 9ba65e2）。两者不一致时，以英文原版为准。

# 公开提供服务（serving）

`startlux-decide serve` 可以直接面向互联网：API 密钥、HTTPS、提前返回 429、OpenRouter 的提供商路由、不含内容的访问日志，以及含内容的请求日志。这些默认全部关闭。

| 选项 | 效果 |
|---|---|
| `--host 0.0.0.0 --port 8443` | 监听所有网络接口（默认是 127.0.0.1，仅本机） |
| `--api-keys-file PATH` | 除 `/health` 外每条路由都需要 `Authorization: Bearer <key>`，密钥取自该文件，否则返回 401 |
| `--ssl-certfile PATH --ssl-keyfile PATH` | 使用 TLS 1.2 或更新版本的 HTTPS；不给定时为明文 HTTP |
| `--max-inflight N` | 当有 N 个决策请求在途时，后续请求立刻得到 429，并带 `Retry-After: 1`；不排队 |
| `--openrouter-models-file PATH` | OpenRouter 提供商模型文档，原样在 `--openrouter-models-path`（默认 `/openrouter/models`）提供 |
| `--access-log PATH` | 每个请求一行 JSON，不含内容 |
| `--request-log DIR` | 每个决策请求一行 JSON，含其内容：客户端地址、密钥 id、请求体、答案与思考（thought）；图片存为文件 |

`POST /v1/decisions` 与 `POST /v1/systemone` 是同一个 API（后者是 OpenRouter 对它的命名）。请求中的 `model` 字段会被接受并忽略；响应里的 `model` 是服务器的 `--name`。

## 示例

27B AutoThink 模型跑在四张 GPU 上，每张 GPU 一个 vLLM 服务器（绑定到 127.0.0.1），决策服务器在 8443 端口：

```bash
startlux-decide serve --model /models/StartLux-Decision-27B-AutoThink --name startlux-decision-27b-autothink \
    --launch-vllm --gpus 0,1,2,3 --vllm-log-dir /var/log/startlux/vllm \
    --host 0.0.0.0 --port 8443 \
    --ssl-certfile /etc/startlux/tls/fullchain.pem --ssl-keyfile /etc/startlux/tls/privkey.pem \
    --api-keys-file /etc/startlux/api_keys --max-inflight 64 \
    --openrouter-models-file /etc/startlux/openrouter_models.json \
    --access-log /var/log/startlux/access.jsonl
```

未列出的都保持默认值（vLLM 参数、门控（gate）0.6、预算 4,096、混合 0.75）。如果某个 vLLM 服务器因显存不足启动不了，在 GPU 没跑其他东西时提高 `--gpu-memory-utilization`（0.88），或降低 `--max-model-len`（启动器会把 `--max-num-batched-tokens` 设成同一个值；比上下文长的提示（prompt）随后会得到 422）。

## API 密钥

* 每行一个密钥；空行和以 `#` 开头的行被忽略。密钥是一个可打印 ASCII 单词，至少 16 个字符。用 `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` 生成密钥，每个客户端一个。
* 文件变化时（修改时间、大小或 inode）会重新读取，每个请求都会检查。以原子方式替换它（写一个新文件，再改名覆盖旧文件）。含非法行的版本不会被采用：服务器保留原有密钥并记录行号。合法但没有密钥的文件会把所有客户端锁在外面。启动时文件必须存在且至少含一个密钥。
* 密钥只以 SHA-256 摘要保存，并与每个密钥做常数时间比较。它们永远不会出现在日志或错误消息里。日志用密钥 id 指代密钥，即其 SHA-256 的前 16 个十六进制数字（对密钥文件中的一行：`printf %s "$line" | sha256sum | cut -c1-16`）。让文件仅对服务器用户可读（`chmod 600`）。
* `/health` 和 `/v1/health` 保持开放，供监督进程和负载均衡器使用；它们不返回任何请求数据。没有有效密钥时，它们只返回状态、模型名和类型，不返回引擎地址或设置。

## TLS

证书必须对公网主机名有效，并来自公共 CA（例如 Let's Encrypt）。证书在启动时加载：续期后重启服务器，或者在前面放一个做 TLS 终止的反向代理（reverse proxy），在其背后于 127.0.0.1 上提供明文 HTTP（此时在决策服务器上保留 `--api-keys-file`）。

## 提前 429

OpenRouter 把 429 响应与错误分开计数，并要求提供商尽早返回它们而不是排队。使用 `--max-inflight N` 时，服务器统计正在应答的决策请求数；到达 N 时，它在读取请求体之前立即返回 429，带 `Retry-After: 1` 和一个 JSON 错误。未认证的请求先得到 401，且从不计数。用 `startlux-decide bench --concurrency` 在真实硬件上选择 N：取 p95 延迟仍可接受的并发数，并记住会思考的请求需要数秒。在 OpenRouter 文档中声明相同的数字作为容量。

## OpenRouter 模型文档

OpenRouter 从提供商提供的一份文档中读取该提供商的模型。其 schema 对 decisions 模态不公开，因此由运营者编写该文件，服务器原样提供（它只检查文件是 JSON，且 `data` 是非空的对象数组，每个对象有字符串 `id`；文件变化时会重新读取，损坏的版本不会被采用）。[openrouter_models.example.json](../openrouter_models.example.json) 使用了 OpenRouter 提供商文档中的字段：

| 字段 | 示例 | 说明 |
|---|---|---|
| `id` | `startlux/startlux-decision-27b-autothink` | 该模型的 OpenRouter slug |
| `name`, `description` | | 展示给用户 |
| `created` | `1791676800` | Unix 时间 |
| `schema_version` | `"2.4"` | |
| `input_modalities` | `[{"type": "text"}]` | |
| `output_modalities` | `[{"type": "decisions"}]` | decisions 模态：OpenRouter 必须确认这个值 |
| `quantization` | `"bf16"` | |
| `datacenters` | `[{"country_code": "CN"}]` | GPU 所在位置 |
| `is_free` | `true` | |
| `capacity` | `[{"type": "rate", "unit": "requests", "per": "minute", "value": 600}, {"type": "concurrency", "value": 32}]` | 可选；与 OpenRouter 确认条目类型，并让并发数不高于 `--max-inflight` |

## 访问日志

`--access-log` 为每个请求追加一行 JSON：

```json
{"time":"2026-10-11T08:15:02.317Z","request_id":"4f1c...","method":"POST","route":"/v1/decisions","status":200,
 "total_ms":2210.5,"client_ip":"203.0.113.7","key_id":"9c1e5a0b2f64d317","questions":3,"input_tokens":912,
 "thought_tokens":640,"wall_ms":2209.8,"first_pass_sum_ms":98.1,"thought_span_ms":2061.4,"after_thought_sum_ms":44.0}
```

`route` 是匹配到的路由（未知路径为 null；查询字符串从不写入）。`client_ip` 取 `X-Forwarded-For` 的第一个条目，否则 `X-Real-IP`，否则 TCP 对端。`key_id` 指代请求所用的密钥（没有 bearer 密钥时为 null；被拒绝的密钥也会给出）。未到达模型的请求（401、429、读取问题之前的 4xx），其 token 计数和耗时都是 null。文件保持打开：用 `copytruncate`（logrotate）轮转，或重启服务器。服务器自己的日志（stderr）也不含请求内容：错误带有请求 id、异常类型和栈，引擎故障带有引擎消息。502 只告诉客户端请求 id；500 告诉请求 id 和异常类型。

## 请求日志

`--request-log DIR` 保存客户端发送的内容和得到的回复，面向必须保留这些内容的运营者（例如调查投诉或滥用）。每个决策请求（`POST /v1/decisions`、`POST /v1/systemone`），无论状态如何，都向 `DIR/requests-YYYY-MM-DD.jsonl` 追加一行 JSON（每个 UTC 日一个新文件）：

| 字段 | 内容 |
|---|---|
| `time`, `request_id`, `route`, `status`, `total_ms` | 同访问日志 |
| `client` | `ip`（同上文的 `client_ip`）、`peer`（TCP 对端）、按收到的原样记录的 `x_forwarded_for` 和 `x_real_ip`、`user_agent` |
| `key_id` | 请求所用的密钥（见 API 密钥）；密钥本身从不写入 |
| `body_bytes` | 客户端声明的请求体大小 |
| `request` | 解析后的请求体；每张图片替换为 `{"sha256", "mime", "bytes", "file"}`；不是 JSON 的请求体为 `{"unparsed": ...}`（前 65,536 个字符）；在读取请求体之前就被拒绝的请求（401、429）为 null |
| `response` | 回复；对已应答的请求，包含每个答案的细节及其思考（thought）文本，即使请求没有要求这些内容（客户端仍然只得到它所要求的） |

图片各写入一次，路径为 `DIR/images/<前 2 个十六进制数字>/<sha256>.<ext>`。目录以 700 模式创建，每个文件以 600 模式创建。写入失败只记录一次，从不导致请求失败。不删除任何内容：规划保留期（例如每天执行 `find DIR -name 'requests-*.jsonl' -mtime +30 -delete`，图片同理），并在隐私政策中说明，因为这些文件保存着你用户的数据。OpenRouter 向用户展示每个提供商的数据政策，所以也要在那里声明保留期。

## 检查清单

- [ ] `--host 0.0.0.0`（或公网接口）和一个防火墙开放的端口；不开放其他端口。vLLM 服务器监听 127.0.0.1 且没有认证：绝不要暴露它们。
- [ ] `--api-keys-file`，每个客户端一个密钥，文件模式 600，密钥变化时原子替换。
- [ ] `--ssl-certfile` / `--ssl-keyfile`，使用对该主机名公开可信的证书；续期后重启。
- [ ] `--max-inflight` 来自在同一批 GPU 上的 bench 运行；与 OpenRouter 文档中的容量相同（或更低）。
- [ ] `--openrouter-models-file` 为本部署编写（id、name、created、description、国家代码、capacity），并与 OpenRouter 确认输出模态。
- [ ] `--access-log` 打开并配置轮转。仅在必须保留内容时使用 `--request-log`，并配保留任务和数据政策说明。使用启动器参数的 vLLM 不会把提示（prompt）写进日志；不要给它加请求日志。
- [ ] 一个进程监督器，负责重启服务器并用 SIGTERM 停止它（服务器随后会停止它启动的 vLLM 服务器），并准备足够的打开文件数供在途请求使用（每个请求两个 socket）。`serve` 启动时会在硬限制内把自己的软限制提高到 65,536，当硬限制使其低于 8,192 时会警告。
- [ ] 从外部监控 `/health`；在这些 GPU 上通过 GPU 验证（在已知集合上运行 `startlux-decide eval`）。

一个 systemd unit：

```ini
[Unit]
Description=StartLux-Decision decision server
After=network-online.target
Wants=network-online.target

[Service]
User=startlux
ExecStart=/opt/startlux/venv/bin/startlux-decide serve --model /models/StartLux-Decision-27B-AutoThink ...
Restart=always
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=120
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

请求体限制为 64 MB。请求的问题数量不限；密钥是控制谁能发送请求的手段。
