# AGENTS.md

StartLux-Decision 的推理引擎与 HTTP 服务端（Python 包 `startlux-decide`）。它不是聊天模型服务：输入 state + 带类型的提问，输出每个选项的**概率**，概率由选项字母 token 的 logprob 读出，不解析自然语言答案。

## 代码地图

- `startlux_decide/` — 唯一在维护的引擎（打包范围 `include = ["startlux_decide*"]`）。
- `v1/` — 首个发布版的**冻结快照**（文档称其对应 tag `v1`；本克隆中没有该 tag），有自己的 `v1/startlux_decision/`、eval、finetune、demo harness。它不在打包范围内，也**不被根目录测试覆盖**；改当前引擎时不要顺手改 `v1/`，反之亦然（v1 的命令要在 `cd v1` 后运行）。
- `docs/` — 面向用户的手册；`patches/llama.cpp/` — llama.cpp 的三个补丁，配 `STARTLUX_LLAMACPP_BIDIR`。
- `finetune/finetune_lora.py` — **v1.1 Instant**（prompt 双向）的 LoRA 微调脚本，支持文本和 base64/data URI 图片（视觉塔冻结，只训练文本解码器），输出 `adapter/` + `merged/`，可选 `--calibrate`。它不属于打包范围；v1 的因果训练器仍在 `v1/finetune/`，两者不要混用。
- `startlux_decide/pipeline.py` 是核心：每个问题一个 asyncio task（问题之间不互相等 thought）；>26 选项先做 knock-out 分组；first read-out → AutoThink gate → 生成 thought（流式或分块，带循环检测）→ second read-out 与原概率按 `mix` 混合。
- `startlux_decide/backends/base.py` 定义引擎契约：`readout` + `generate`（AutoThink 才用），可选 `generate_stream`、`prefill`（前缀缓存共享长 state）、`reads_images`。新增引擎实现这个接口，并让 `backends/__init__.py` 与 `cli.py --backend` 同时认识它。
- `server.py` / `serving.py` 提供 `/v1/systemone`（别名 `/v1/decisions`）、`/v1/models`、`/health`、`/openrouter/models`。
- `vllm_plugin.py`（约 860 行）是 vLLM 通用插件，给 Instant 模型做 prompt 双向注意力；由 `STARTLUX_PREFIX_BIDIR=1` 门控，版本白名单为 vLLM 0.19.1 / 0.31.0，其他版本需 `STARTLUX_PREFIX_BIDIR_ALLOW_VERSION=1` 并重跑 fp32 校验。

## 命令

```bash
pip install -e '.[vllm]'          # 开发/服务；核心依赖只有 aiohttp + transformers
pip install -e '.[finetune]'      # v1.1 Instant 微调：torch + transformers 5.8 + peft + Pillow/torchvision + 快速线性注意力核
python3 -m unittest -v            # 全量测试（~270 tests / ~73 s，stdlib 即可跑，缺可选依赖的用例 skip）
python3 -m unittest -q tests.test_controls tests.test_readout          # 单个/多个文件
python3 -m unittest -q tests.test_finetune                             # 微调脚本的数据/分桶/温度单测；tiny 模型用例缺依赖时 skip
python3 -m unittest -q tests.test_golden.GoldenTest.test_against_release_proxy   # 黄金测试，单跑约 33 s
python3 -m unittest -q tests.test_pipeline.SomeTest.test_name          # 单个用例
STARTLUX_TEST_MODEL_DIR=/path/to/model python3 -m unittest -q tests.test_real_tokenizer tests.test_golden   # 用真实 tokenizer 加测
python finetune/finetune_lora.py --model models/StartLux-Decision-v1.1-0.8B-Instant \
  --train data/train.jsonl --dev data/dev.jsonl --out runs/0.8b-lora --calibrate
```

- 仓库**没有** lint / typecheck / formatter 配置，也没有 pytest 配置：验证就是 `unittest` + `startlux-decide bench|eval`。别引入 ruff/black 配置。
- 测试必须在仓库根目录跑：测试以 `tests.*` 包形式相对导入（`from . import reference_impl`）。
- CLI：`startlux-decide serve|launch-vllm|bench|eval|quantize`；`serve` 必须给 `--model`，`--launch-vllm` 才会自己拉起 vLLM。

## 硬约束（违反会被测试或评审挡下）

- **黄金测试是比特级契约**：`tests/test_golden.py` 要求本包与 `tests/reference_impl.py`（发布版 proxy 的逐字副本）对同一 fake vLLM 发出的**请求体**（含每个 thought 分块、seed、read-out prompt）以及概率/choice/score/thought 长度**完全一致**。改 prompt、read-out、AutoThink 路径必须保持概率不变，或在 PR 中说明偏移原因与幅度（`CONTRIBUTING.md`）。
- **不要"修" `tests/reference_impl.py`**：它是 oracle，不是待整理代码。`tests/fake_vllm.py` 同理，是协议级假服务端。
- **Python 3.9 兼容**：代码普遍使用内置泛型与 `X | None` 注解，每个模块首行都有 `from __future__ import annotations`。新增文件保留该 import，否则 3.9 运行期会炸。
- **保持依赖可选**：vLLM / torch / transformers / mlx-vlm 全在 optional extras；测试套件在只有标准库时必须能跑通（缺依赖就 `skipUnless`）。
- **Instant 微调必须双向**：`finetune/finetune_lora.py` 的训练前向走 `bidirectional.text_forward(..., causal=False)`；只调用 `bidirectional.enable()` 不够，`body(...)` 仍是 causal。v1 的因果训练器不要拿来跑 Instant。
- **图片训练走 serving 同一条路**：图片样本单条 forward，`load_media` → `processor` → `inner.get_image_features` / `get_placeholder_mask` / `get_rope_index` → `bidirectional.text_forward(inputs_embeds=..., position_ids=...)`；视觉塔冻结，LoRA 只挂文本解码器。
- **改动准确率/延迟**：用 `startlux-decide eval` / `bench` 在相同硬件上给出改前改后数字。

## 测试机制

- 无需 GPU、模型或网络：`backends/fake.py` 的 `FakeBackend` + `FakeTokenizer` 提供确定性读out/thought，`tests/fake_vllm.py` 模拟 vLLM HTTP 协议。新功能的测试优先复用它们，而不是加载真模型。
- 常用跳过条件写法：`@unittest.skipUnless(...)`，依赖 `aiohttp` / `numpy` / `torch` / `transformers` / `mlx_vlm`；CI 矩阵为 Python 3.9 + 3.12 × {stdlib, server(aiohttp+numpy)}（见 `.github/workflows/tests.yml`）。
- `tests/folders.py`、`tests/certs.py` 提供临时模型目录与自签 TLS 证书的 fixture 式辅助。

## 文档与风格

- 每个模块顶部的大段 docstring 就是设计文档（行为、边界、失败模式、环境变量），`pipeline.py`、`backends/*.py`、`vllm_plugin.py` 尤其如此。**改行为必须同步改对应 docstring**；面向用户的部分同步 `docs/*.md` 与 `CHANGELOG.md`。
- 注释解释"为什么"而不是"做了什么"，常见语气是记录被否决的方案与踩过的坑；保持这种密度。
- 沿用周围代码风格（英文注释/文档、约 120 列、少抽象）；README 里的控制项表格与 `docs/api.md` 需保持一致。

## 环境变量速查

| 变量 | 作用 |
|---|---|
| `STARTLUX_PREFIX_BIDIR` | `1` 才启用 vLLM 双向插件；未设置时插件完全不动 |
| `STARTLUX_PREFIX_BIDIR_*` | `ALLOW_VERSION` / `ALLOW_TRITON` / `CHECK` / `UNMATCHED` / `LEAN_ROWS`，见 `docs/vllm_bidirectional.md` |
| `STARTLUX_LLAMACPP_BIDIR` | `auto`（默认，读 `/props` 判断）/ `request` / `server` / `off`；带图片的 Instant 模型必须 `server` + 非因果 GGUF |
| `STARTLUX_TEST_MODEL_DIR` | 指向真实模型目录，启用真实 tokenizer 测试 |
| `STARTLUX_API_KEY` | `examples/client.py` 读它发 Bearer 头 |
