# decision 数据集测试样本（decision-index 同族格式）

用于校验「decision 数据格式检验脚本」的小体量样本集。总大小 < 700 KB，共 360 行，无需登录、无需下载大文件。

- **内容来源**：Hugging Face `mghafiri/decision-model-scenarios`（2,000 条合成场景 / 9,716 个类型化问题 / 32 个业务域，MIT 许可）
- **为什么不用 decision-index**：官方套件 12.5 万条请求、约 7 GB 下载量，且需要登录接受 HALE/GSM8K 条款；本目录只取同一范式（`state` + 类型化 `questions` + 概率 `targets`）的轻量数据
- **与上游的关系**：不是原样拷贝，而是按你给的样例格式做了规范化（见「与上游的差异」）

## 文件清单

| 文件 | 行数 | 用途 |
|---|---|---|
| `out/decision_sample_300.jsonl` | 300 | **主样本**，严格符合你给的最小格式：字符串 `state` + `choice`/`noul` + 扁平概率 `targets` |
| `out/decision_sample_extended_40.jsonl` | 40 | 更宽的**合法**空间：对象/数组 `state`、`criteria` 为 null、`score` 题型、对象型 `instructions` |
| `out/invalid_cases.jsonl` | 20 | **负样本**，每行注入且仅注入一个缺陷，用于验证脚本「该拒的确实拒了」 |
| `out/invalid_cases_expected.json` | — | 负样本清单：行号、缺陷描述、期望报错、可能的连带违规 |
| `build_dataset.py` | — | 生成脚本，确定性（seed=20261111），可复现 |
| `verify.py` | — | 独立实现的严格校验器，可作为你脚本的对照基准 |
| `data_raw/` | 2,000 | 上游原始 jsonl（7.3 MB），离线复现用 |

## 核心格式（`decision_sample_300.jsonl`）

```json
{"state": "客户说上周的退款还没到账……",
 "questions": {"team": {"type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "criteria": {"billing": "Payments, refunds and invoices"}},
               "urgent": {"type": "noul", "instructions": "Does the customer need an answer today?"}},
 "targets": {"team": {"billing": 1.0}, "urgent": {"true": 1.0}},
 "id": "b16-0044", "domain": "customer_support"}
```

约束（`verify.py` 严格模式逐条检查）：

1. `state` — 非空字符串
2. `questions` — 非空对象；键非空
3. 每个问题 `type ∈ {choice, noul}`；`instructions` 非空字符串
4. `choice` 必须有 `criteria` 对象，**≥ 2 个选项**，每个选项的描述为非空字符串
5. `noul` **不得**带 `criteria`
6. `targets` 与 `questions` 键集**完全一致**（不多不少）
7. `choice` 的 target 键集 = `criteria` 键集；`noul` 的 target 键集 = `{true, false}`
8. 概率为数字（非字符串、非布尔），∈ [0, 1]，每个问题的概率和 = 1（容差 1e-6）
9. `id` / `domain` 为可选附加字段，`python build_dataset.py --minimal` 可去掉

## 与上游的差异（规范化做了什么）

| 项 | 上游 `mghafiri/decision-model-scenarios` | 本样本（核心文件） |
|---|---|---|
| `state` | JSON 字符串，解码后可为 str / object / array | 全部为 str（对象/数组形态放进扩展文件） |
| `targets` 结构 | `{"q": {"probabilities": {...}, "difficulty": ..., "note": ...}}`，noul 为 `{"q": {"noul": 0.93}}` | 扁平化 `{"q": {option: prob}}`，noul 为 `{"true": p, "false": 1-p}`；`difficulty`/`note` 丢弃 |
| `criteria` 为 null | 1,319 个 choice 题只给选项键、描述为 null | 由选项键机械生成可读描述（`tier_a` → `Tier a`），非空字符串 |
| `score` 题型 | 2,149 个，`criteria` 有 8 种异构形态 | 核心文件剔除；扩展文件统一为 `[{level, definition}]` |
| `instructions` | 9,407 个字符串 / 309 个对象 | 核心文件仅保留字符串形态；对象形态进扩展文件 |
| 概率精度 | 浮点，和恰为 1 | 保留 6 位小数，舍入残差并入最大项，和精确为 1 |

原样信息全部保留：问题文本、选项键、概率分布、`id`、`domain` 均来自上游未改动；被丢弃的只有 `difficulty`/`note` 与 `score` 题。

## 样本特征

核心 300 行：

- 32 个域全覆盖（customer_support / it_helpdesk / banking_fintech / 保险理赔 / 招聘 / 合规 / 智能家居 …）
- 1,079 个问题：`choice` 503、`noul` 576
- 每行 2–6 个问题（中位 3）
- 每道 choice 2–28 个选项（中位 4），覆盖到「远超常见 4 选项」的极端形态
- `state` 长度 11–2,873 字符（中位 309），含中文引号、破折号等非 ASCII 字符，可顺带验证 UTF-8 与转义处理
- `id` 全局唯一，便于定位「第几条校验失败」

## 负样本怎么用

`invalid_cases.jsonl` 的 20 行覆盖：缺 `state`、缺 `targets`、`questions` 为数组、缺 `type`、未知 `type`、单选项 choice、缺 `criteria`、空 `instructions`、target 键与 criteria 不符、概率和 ≠ 1、负概率、概率为字符串、缺某个问题的 target、target 指向不存在的问题、`state` 为数字、空问题键、`targets` 为数组、noul 带 criteria、noul 用 yes/no 键、概率 > 1。

用法：逐行喂给你的脚本，期望**全部被拒**；再对照 `invalid_cases_expected.json` 检查报错定位是否指向正确字段。`also_violates` 字段标出「同一处必然会连带触发」的次要违规，避免把正确行为误判成误报。

## 复现与调整

```bash
python build_dataset.py                     # 默认：300 行概率分布版
python build_dataset.py --n 500             # 换行数（池上限 444 行）
python build_dataset.py --hard-labels       # 折叠为 one-hot 硬标签（argmax=1.0），对应你样例里的 1.0 写法
python build_dataset.py --minimal           # 去掉 id/domain，只留 state/questions/targets
python verify.py                            # 正样本全通过 + 20/20 负样本被拒
```

## 使用须知

- **语言为英文**。格式校验与语种无关；若你的脚本涉及中文分词或字数统计，需要另备中文样本（可基于本集翻译，或告我直接生成中文版）。
- 内容为 LLM 合成的英文业务场景，概率标签代表「专家意见分布」而非真实结果，不适合做效果评估基准，只适合做格式与流程校验。
- 上游许可 MIT，可自由使用与再分发；本目录产物继承同一许可。
