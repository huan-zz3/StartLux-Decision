# 用 decision_sample 验证数据检查器：基线与边界

这份文档记录 `finetune/check_data.py` 在 `data/decision_sample/` 上的实测结果，以及这个样本能证明什么、不能证明什么。规则本身见 [finetune_data.md](finetune_data.md)。

`data/decision_sample` 随仓库分发。`tests/test_check_data_decision_sample.py` 复现下面的矩阵；目录缺失时（例如裁剪过的检出）整组跳过。脚本与训练端的一致性由 `tests/test_check_data.py` 单独保证。

## 怎么重跑

在仓库根目录执行：

```bash
python3 -m unittest -q tests.test_check_data              # 检查器 vs 训练端（权威）
python3 -m unittest -q tests.test_check_data_decision_sample   # 本文的矩阵（需要本地样本）

python3 finetune/check_data.py data/decision_sample/out/decision_sample_300.jsonl
python3 finetune/check_data.py data/decision_sample/out/decision_sample_extended_40.jsonl
python3 finetune/check_data.py data/decision_sample/out/invalid_cases.jsonl
python3 finetune/check_data.py data/decision_sample/out/image_cases.jsonl
python3 finetune/check_data.py data/decision_sample/out/image_quality_cases.jsonl
```

## 基线

| 文件 | 记录 | 问题 | 可训练 | 阻断 | 预期发现 |
|---|---|---|---|---|---|
| `decision_sample_300.jsonl` | 300 | 1079 | 1078 | 1 | 仅 `E08`（第 205 行 `service` 有 28 个选项）；`W01` 76 条；`W03` 含 `domain` 未知键与超长 state |
| `decision_sample_extended_40.jsonl` | 40 | 200 | 200 | 0 | 无 E；对象 `instructions`、对象型 `score.criteria`、对象/数组 `state` 均被接受 |
| `invalid_cases.jsonl` | 20 | 64 | 44 | 20 | 13 行被拒、7 行被接受，见下表 |
| `image_cases.jsonl` | 5 | 5 | 1 | 4 | 第 1 行合法；`E03` 坏 base64、`E03` 不可读图片、`E05` 201×1、`E04` 标记数不匹配；0 条 W05 |
| `image_quality_cases.jsonl` | 7 | 7 | 7 | 0 | 第 1 行合法；其余 6 行各 1 条 `W05`：JPEG 质量、非 RGB、透明度、EXIF、MIME、放大；**退出码 0** |

核心 300 条里另有 1 个并列最高概率（第 147 行 `b03-0084` 的 `greener_than`，0.44/0.44）。它留在 train 里合法；如果拆进 dev，会按 `E16` 阻断。

拆 240/60 跑 `--role dev` 时：dev 221 个问题全部可训练，`E17` 为空，稳定报两条 `W04`（`choice` 101、`noul` 120，均 <200）。

## 负样本期望矩阵

`invalid_cases_expected.json` 的 20 条里，只有 13 条在本文档规则下非法：

| 行 | 阻断规则 |
|---|---|
| 1 | `E02` |
| 2 | `E12` |
| 3 | `E06` |
| 5 | `E07` |
| 6 | `E08`、`E13` |
| 7 | `E08` |
| 8 | `E11` |
| 9 | `E13`、`E14` |
| 11 | `E15` |
| 13 | `E12` |
| 16 | `E12` |
| 17 | `E12`、`E14` |
| 19 | `E13`、`E14` |

另外 7 条是**合法**的，不能拿来当负样本：

| 行 | 样本主张 | 本文档规则 |
|---|---|---|
| 4 | 缺 `type` 非法 | `type` 缺省按 `choice` |
| 10 | 概率和 ≠ 1 非法 | 不需要自己归一化 |
| 12 | 概率是字符串非法 | 数字字符串按 `float()` 转换后可用 |
| 14 | target 指向不存在的问题非法 | 训练端忽略这些键；检查器报 `W03` |
| 15 | `state` 是数字非法 | train 允许；只有 dev 限制为字符串或对象（`E02`） |
| 18 | `noul` 带 `criteria` 非法 | 整体忽略并报 `W03` |
| 20 | 概率 > 1 非法 | 只要求非负且总质量非零，>1 会被归一化 |

## 这个样本覆盖不到什么

- **图片规则**：`image_cases.jsonl` 自带 1 条合法与 4 条缺陷记录（坏 base64、可解码但 Pillow 读不出、长宽比 >200、标记数不匹配），覆盖 `E03`/`E04`/`E05`；`image_quality_cases.jsonl` 用真实图片裁剪派生 1 条合法与 6 条 `W05` 建议（JPEG 质量、非 RGB、透明度、EXIF、MIME、放大），验证“只警告不报错”。`tests/test_check_data.py::ImageRuleMappingTest` 与 `ImageQualityWarningTest` 另外锁定规则归属和阈值。图片 token 估算、`--image-factor`、E18 的图片部分仍不在此样本范围内。
- **E18 文本长度**：需要 `--tokenizer` 和 `transformers`；最长 state 约 2,873 字符，通常不会越界，造不出阻断用例。
- **E01**：样本 20 条负样本都是合法 JSON；坏 JSON 行由 `tests/test_check_data_decision_sample.py` 的 fixture 覆盖。
- **E09/E10/E19**：样本没有非法 score 层数、非法选项 id、`NaN`/`Infinity`；这些由 `tests/test_check_data.py` 的 fixture 覆盖。

## 使用建议

- 把样本当作**正样本语料 + 覆盖矩阵**，不要当作“全部 20 条都应被拒”的 oracle；上面的 7 条差异是规则设计，不是脚本缺陷。
- 不要拿核心 300 当“干净基线”：第 205 行自带 `E08`，每行的 `domain` 会报 `W03`，76 条 `W01` 说明同名问题键跨域定义不一致。
- `W04` 在这份样本上无法满足：300 行按 10%–20% 拆 dev，每类问题到不了 200 条。看到 `W04` 是样本规模问题，不是数据错误。
- 样本是英文，且 `state` 是二次编码的 JSON 字符串（渲染后带字面引号）；中文场景与真实 `state` 形态要另备数据。
