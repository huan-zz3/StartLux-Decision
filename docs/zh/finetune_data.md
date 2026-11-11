# 微调数据规范（v1.1 Instant）

这份文档写给**准备数据的人**，不要求你读过本项目的引擎代码。它和 `finetune/check_data.py` 是同一份规则的两种形式：
文档解释为什么，脚本负责判定。**规则以脚本为准**——脚本报 `E` 的那一条就是不能训练的那一条。

一句话流程：

```bash
# 1. 数据方自检（只需要 Python 和 Pillow）
python3 check_data.py data/train.jsonl --report data/train.report.json --against data/dev.jsonl
python3 check_data.py data/dev.jsonl --role dev --report data/dev.report.json --against data/train.jsonl

# 2. 看一眼模型实际会读到什么（强烈建议，至少看 5 条）
python3 check_data.py data/train.jsonl --render 5

# 3. dev 集转成评测文件（可选，评测时才用）
python3 check_data.py data/dev.jsonl --role dev --emit-eval data/dev_eval.jsonl
```

`--against` 传另一份划分，用来查 train/dev 是否重叠（E17）；不传时报告会把 E17 列为“未验证”。
数据问题只有 `E` 类会让脚本返回 1；`W` 类只进报告，不改退出码。报告里的 `unverified` 列出这次**没有真正检查**的规则
（没带 `--against` 是 E17，没带 `--tokenizer` 是 E18 的文本部分），它们是“不知道”，不是“通过”；
`not_applicable` 列出这次运行**不适用**的规则（例如 train 角色下的 E16 和 W04）。

## 一条记录

一行一个 JSON 对象，UTF-8。空行会被忽略（与训练端一致），但一行里只要有内容就必须是合法对象。

```json
{"state": {"ticket": "Refund still missing after two weeks, order 5521", "customer_tier": "pro"},
 "questions": {
   "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                         "technical": "App bugs and outages"}},
   "urgent": {"type": "noul", "instructions": "Does the customer ask for this to be solved today?"},
   "severity": {"type": "score", "instructions": "How badly is the customer affected?",
                "criteria": ["cosmetic", "annoying", "blocks the customer"]}},
 "targets": {"team": {"billing": 1.0}, "urgent": {"false": 1.0}, "severity": {"1": 0.7, "2": 0.3}}}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `state` | 是 | 证据。字符串原样使用；对象/数组会渲染成紧凑 JSON。`null`、`{}`、`[]`、空串或缺失算 `E02`；**dev 集**里 `state` 还必须是字符串或对象（评测转换只接受这两种） |
| `questions` | 是 | 问题集合，键是问题 id（同一数据集内唯一且稳定），值是问题定义 |
| `targets` | 是 | 与 `questions` 同名键，每个问题都必须有（`E12`）；值是「选项 id → 概率」，不是「选项序号」 |
| `images` | 否 | 记录级图片列表，base64 字符串或 data URI；最多 16 张 |
| `id` | 否 | 记录 id，缺省是 `line-<行号>` |

顶层出现未知键不算错，但会报 `W03`——多半是字段名拼错了。

## 问题类型

`type` 缺省时按 `choice` 处理；`bool` 是 `noul` 的别名。

### choice（多选一，按字母）

- `criteria` 写成对象 `{选项 id: 说明}`，或列表 `["a", "b", "c"]`（此时没有说明文案）。
- 选项数 **2 到 26**（`E08`）。26 是硬上限：模型每个选项用一个字母读出概率（A–Z）。
- 选项 id 不能为空、不能含换行、不能超过 200 字符（`E10`）。
- 选项在 prompt 里的顺序就是你写的顺序；字母按下标分配（A、B、C…）。
- 选项 id 同时也是 `targets` 的键，所以**别用会重名的 id**，也别在标注后改名。

### score（有序档位）

- `criteria` 写成列表，**从低到高**：`["cosmetic", "annoying", "blocks the customer"]`。
- 档位数 **2 到 10**（`E09`）。注意上限是 10，不是 26。
- `targets` 的键是档位的选项 id，也就是 `"0"` 到 `"n-1"`（`"0"` 表示你列表里的第一档，不是你在下游系统里的等级号）。
- 写成对象 `{"3": "严重", "1": "轻微", "2": "中等"}` 也接受，但内部会**按数值键排序**再用；
  如果键不是数字，就按你写的顺序。为了不踩这个坑，建议一律用列表。

### noul（是否）

- 只有两个选项，id 固定为 `"true"` / `"false"`，`targets` 的键也必须是这两个字符串。
- `instructions` 可省略，省略时使用默认文案 `Which answer fits the evidence?`；
  写了空字符串则算 `E11`。
- `criteria` 若写成列表会被渲染器整体忽略（只保留 true/false 两个选项），
  想给两个选项加说明请写成 `{"true": "…", "false": "…"}`；对象里除 `true`/`false` 之外的键同样被忽略，
  并会报 `W03`。

## 目标概率

- 键必须来自该问题的选项 id，否则算 `E13`。这一条以前会被**静默当成 0**，现在直接报错。
- 值必须是非负数（`E15`），且总和不能为 0（`E14`）。
- **不需要自己归一化**：写成票数 `{"billing": 3, "shipping": 1}` 和写成概率 `{"billing": 0.75, "shipping": 0.25}`
  等价，脚本会按比例归一化。多个标注者的票数可以直接写。
- 只有一个标签就写 one-hot `{"billing": 1.0}`。
- 并列最高概率在**训练集**里合法（当作软标签）；在 **dev 集**里算 `E16`，
  因为评测数据必须能取出唯一答案。

## 图片

- 只接受 base64 字符串或 data URI（`data:image/png;base64,...`）；**不接受本地文件路径**。
- 一条记录最多 16 张（`E03`）。
- 在 `state` 里用 `<image>` 标记每张图的位置，标记数必须等于 `images` 张数（`E04`）。
  不写标记时，图片会被自动放在证据开头（多张时是 `Image 1: …`、`Image 2: …`）。
- 每张图会被缩放到「边长都是 32 的倍数、总像素在 65,536 到 1,048,576 之间、保持长宽比」，
  然后按 **每 32×32 像素 1 个 token** 计入 prompt 长度。1024×1024 的图 = 1024 个 token。
- 长宽比超过 200 倍的图会报 `E05`。
- 一条记录里的所有问题共用这批图片。

**图片是最容易把 prompt 顶爆的地方**：16 张 1024×1024 就是 16,384 个 token，超过默认的 8,192 上限，
整条记录不会被训练（`E18`）。图片 token 数离线就能算准，脚本会直接告诉你。

计算按「每 32×32 像素 1 个 token」。这个比例来自模型目录里的 `patch_size × spatial_merge_size`，
出厂模型是 32；只有在 32 不成立的自定义模型上，离线的图片侧判断才会偏。带 `--tokenizer` 的验收运行会直接读
模型目录的 `config.json` 取这个比例；读不到时报告会把 E18 标为未验证，不会假装通过。

## 训练集 / dev 集

- 留出 **10%–20%** 作为 dev 集，两者不要来自同一批样本。
- dev 集额外要求：**不能有并列最高概率**（`E16`）；`state` 只能是字符串或对象（评测转换的限制，`E02`）；
  **每类问题至少 200 条**，否则温度校准会退回合并温度（`W04`）。
- train 和 dev 的 `state`（连同图片）不能重复（`E17`），否则 dev 指标虚高。`E17` 比较的是渲染后的
  `state` 文本加图片字节；`W03` 的“重复 state”只看文本，所以同一段文本配不同图片时是 `W03`，
  不是 `E17`。重新编码或缩放过的同一张图不算重复。
- 同一个问题 id 在全部记录里必须是同一个问题：`type`、`instructions`、选项集合都应一致（`W01`）。
  选项集合逐条变化在技术上合法，但会让同一个字母在不同记录里指向不同东西，学习效果明显变差。
- 问题 id 不能是空字符串（`W03`）：训练端能跑，但报告和 `targets` 里没法稳定引用这个问题。

## 模型会读到什么

`--render` 打印的是**喂给对话模板之前**的系统与用户两段文本，逐字一致。对话模板自己会在这两段外面套上
模型专属的标记（例如 `<|im_start|>`），那部分由模型目录决定，这里不显示——它不随你的数据变化。

```
[system] Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.
[user]
Evidence:
客户说上周的退款还没到账，订单号 5521，语气有点着急。

Question: Which team should handle this ticket?
Options:
A) billing: Payments, refunds and invoices
B) shipping: Delivery and tracking
C) technical: App bugs and outages
D) account: Login and account access
```

几个只在这个视图里才看得见的细节：

- 选项按 `字母) id: 说明` 渲染；`id` 本身会出现在 prompt 里，别用内部代号。
- `state` 如果是对象，数组长度 ≥ 8 时每个元素会被注入 `_index` 字段（模型靠它引用第几项）。
  不想要这个行为就把 state 写成字符串。
- 分数题的档位顺序就是渲染顺序，写反了模型学到的是反的。

`--canonical out.jsonl` 写出**训练实际消费的形态**：`targets` 补齐到选项顺序并归一化、`state` 渲染成文本、
图片保持 data URI、每条记录补一个 `id`。它适合用来对比“我写的数据”和“模型读的数据”；
注意 `state` 里的 `<image>` 标记会原样保留，不要照着 canonical 再手工插一遍图片。

## 规则总表

| 编号 | 级别 | 含义 |
|---|---|---|
| E01 | 阻断 | 不是合法 JSON，或不是 JSON 对象 |
| E02 | 阻断 | `state` 缺失、为 `null`/`{}`/`[]`/空串；dev 集里 `state` 还不是字符串或对象 |
| E03 | 阻断 | `images` 形状错误、超过 16 张、无法解码 |
| E04 | 阻断 | `<image>` 标记数与图片张数不一致 |
| E05 | 阻断 | 图片长宽比超过 200 倍 |
| E06 | 阻断 | `questions` 形状错误、问题定义不是对象 |
| E07 | 阻断 | `type` 不是 `choice`/`noul`/`bool`/`score` |
| E08 | 阻断 | choice 的 `criteria` 形状错误，或选项数不在 2–26 |
| E09 | 阻断 | score 的 `criteria` 形状错误，或档位数不在 2–10 |
| E10 | 阻断 | 选项 id 为空、含换行、或超过 200 字符 |
| E11 | 阻断 | choice/score 的 `instructions` 为空 |
| E12 | 阻断 | 某个问题没有 `targets` 条目 |
| E13 | 阻断 | `targets` 里出现不属于该选项集合的键 |
| E14 | 阻断 | 目标值不是数字，或总质量为 0 |
| E15 | 阻断 | 目标值出现负数 |
| E16 | 阻断 | dev 集出现并列最高概率 |
| E17 | 阻断 | train/dev 出现重复证据（需要 `--against`；不传则该规则标为未验证） |
| E18 | 阻断 | prompt 超过 `--max-length`（`--tokenizer` 模式精确判定；离线只能判定图片侧下界） |
| E19 | 阻断 | 目标值不是有限数字（`NaN` / `Infinity` 会让训练 loss 变成 NaN） |
| W01 | 观察 | 同一问题 id 在不同记录里定义不一致 |
| W02 | 观察 | 某个选项从未作为答案出现 |
| W03 | 观察 | 未知顶层键、被忽略的 `noul` criteria、问题键为空、重复 state（只看文本）、超长 state |
| W04 | 观察 | dev 集某类问题少于 200 条 |

## 交接流程

1. **数据方自检**：`check_data.py` 退出码为 0，报告里的 `trainable` 是**结构上**可训练的问题数；
   文本长度要等第 3 步带 `--tokenizer` 的那次运行才算完整，两次数字不一致是正常的。
2. **交数据**：`train.jsonl`、`dev.jsonl` 和两份报告一起交付，报告和数据的行号能对上。
3. **开发者验收**：用**同一个脚本**加 `--tokenizer <模型目录>` 再跑一遍。这一步做两件事：
   - 判定真实 prompt 长度（`E18`），报告里 `length_verified` 会变成 `true`；
   - 给出最终账目：交付多少问题、可训练多少、每条被拦下的原因。

   没有 `--tokenizer` 的那次运行，`length_verified` 是 `false`——这不是通过，只是“结构部分已通过”。
4. **训练**：训练日志会分别打印「超过 26 个选项」和「超过长度」两种跳过数；
   它们都应该是 0，不是 0 说明数据和验收报告对不上，先查原因再训练。

## 报错怎么读

```
共 2 个文件：12480 个问题，11901 个可训练，579 个被阻断；3 条阻断发现，12 条观察项。
未验证的规则：E17, E18
[E13] data/train.jsonl:412: 问题 'team' 的 target 含不属于选项的键：['billling']
    修复：把这些键改成选项 id 之一：['billing', 'shipping']
[E04] data/train.jsonl:917: 图片标记数与图片数量不一致：state 里有 2 个 <image> 标记，记录有 1 张图片
    修复：让 state 里的 <image> 标记数与 images 数量一致，或删掉标记
[W04] data/dev.jsonl: dev 集里 mood 问题只有 83 个（低于 200，校准会退回合并温度）
```

- `[E…]` 修到 0 再交。
- `[W…]` 不影响退出码，逐条判断是否要处理。
- `用法错误：…` 是命令行或路径的问题（返回码 2），按提示改命令，不是数据问题。
- 如果看到“这是检查器的问题，请把这条记录发回来”，那是脚本自身的缺陷，不要改数据。
