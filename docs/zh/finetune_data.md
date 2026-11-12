# 微调数据规范（v1.1 Instant）

这份文档写给**准备数据的人**，不要求你读过本项目的引擎代码。它和 `finetune/check_data.py` 是同一份规则的两种形式：
文档解释为什么，脚本负责判定。**规则以脚本为准**——脚本报 `E` 的那一条就是不能训练的那一条。

一句话流程（以下命令在仓库根目录执行；脚本在 `finetune/` 下，也可以先 `cd finetune` 再省略路径前缀）：

```bash
# 1. 数据方自检（只需要 Python 和 Pillow）
python3 finetune/check_data.py data/train.jsonl --report data/train.report.json --against data/dev.jsonl
python3 finetune/check_data.py data/dev.jsonl --role dev --report data/dev.report.json --against data/train.jsonl

# 2. 看一眼模型实际会读到什么（强烈建议，至少看 5 条）
python3 finetune/check_data.py data/train.jsonl --render 5

# 3. dev 集转成评测文件（可选，评测时才用）
python3 finetune/check_data.py data/dev.jsonl --role dev --emit-eval data/dev_eval.jsonl
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

### 其他被接受的写法

下面这些形态训练端和检查器都接受，只是不推荐作为主要写法：

- `instructions` 可以是**对象**：会渲染成紧凑 JSON 文本作为问题文案。空对象会变成字面量 `"{}"`，
  不算 `E11`；只有空字符串/纯空白才报 `E11`。
- `choice` 的 `criteria` 值可以是 `null`：表示该选项没有说明文案，渲染器只写选项 id
  （效果等同于把 `criteria` 写成选项列表）。
- `score` 的 `criteria` 可以是**对象列表**，例如 `[{"level": "0", "definition": "…"}]`：
  每个元素会被渲染成紧凑 JSON 字符串，档位 id 仍然是 `"0"`…`"n-1"`。推荐直接用字符串列表。
- `targets` 里的数字字符串（如 `"0.75"`）会按 `float()` 转换后使用，见「目标概率」。

## 目标概率

- 键必须来自该问题的选项 id，否则算 `E13`。这一条以前会被**静默当成 0**，现在直接报错。
- `targets` 里出现不属于任何问题的键会报 `W03`：训练端会忽略这些键，通常是问题 id 拼错了。
- 值必须是非负数（`E15`），且总和不能为 0（`E14`）。
- 值会先按 `float()` 转换再使用，所以 `"0.75"` 这样的数字字符串也能训练；转不成数字的字符串才算 `E14`。
  大于 1 的值同样合法，会和其他值一起按比例归一化。
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

### 图片本身的质量（决定识别精度）

这一节分两层：`E03–E05` 只判定结构问题（能否解码、`<image>` 标记数、长宽比），token 上限由 `E18` 表达；能从图片元数据与像素尺寸离线判定的质量建议由 `W05` 提示（**只警告，不阻断，不改退出码**）。像素糊不糊、字够不够大仍然看不见——那部分只能靠人工和上游保证。下面把**代码事实**与**由实现推导的做法**分开写，后者是经验值，`W05` 只覆盖其中可机械判定的部分。

`W05` 目前覆盖：实际透明像素（alpha <255 或 `tRNS`；全不透明的 alpha 图不算问题）、非 RGB 的像素模式（灰度、调色板、CMYK、16-bit…）、EXIF 方向不是 1、JPEG 估计质量 <90、data URI 的 MIME 与实际格式不符、原图面积低于 `--min-pixels`（会被插值放大）。ICC/宽色域造成的实际偏色、是否二次有损转码、文字够不够大，`W05` 都不判断（离线只能看到 ICC 是否存在，看不到它的影响），仍需人工检查。

**代码事实**

- **只看像素**：训练走 `finetune_lora._image_loss` → 模型图片处理器（Qwen2-VL 系列的 `Qwen2VLProcessor`），
  进去的是 PIL 图，原始编码字节不参与训练。格式本身不影响模型，只有解码出的像素影响。
- **格式**：任何 Pillow 能解码的格式都收（PNG / JPEG / WebP / GIF 首帧…），解不开才算 `E03`。
- **色彩**：训练与服务路径上只有 `img.convert("RGB")`（`startlux_decide/images.py:125` 与 `finetune/check_data.py` 的 `image_info`），
  没有任何 ICC / `ImageCms` 调用，所以宽色域图按原始数值当 sRGB 用，颜色会偏。
  这一步还会**直接截断**而不是缩放：16-bit 灰度 PNG 的 8,000 会变成 255（`convert("RGB")` 裁剪，实测 256 就已经是纯白），
  16-bit 灰图基本会读成白纸；CMYK / 灰度也都在这里被压成 8-bit RGB。
- **透明度**：`convert("RGB")` 丢弃 alpha、**不做背景合成**。实测 RGBA 保留该像素原有的 RGB 值，
  带 `tRNS` 的调色板 PNG 透明区变黑——透明区域的颜色完全取决于生成工具，不可预期。
- **EXIF 方向**：训练路径没有 `ImageOps.exif_transpose`（`load_media` 只有 `open` + `convert`），
  带旋转标记的图以**未旋转**像素训练。`transformers` 只在 `load_image` 这类**字符串输入**路径里转正，
  推理后端走哪条路不确定，所以别指望 EXIF 被处理。
- **缩放**：`smart_resize` 把边长取 32 的倍数、面积夹在 65,536–1,048,576（约 256²–1024²）；
  训练端实现用 BICUBIC（`finetune_lora.py:104`），处理器默认也是 BICUBIC。
  低于下界是插值**放大**（不产生新细节），高于上界是**缩小**丢细节。
- **token 只跟面积走**：`(h/32)*(w/32)`，1024² = 1024 token，最小一张 64 token。
  长宽比不改变同面积的 token 数，也没有补边；极端比例只是把短边压小（超过 200 倍报 `E05`）。

**由实现推导的做法（经验，不是硬规则）**

- 含文字的图（截图 / UI / 票据）用 **PNG**；照片用 JPEG 时别留肉眼可见的块状伪影，经验上质量 ≥ 90，
  80 以下小字和细图标先坏；已有的 JPEG 不要二次有损转码（`W05` 会从量化表估计 JPEG 质量，<90 时提醒）。
- 送数据前统一 **sRGB + 8-bit**，剥掉 EXIF 并按 EXIF 转正；透明图先铺到白底（或线上底色）再编码
  （`W05` 会检查实际透明像素、非 RGB 模式和 EXIF 方向）。
- **别把要辨认的细节交给缩放**：缩放后文字行高至少留 1–2 个视觉 patch（16–32 px，patch 见 `images.py` 的 `FACTOR` 注释）。
  整页 / 整屏字太小就裁成几张局部图——这是 `max_pixels` 上限决定的：一张 4K 全图最多只剩约 100 万像素。
  原图面积低于 `--min-pixels` 时会被插值放大，`W05` 会提醒。
- 一张图只放判据需要的区域，四周留一点上下文；多个视角 / 多页拆成多张（上限 16 张）比拼接大图更能保住细节，
  `<image>` 的书写顺序就是 `images` 的顺序。
- base64 里的空白会被剥掉（可以折行写）；data URI 的 MIME 会被沿用，缺省按 `data:image/png`（`media.data_uris`），
  别写与实际格式不符的类型（`W05` 会比对声明值和解码出的真实格式）。
- 训练图和线上真正会送进来的图，在分辨率、裁剪、底色、压缩上要接近——线上往往是缩略图或二次压缩图，
  这是「训练指标好、线上认不出」的常见原因。

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
| E14 | 阻断 | 目标值无法转成数字，或总质量为 0 |
| E15 | 阻断 | 目标值出现负数 |
| E16 | 阻断 | dev 集出现并列最高概率 |
| E17 | 阻断 | train/dev 出现重复证据（需要 `--against`；不传则该规则标为未验证） |
| E18 | 阻断 | prompt 超过 `--max-length`（`--tokenizer` 模式精确判定；离线只能判定图片侧下界） |
| E19 | 阻断 | 目标值不是有限数字（`NaN` / `Infinity` 会让训练 loss 变成 NaN） |
| W01 | 观察 | 同一问题 id 在不同记录里定义不一致 |
| W02 | 观察 | 某个选项从未作为答案出现（按问题键 + 选项 id 分别统计） |
| W03 | 观察 | 未知顶层键、被忽略的 `noul` criteria、问题键为空、重复 state（只看文本）、超长 state、不属于任何问题的 target 键 |
| W04 | 观察 | dev 集某类问题少于 200 条 |
| W05 | 观察 | 图片质量建议：实际透明像素、非 RGB 模式、EXIF 方向、JPEG 估计质量 <90、data URI MIME 不符、原图过小会被放大（不阻断） |

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
