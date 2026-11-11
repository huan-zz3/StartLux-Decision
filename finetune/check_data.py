"""StartLux-Decision 微调数据自检器：一个数据提供方可以单独拷走的单文件脚本。

    python3 check_data.py train.jsonl
    python3 check_data.py train.jsonl --render 5
    python3 check_data.py train.jsonl --canonical out.jsonl
    python3 check_data.py train.jsonl --report train.report.json
    python3 check_data.py train.jsonl --against dev.jsonl
    python3 check_data.py train.jsonl --tokenizer MODEL_DIR
    python3 check_data.py dev.jsonl --role dev --emit-eval eval.jsonl

只依赖标准库和 Pillow（读图必须有 Pillow；缺了会在启动时用中文提示安装）。它不导入
``startlux_decide``，也不需要 torch / transformers；只有 ``--tokenizer`` 模式会在函数内部按需导入
``transformers``，该模式只在微调开发者的机器上跑。所有面向用户的文字（帮助、发现项、摘要）都是中文，
只有渲染出来的 prompt 正文和原样透传的数据值是英文/原文。

退出码：0 = 没有任何阻断发现；1 = 有阻断发现，或 ``--emit-eval`` 转换失败；2 = 用法/路径问题、
模型目录或聊天模板问题、以及检查器自身的缺陷。数据问题和检查器自身的问题从不相混：数据问题输出
``[E01] 文件:行 问题 'x': ...`` 和修复建议；用法或模型目录问题输出 ``用法错误：...`` 或中文的模型
目录诊断；只有真正意外的异常才输出 traceback 和"这是检查器的问题，请把这条记录发回来"（同样是 2，
靠这段文字区分）。

规则编号（``RULES`` 与 ``RULE_INFO`` 是唯一权威；方括号是标签）：

    E01 [trainer]  这一行不是合法 JSON，或不是 JSON 对象
    E02 [contract] state 缺失、null、{}、[]，或 state_text 之后是空白
    E03 [trainer]  images 形状不对、超过 16 张、不是 base64/data URI、Pillow 读不出
    E04 [trainer]  state 里的 <image> 标记数与 images 数量不一致
    E05 [trainer]  图片两边相差超过 200 倍
    E06 [trainer]  questions 形状不对、问题规格不是对象（空问题键是 W03 观察项）
    E07 [trainer]  type 存在但不是 choice/noul/bool/score（缺省 = choice）
    E08 [trainer]  choice 的 criteria 形状不对，或选项数不在 2..26
    E09 [trainer]  score 的 criteria 形状不对，或层数不在 2..10
    E10 [trainer]  选项 id 为空、含换行、或超过 200 字符
    E11 [trainer]  choice/score 的 instructions 为空白
    E12 [contract] 某个问题在 targets 里没有条目
    E13 [trainer]  target 的键不是该问题的选项 id
    E14 [trainer]  target 的值不是数字，或总质量为零
    E15 [contract] target 的值为负数
    E16 [contract] --role dev：最高概率在 1e-9 内并列
    E17 [contract] --against：train/dev 的 state + 图片重叠
    E18 [trainer]  prompt 有效长度超过 --max-length（无分词器时按图片 token 下界检查）
    E19 [contract] target 值不是有限数字（NaN/Infinity）

    W01 同一个问题键在不同记录里 type/instructions/选项集合不同
    W02 某个问题里的选项从未作为该问题的答案出现
    W03 未知顶层键；空问题键；渲染器丢弃的 noul criteria；同一 split 内重复 state；state 长度超过中位数 4 倍
    W04 --role dev：某类型的问题少于 200 个

``--report`` 写出的 JSON 键固定为：

    checker_version  检查器版本字符串
    commit           生成报告时的 git commit（拿不到时为 "unknown"，可用 --commit 覆盖）
    defaults         全部默认常量的快照（max_length/min_pixels/max_pixels/image_factor/max_images/
                     max_options/max_levels/annotate_min/tie_eps/aspect_limit）
    settings         本次实际生效的值（含命令行覆盖）
    overrides        与 defaults 不同的项
    role             "train" 或 "dev"
    mode             "tokenizer" 或 "plain"
    length_verified  是否真的校验过长度（只有 tokenizer 模式为 true）
    unverified       没有验证过的规则编号：未用 --tokenizer 时含 E18，未给 --against 时含 E17；
                     --tokenizer 模式读不出模型图片边长时，含图片的数据也把 E18 记为未验证
    not_applicable   本次运行不适用的规则：role=train 时的 E16、W04
    image_factor_source  图片 token 边长的来源："cli"（显式 --image-factor）、"model"（模型 config.json）
                     或 "default"（32）
    model_folder     使用的模型目录，未用 --tokenizer 时为 null
    files            每个文件一项：path/records/questions/trainable/blocked/blocked_records
    findings         全部 E 发现：rule/kind/tag/file/line/question/message/fix
    observations     全部 W 观察项，字段同 findings

``--canonical`` 写出的每行是 ``{"id", "state", "images", "questions", "targets"}``：state 是渲染后的
``state_text``，images 是原图的 data URI，questions 的选项带 criterion，targets 按选项顺序归一到和为 1。
只写出整条记录没有阻断发现的问题；一条记录没有可训练问题时整条跳过。``id`` 缺省为 ``line-<n>``。

``--emit-eval`` 只在 ``--role dev`` 且只有一个输入文件时可用；它跑在**原始记录**上，逐字镜像
``finetune/to_eval_data.convert``（同一个 gold 选择、同一个并列报错、同一个输出形状、同样的
written/skipped/dropped 统计），输出在整份输入转换成功后才写盘。空白行在本检查器所有阶段都被静默跳过
（与 ``load_items`` 一致）；``to_eval_data.convert`` 自己会把空白行当成非法 JSON 报错，检查器在这里刻意
更宽松，因此含空白行时两者只在空白行上有差别。
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import math
import os
import re
import statistics
import string
import subprocess
import sys
import traceback

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:                     # pragma: no cover - 只在提供方没装 Pillow 时触发
    print("这个检查器需要 Pillow 才能读取图片。请先运行：python3 -m pip install Pillow", file=sys.stderr)
    raise SystemExit(2)

CHECKER_VERSION = "1.0.0"
TIE_EPS = 1e-9
ASPECT_LIMIT = 200
DEFAULT_MAX_LENGTH = 8192
ALLOWED_KEYS = ("state", "questions", "targets", "images", "id")

# ---------------------------------------------------------------- 逐字复制的渲染常量（与 startlux_decide 一致）

SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
LETTERS = string.ascii_uppercase
TYPES = ("choice", "score", "noul")
MAX_OPTIONS = 26          # one letter per option, A to Z
MAX_LEVELS = 10           # score levels
ANNOTATE_MIN = 8          # JSON arrays at least this long carry an "_index" on every element
NOUL_INSTRUCTIONS = "Which answer fits the evidence?"
_BARE = re.compile(r"^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$", re.I)

IMAGE_MARK = "<image>"
SLOT = "\x00IMAGE\x00"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
FACTOR = 32
MIN_PIXELS = 65536
MAX_PIXELS = 1 << 20
MAX_IMAGES = 16
_DATA_URI = re.compile(r"^data:(image/[\w.+-]+)?(;[\w=.+-]+)*;base64,", re.I)

# ---------------------------------------------------------------- 规则表（测试 6 通过 ast 读取）

# (源函数名, 该函数原始英文报错里的独特片段) -> 规则编号。
# 这些片段对应 startlux_decide/prompt.py、startlux_decide/images.py 与 finetune/finetune_lora.py 里的字面量，
# 供 tests/test_check_data.py 的规则完整性守卫逐条核对。跳过的位置（超过 26 个选项、prompt 过长）下方另有标注。
RULES = {
    ("load_items", "invalid JSON"): "E01",
    ("load_items", 'each line needs a "questions" object'): "E06",
    ("load_items", "targets must be an object"): "E14",
    ("load_items", "target must be an object"): "E14",
    ("load_items", "target probabilities must be numbers"): "E14",
    ("load_items", "target has no mass"): "E14",
    ("load_items", "not a usable distribution"): "E14",
    ("load_items", "not one of the option ids"): "E13",
    ("load_items", "marks but the record has no images"): "E04",
    ("load_items", "image placeholders for"): "E04",
    ("from_systemone", "a choice question needs criteria"): "E08",
    ("from_systemone", "a score question needs criteria"): "E09",
    ("from_systemone", "unknown question type"): "E07",
    ("_render", "image placeholders for"): "E04",
    ("validate_row", "unknown question type"): "E07",
    ("validate_row", "instructions are required"): "E11",
    ("validate_row", "a question needs 2 to"): "E08",
    ("validate_row", "bad option id"): "E10",
    ("validate_row", "a score question needs 2 to"): "E09",
    ("place_images", "marks for"): "E04",
    ("smart_resize", "sides may differ by a factor"): "E05",
    ("decode", "an image must be a base64 string or a data URI"): "E03",
    ("load_media", "images must be a list"): "E03",
    ("load_media", "images per request"): "E03",
    ("load_media", "is not a readable image"): "E03",
}

# 同一批函数里不映射到规则的 raise：(函数, 报错片段) -> 中文理由。
# 已折叠的不可达分支：from_systemone 的 allow_none 分支（本检查器 allow_none=False）、validate_row 的
# "noul 选项必须是 true/false"（from_systemone 恒定产出这两个 id）、"score 层 id 必须是 0..n-1"
# （from_systemone 用 enumerate 生成 id，恒成立）、"重复选项 id"（JSON 对象键天然唯一）、
# "选项描述必须是字符串或 null"（from_systemone 已经用 value_text 归一）。
EXEMPT = {
    ("from_systemone", "allow_none adds the option"):
        "allow_none=False，这个分支不会被走到",
    ("validate_row", "the state must be a string after rendering"):
        "调用方永远传入 state_text() 的字符串结果",
    ("validate_row", "an option description must be a string or null"):
        "from_systemone 用 value_text 生成 criterion，只会是 str 或 None",
    ("validate_row", "duplicate option id"):
        "选项来自 JSON 对象键或 enumerate，天然唯一",
    ("validate_row", "yes/no options must be"):
        "from_systemone 对 noul 恒产出 true/false 两个 id",
    ("load_media", "this server cannot read images"):
        "本模块在导入时已强制要求 Pillow，走不到这里",
    ("load_items", "{e}"):
        "包装内层异常（真正原因由 place_images/load_media/from_systemone/plain_prompt 映射）",
    ("_record_media", "{e}"):
        "包装 load_media / place_images 的异常，真正原因由 E03/E04 映射",
    ("check_tokenizer", "the tokenizer's chat template lacks the thinking-off assistant prefix"):
        "描述模型目录的聊天模板，不是数据；训练器启动时已经检查",
    ("check_tokenizer", "is not a separate single token after the prompt"):
        "描述模型目录的分词器，不是数据；训练器启动时已经检查",
    ("check_tokenizer", "the letter token ids are not distinct"):
        "描述模型目录的分词器，不是数据；训练器启动时已经检查",
}

# 训练器里的跳过位置（不是 raise，但属于拒绝）：load_items 的 choice > MAX_OPTIONS -> E08，
# load_items 的 effective length > max_length -> E18。

RULE_INFO = {
    "E01": ("trainer", "行不是合法 JSON 对象"),
    "E02": ("contract", "缺少证据 state"),
    "E03": ("trainer", "图片无法读取或形状不对"),
    "E04": ("trainer", "图片标记与图片数量不一致"),
    "E05": ("trainer", "图片长宽比超过 200 倍"),
    "E06": ("trainer", "questions 结构不对"),
    "E07": ("trainer", "未知问题类型"),
    "E08": ("trainer", "choice 的 criteria 或选项数不对"),
    "E09": ("trainer", "score 的 criteria 或层数不对"),
    "E10": ("trainer", "选项 id 不合法"),
    "E11": ("trainer", "choice/score 缺少 instructions"),
    "E12": ("contract", "问题没有 target"),
    "E13": ("trainer", "target 的键不是选项 id"),
    "E14": ("trainer", "target 的值或总质量不合法"),
    "E15": ("contract", "target 含负数"),
    "E16": ("contract", "dev 集最高概率并列"),
    "E17": ("contract", "train/dev 数据重叠"),
    "E18": ("trainer", "prompt 有效长度超限"),
    "E19": ("contract", "target 含非有限数字"),
    "W01": ("observation", "同一个问题键的规格不一致"),
    "W02": ("observation", "问题里的选项从未作为答案出现"),
    "W03": ("observation", "数据观察项"),
    "W04": ("observation", "dev 集某类型问题过少"),
}


class InternalError(Exception):
    """检查器自身的问题（不是数据问题）。"""


class _PlaceholderMismatch(Exception):
    """聊天模板写出的图片占位符数量与图片数不一致。"""

    def __init__(self, got, expected):
        super().__init__(f"{got} placeholders for {expected} images")
        self.got = got
        self.expected = expected


class _ImageAspectError(ValueError):
    """图片长宽比超过 200 倍（消息是面向用户的中文）。"""


class TokenizerError(Exception):
    """--tokenizer 的模型目录/聊天模板/分词器无法处理数据（不是检查器 bug）。"""


def _finding(rule, path, line, question, message, fix):
    tag = RULE_INFO[rule][0]
    return {"rule": rule, "kind": rule[0], "tag": tag, "file": path, "line": line, "question": question,
            "message": message, "fix": fix}


def _rule_for(func, message):
    for (name, needle), rule in RULES.items():
        if name == func and needle in message:
            return rule
    return None


def _shape_message(rule, error):
    """from_systemone 拒绝一个 spec 时的中文发现项（原始英文只作为可操作的细节附在末尾）。"""
    if rule == "E07":
        return f"type 只能是 choice/noul/bool/score（原始信息：{error}）"
    if rule == "E08":
        return f"choice 的 criteria 必须是对象或列表（原始信息：{error}）"
    if rule == "E09":
        return f"score 的 criteria 必须是列表或对象（原始信息：{error}）"
    return f"问题无法渲染（原始信息：{error}）"


def _validate_message(rule, error):
    """validate_row 拒绝一个已渲染问题时的中文发现项。"""
    if rule == "E07":
        return f"type 只能是 choice/noul/bool/score（原始信息：{error}）"
    if rule == "E08":
        return f"choice 的选项数不在 2..{MAX_OPTIONS}（原始信息：{error}）"
    if rule == "E09":
        return f"score 的档位数不在 2..{MAX_LEVELS}（原始信息：{error}）"
    if rule == "E10":
        return f"选项 id 不合法：必须是非空、单行、不超过 200 字符的字符串（原始信息：{error}）"
    if rule == "E11":
        return "choice/score 必须写非空的 instructions"
    return f"问题无法通过校验（原始信息：{error}）"


# ---------------------------------------------------------------- 逐字复制的渲染逻辑

def annotate_indices(x, min_len=ANNOTATE_MIN):
    """长数组带上元素下标，让模型可以按位置引用条目。"""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                     else {"_index": i, "value": annotate_indices(v, min_len)}) for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def state_text(state):
    """证据文本：字符串原样返回，其它 JSON 值返回带数组下标的紧凑 JSON。"""
    if isinstance(state, str):
        return state
    if state is None or state == {} or state == []:
        return ""
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def value_text(v):
    """选项描述或指令的文本（空则为 None）。"""
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.strip()
    return s or None


def _norm(s):
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


def validate_row(row):
    """检查一个已渲染的问题并返回选项 id；有问题时抛 ValueError。"""
    t = row.get("type")
    if t not in TYPES:
        raise ValueError(f"unknown question type {t!r}")
    if not isinstance(row.get("state"), str):
        raise ValueError("the state must be a string after rendering")
    if not isinstance(row.get("instructions"), str) or not row["instructions"].strip():
        raise ValueError("instructions are required for choice and score questions")
    opts = row.get("options")
    if not isinstance(opts, list) or not 2 <= len(opts) <= MAX_OPTIONS:
        got = len(opts) if isinstance(opts, list) else opts
        raise ValueError(f"a question needs 2 to {MAX_OPTIONS} options in one pass, got {got!r}")
    ids = []
    for o in opts:
        i = o.get("id")
        if not isinstance(i, str) or not i.strip() or "\n" in i or "\r" in i or len(i) > 200:
            raise ValueError(f"bad option id {i!r}: ids must be non-empty single-line strings of at most 200 "
                             "characters")
        c = o.get("criterion")
        if c is not None and not isinstance(c, str):
            raise ValueError("an option description must be a string or null")
        ids.append(i)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate option id")
    if t == "noul" and set(ids) != {"true", "false"}:
        raise ValueError("yes/no options must be 'true' and 'false'")
    if t == "score" and not 2 <= len(ids) <= MAX_LEVELS:
        raise ValueError(f"a score question needs 2 to {MAX_LEVELS} levels, got {len(ids)}")
    return ids


def option_lines(row):
    """一个已渲染问题的带字母选项行，顺序与选项一致。"""
    order = [o["id"] for o in row["options"]]
    crit = {o["id"]: o.get("criterion") for o in row["options"]}
    t = row["type"]
    extra = len(order) - 1 if row.get("none_option") else None      # position of the extra option
    listed = order if extra is None else order[:extra]
    hide = t == "choice" and all(_BARE.match(i) for i in listed) and all(crit[i] for i in listed)
    lines = []
    for k, i in enumerate(order):
        name = ("yes" if i == "true" else "no") if t == "noul" else i
        c = crit[i]
        if hide and k != extra:
            body = c
        elif c is None or not c.strip() or _norm(c) == _norm(name):
            body = name
        else:
            body = f"{name}: {c}"
        lines.append(f"{LETTERS[k]}) {body}")
    return lines


def messages(row):
    """一个已渲染问题的对话消息（system 与 user 两轮）。"""
    validate_row(row)
    state = row["state"] if row["state"].strip() else "(none)"
    content = ("Evidence:\n" + state + "\n\nQuestion: " + row["instructions"].strip() + "\nOptions:\n"
               + "\n".join(option_lines(row)))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def score_keys(criteria):
    """score 问题的层顺序：数字键升序，否则按原顺序。"""
    try:
        return sorted(criteria, key=lambda k: float(k))
    except (TypeError, ValueError):
        return list(criteria)


def from_systemone(state, spec, allow_none=False):
    """一个 /v1/systemone 问题 -> 已渲染问题。score 层 id 为 "0".."n-1"，最底层在前。"""
    t = spec.get("type", "choice")
    t = "noul" if t == "bool" else t
    crit = spec.get("criteria", spec.get("options"))
    ins = value_text(spec.get("instructions", spec.get("question"))) or ""
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict):
            raise ValueError("a choice question needs criteria: {option: description or null} or a list of options")
        opts = [{"id": str(k), "criterion": value_text(v)} for k, v in crit.items()]
        if allow_none:
            if any(_norm(o["id"]) == _norm("None of the above") for o in opts):
                raise ValueError("allow_none adds the option \"None of the above\" itself: remove the option of "
                                 "that name or leave allow_none out")
            opts.append({"id": "None of the above", "criterion": None})
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in score_keys(crit)]
        if not isinstance(crit, (list, tuple)):
            raise ValueError("a score question needs criteria: a list of levels, lowest first")
        opts = [{"id": str(i), "criterion": value_text(c)} for i, c in enumerate(crit)]
    elif t == "noul":
        c = crit if isinstance(crit, dict) else {}
        tr, fa = c.get("true", c.get(True)), c.get("false", c.get(False))
        opts = [{"id": "true", "criterion": value_text(tr)}, {"id": "false", "criterion": value_text(fa)}]
        if not ins:
            ins = NOUL_INSTRUCTIONS
    else:
        raise ValueError(f"unknown question type {t!r}: use choice, noul or score")
    row = {"type": t, "state": state_text(state), "instructions": ins, "options": opts}
    if t == "choice" and allow_none:
        row["none_option"] = True
    return row


# ---------------------------------------------------------------- 逐字复制的图片逻辑

def place_images(text, n):
    """state 文本里 n 张图片的位置（与 startlux_decide.images 一致）。"""
    marks = text.count(IMAGE_MARK)
    if marks == n:
        return text.replace(IMAGE_MARK, SLOT)
    if marks:
        raise ValueError(f"the state has {marks} {IMAGE_MARK} marks for {n} images")
    head = SLOT if n == 1 else "\n".join(f"Image {k + 1}: {SLOT}" for k in range(n))
    return head + ("\n\n" + text if text.strip() else "")


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """模型图片处理器会把图片缩放到的尺寸（Qwen2-VL 的 smart_resize）。"""
    if max(height, width) / min(height, width) > ASPECT_LIMIT:
        raise ValueError(f"an image's sides may differ by a factor of 200 at most, not {height} x {width}")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def decode(value):
    """base64 字符串或 data URI -> (图片字节, MIME 类型)。"""
    if not isinstance(value, str):
        raise ValueError(f"an image must be a base64 string or a data URI, not {type(value).__name__}")
    m = _DATA_URI.match(value)
    mime = (m.group(1) or "").lower() if m else ""
    payload = value[m.end():] if m else value
    try:
        return base64.b64decode(re.sub(r"\s+", "", payload), validate=True), mime
    except (binascii.Error, ValueError):
        raise ValueError("an image must be a base64 string or a data URI") from None


def image_info(values, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, max_images=MAX_IMAGES):
    """解码一份图片列表 -> (raw, sizes, tokens)。报错文字与 startlux_decide.images.load_media 相同。"""
    if not isinstance(values, (list, tuple)):
        raise ValueError("images must be a list of base64 strings or data URIs")
    if len(values) > max_images:
        raise ValueError(f"at most {max_images} images per request, not {len(values)}")
    raw, sizes, tokens = [], [], []
    for k, value in enumerate(values):
        data, mime = decode(value)
        try:
            img = Image.open(io.BytesIO(data))
            fmt = (img.format or "").lower()
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"image {k + 1} is not a readable image ({e})") from None
        try:
            h, w = smart_resize(img.height, img.width, factor, min_pixels, max_pixels)
        except ValueError:
            raise _ImageAspectError(f"图片长宽比超过 200 倍（宽 {img.width}，高 {img.height}）") from None
        raw.append((data, mime or (f"image/{fmt}" if fmt else "")))
        sizes.append((h, w))
        tokens.append((h // factor) * (w // factor))
    return raw, sizes, tokens


def multimodal_messages(msgs):
    """内容里含 SLOT 的消息 -> 带文本/图片分段的消息。"""
    out = []
    for m in msgs:
        content = m["content"]
        if not isinstance(content, str):
            out.append(m)
            continue
        parts = []
        for k, piece in enumerate(content.split(SLOT)):
            if k:
                parts.append({"type": "image"})
            if piece:
                parts.append({"type": "text", "text": piece})
        out.append({**m, "content": parts})
    return out


# ---------------------------------------------------------------- 长度与泄漏键

def effective_length(tokenizer, row, image_tokens):
    """(有效长度, prompt)：镜像 startlux_decide.prompt._render 与 finetune_lora.load_items 的算法。"""
    msgs = messages(row)
    slots = sum(m["content"].count(SLOT) for m in msgs if isinstance(m.get("content"), str))
    chat = multimodal_messages(msgs) if slots else msgs
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if slots and prompt.count(IMAGE_PLACEHOLDER) != slots:
        raise _PlaceholderMismatch(prompt.count(IMAGE_PLACEHOLDER), slots)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    length = len(ids) + sum(t - 1 for t in image_tokens)
    return length, prompt


def _image_digests(rec):
    """记录里每张图片字节的 sha256；有图片读不出时返回 None（该记录不参与泄漏键）。"""
    values = rec.get("images")
    if values is None or values == []:
        return []
    if not isinstance(values, list):
        return None
    digests = []
    for value in values:
        try:
            raw, _ = decode(value)
        except ValueError:
            return None
        digests.append(hashlib.sha256(raw).hexdigest())
    return sorted(digests)


def _leak_key(rec):
    """泄漏键：渲染后的 state_text + 排序后的图片字节摘要。"""
    digests = _image_digests(rec)
    if digests is None:
        return None
    return {"state": state_text(rec.get("state")), "images": digests}


# ---------------------------------------------------------------- 设置与单条记录校验

class Settings:
    """一次运行的阈值与图片参数；默认值与训练器一致。"""

    def __init__(self, max_length=DEFAULT_MAX_LENGTH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                 image_factor=FACTOR, max_images=MAX_IMAGES, image_factor_source="default"):
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_factor = image_factor
        self.max_images = max_images
        self.image_factor_source = image_factor_source

    def as_dict(self):
        return {"max_length": self.max_length, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
                "image_factor": self.image_factor, "max_images": self.max_images}

    @staticmethod
    def defaults_dict():
        return {"max_length": DEFAULT_MAX_LENGTH, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                "image_factor": FACTOR, "max_images": MAX_IMAGES, "max_options": MAX_OPTIONS,
                "max_levels": MAX_LEVELS, "annotate_min": ANNOTATE_MIN, "tie_eps": TIE_EPS,
                "aspect_limit": ASPECT_LIMIT}

    def overrides(self):
        base = {"max_length": DEFAULT_MAX_LENGTH, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                "image_factor": FACTOR, "max_images": MAX_IMAGES}
        return {k: v for k, v in self.as_dict().items() if v != base[k]}


def _check_question(qres, spec, target, state, image_count, image_tokens, path, line, settings, tokenizer, role,
                    model_folder=None):
    """校验一个问题，把发现项写进 qres。"""
    key = qres["key"]

    def add(rule, message, fix, question=key):
        qres["findings"].append(_finding(rule, path, line, question, message, fix))

    if not isinstance(spec, dict):
        add("E06", f"问题 {key!r} 的规格不是 JSON 对象", "把该问题写成 {\"type\": ..., \"instructions\": ...} 对象")
        return
    raw_type = spec.get("type", "choice")
    if raw_type not in ("choice", "noul", "score", "bool"):
        add("E07", f"type 只能是 choice/noul/bool/score（当前值：{raw_type!r}）",
            "把 type 改成 choice、noul、score（bool 等同于 noul），或删掉 type 使用默认的 choice")
        return
    if target is None:
        add("E12", f"问题 {key!r} 在 targets 里没有条目", "在 targets 里给这个问题加一个 {选项id: 概率} 对象，"
                                                          "或删掉这个问题")
        return
    if not isinstance(target, dict):
        add("E14", f"问题 {key!r} 的 target 不是对象", "把 target 写成 {选项id: 概率} 的对象")
        return

    try:
        row = from_systemone(state, spec)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        rule = _rule_for("from_systemone", str(e))
        if rule is None:
            raise InternalError(f"from_systemone 报错未映射到规则：{e}") from None
        add(rule, _shape_message(rule, e), _fix_for(rule))
        return
    qres["row"] = row
    qres["type"] = row["type"]
    order = [o["id"] for o in row["options"]]
    qres["options"] = order

    bad_ids = [i for i in order if not isinstance(i, str) or not i.strip() or "\n" in i or "\r" in i or len(i) > 200]
    if bad_ids:
        add("E10", "选项 id 不合法：必须是非空、单行、不超过 200 字符的字符串（问题 id："
                   f"{bad_ids!r}）", "选项 id 必须是非空、单行、不超过 200 字符的字符串")
    if row["type"] in ("choice", "score") and not row["instructions"].strip():
        add("E11", "choice/score 必须写非空的 instructions", "给 choice/score 问题写一句非空的 instructions")
    if row["type"] == "choice" and not 2 <= len(order) <= MAX_OPTIONS:
        add("E08", f"choice 的选项数不在 2..{MAX_OPTIONS}：当前有 {len(order)} 个",
            "把选项数改到 2 到 26 之间；更宽的列表请拆成多组")
    if row["type"] == "score" and not 2 <= len(order) <= MAX_LEVELS:
        add("E09", f"score 的档位数不在 2..{MAX_LEVELS}：当前有 {len(order)} 个",
            f"把 score 的等级数改到 2 到 {MAX_LEVELS} 之间")
    try:
        validate_row(row)
    except ValueError as e:
        rule = _rule_for("validate_row", str(e))
        if rule is None:
            raise InternalError(f"validate_row 报错未映射到规则：{e}") from None
        # score 的选项行会先触发 validate_row 的通用"2..26 个选项"检查；层数问题已经由 E09 表达，
        # 这里不再补一条 E08，避免同一件事被记成两个规则。
        if not (rule == "E08" and row["type"] == "score") and \
                not any(f["rule"] == rule for f in qres["findings"]):
            add(rule, _validate_message(rule, e), _fix_for(rule))

    unknown = [k for k in target if k not in order]
    if unknown:
        add("E13", f"问题 {key!r} 的 target 含不属于选项的键：{unknown!r}",
            f"把这些键改成选项 id 之一：{order!r}")
    values = None
    try:
        values = [float(target.get(o, 0.0)) for o in order]
    except (TypeError, ValueError, ArithmeticError):
        add("E14", f"问题 {key!r} 的 target 概率不是数字或超出数值范围", "把每个 target 值写成有限数字")
    finite = values is not None and all(math.isfinite(v) for v in values)
    if values is not None and not finite:
        add("E19", "target 值必须是有限数字（NaN/Infinity 会让训练 loss 变成 NaN）",
            "把 NaN/Infinity 改成有限概率")
    if finite:
        if any(v < 0 for v in values):
            add("E15", f"问题 {key!r} 的 target 含负数：{values!r}", "概率不能为负；请提供非负的分布")
        try:
            total = math.fsum(values)
        except ArithmeticError:
            total = None
        if total is None:
            add("E14", f"问题 {key!r} 的 target 总质量超出数值范围，无法归一",
                "把概率改小到有限且可求和的范围（例如总和接近 1）")
        elif total <= 0:
            add("E14", f"问题 {key!r} 的 target 总质量为零", "至少给一个选项一个正概率")
        else:
            qres["target"] = [v / total for v in values]

    if role == "dev" and finite:
        position = {option: k for k, option in enumerate(order)}
        listed = [(k, values[position[k]]) for k in target if k in position]
        if listed:
            top = max(value for _option, value in listed)
            tied = [option for option, value in listed if top - value <= TIE_EPS]
            if len(tied) > 1:
                add("E16", f"问题 {key!r} 的最高概率并列：{tied!r}", "dev 集的 gold 必须唯一；去掉并列的最高概率")

    # 没有分词器时，图片 token 数是有效长度的下界：prompt 至少有每个图片占位符一个 token。
    # 仅当这个下界已经越界才报 E18；否则保持"未验证"。
    if tokenizer is None and image_tokens and sum(image_tokens) > settings.max_length:
        add("E18", f"仅图片就占 {sum(image_tokens)} 个 token，已超过 --max-length {settings.max_length}"
                   "（文本部分还会更多）",
            "减少图片数量、缩小图片（--max-pixels），或给训练和验收同时设更大的 --max-length")

    if tokenizer is not None and qres["row"] is not None and not qres["findings"]:
        try:
            length, _prompt = effective_length(tokenizer, row, image_tokens)
        except _PlaceholderMismatch as e:
            add("E04", f"图片占位符数量不一致：聊天模板写出 {e.got} 个占位符，记录有 {e.expected} 张图片",
                "让 state 里的 <image> 标记数与 images 数量一致")
        except Exception as e:
            raise TokenizerError(f"--tokenizer {model_folder or '<模型目录>'} 的聊天模板/分词器无法处理这条记录："
                                 f"{e}") from None
        else:
            qres["length"] = length
            if length > settings.max_length:
                add("E18", f"问题 {key!r} 的有效长度 {length} 超过 --max-length {settings.max_length}",
                    "缩短 state/instructions，或给训练加上更大的 --max-length（两边必须一致）")


def _fix_for(rule):
    return {
        "E07": "把 type 改成 choice、noul、score（bool 等同于 noul），或删掉 type 使用默认的 choice",
        "E08": "choice 的 criteria 必须是 {选项id: 描述} 对象或选项列表，选项数 2..26",
        "E09": f"score 的 criteria 必须是等级列表或 {{层id: 描述}} 对象，层数 2..{MAX_LEVELS}",
        "E10": "选项 id 必须是非空、单行、不超过 200 字符的字符串",
        "E03": "把每张图片写成可解码的 base64 字符串或 data URI（Pillow 能读的 PNG/JPEG 等）",
        "E04": "让 state 里的 <image> 标记数与 images 数量一致，或删掉标记",
        "E05": "裁剪或等比缩放图片，让长边与短边相差不超过 200 倍",
        "E11": "给 choice/score 问题写一句非空的 instructions",
        "E14": "把 target 写成 {选项id: 概率} 的对象，且至少一个选项的概率为正",
    }.get(rule, "按规则说明修改该字段")


def check_file(path, settings=None, tokenizer=None, role="train", context=None, against=None, model_folder=None):
    """校验一个 JSONL 文件，返回 {path, records, findings, counts}；不抛数据错误。"""
    settings = settings or Settings()
    context = context if context is not None else {"signatures": {}, "option_lines": {}, "target_keys": set()}
    against = against or {}
    result = {"path": path, "records": [], "findings": [], "counts": {}}
    seen_states = {}
    state_lengths = []
    type_counts = {}

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec_findings = []
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                rec_findings.append(_finding("E01", path, line_no, None, f"不是合法 JSON：{e}",
                                             "修好这一行的 JSON 语法"))
                result["records"].append(_record_result(path, line_no, None, "", None, [], [], rec_findings))
                continue
            if not isinstance(rec, dict):
                rec_findings.append(_finding("E01", path, line_no, None, "这一行不是 JSON 对象",
                                             "每行写一个 {...} 对象"))
                result["records"].append(_record_result(path, line_no, None, "", None, [], [], rec_findings))
                continue

            unknown_keys = [k for k in rec if k not in ALLOWED_KEYS]
            if unknown_keys:
                rec_findings.append(_finding("W03", path, line_no, None, f"未知顶层键：{unknown_keys!r}（训练器会忽略）",
                                             "删掉这些键，或确认它们确实不需要"))

            state = rec.get("state")
            text = state_text(state)
            if "state" not in rec or state is None or state == {} or state == [] or not text.strip():
                rec_findings.append(_finding("E02", path, line_no, None, "state 缺失或渲染后为空",
                                             "给每条记录提供非空的 state（字符串或 JSON 对象）"))
            elif role == "dev" and not isinstance(state, (str, dict)):
                rec_findings.append(_finding("E02", path, line_no, None,
                                             "dev 集的 state 必须是字符串或对象，评测转换只接受这两种",
                                             "把 dev 记录的 state 改成字符串或对象；列表/数字等评测转换不接受"))
            if text.strip():
                state_lengths.append(len(text))
                if text in seen_states:
                    rec_findings.append(_finding("W03", path, line_no, None,
                                                 f"state 与第 {seen_states[text]} 行重复（判重键是渲染后的 state 文本，不含图片）",
                                                 "检查是否误复制了记录"))
                else:
                    seen_states[text] = line_no

            image_raw, image_sizes, image_tokens = [], [], []
            images = rec.get("images")
            if images is not None and images != []:
                if not isinstance(images, list):
                    rec_findings.append(_finding("E03", path, line_no, None, "images 不是列表",
                                                 "把 images 写成 base64 字符串或 data URI 的列表"))
                elif len(images) > settings.max_images:
                    rec_findings.append(_finding("E03", path, line_no, None,
                                                 f"images 有 {len(images)} 张，超过上限 {settings.max_images}",
                                                 f"把每条记录的图片数降到 {settings.max_images} 张以内"))
                else:
                    try:
                        image_raw, image_sizes, image_tokens = image_info(
                            images, settings.image_factor, settings.min_pixels, settings.max_pixels,
                            settings.max_images)
                    except _ImageAspectError as e:
                        rec_findings.append(_finding("E05", path, line_no, None, str(e), _fix_for("E05")))
                    except ValueError as e:
                        rec_findings.append(_finding(
                            "E03", path, line_no, None,
                            f"图片无法读取：必须是 base64 字符串或 data URI，且能被 Pillow 解码（原始信息：{e}）",
                            _fix_for("E03")))

            placed = text
            if not any(f["rule"] in ("E03", "E05") for f in rec_findings):
                marks = text.count(IMAGE_MARK)
                if marks and marks != len(image_raw):
                    rec_findings.append(_finding(
                        "E04", path, line_no, None,
                        f"图片标记数与图片数量不一致：state 里有 {marks} 个 <image> 标记，记录有 "
                        f"{len(image_raw)} 张图片",
                        "让 state 里的 <image> 标记数与 images 数量一致，或删掉标记"))
                else:
                    placed = place_images(text, len(image_raw))

            questions_raw = rec.get("questions")
            if not isinstance(questions_raw, dict):
                rec_findings.append(_finding("E06", path, line_no, None, "questions 缺失或不是对象",
                                             "把 questions 写成 {问题键: 问题规格} 的对象"))
                questions_raw = {}
            targets_raw = rec.get("targets") or {}
            if not isinstance(targets_raw, dict):
                rec_findings.append(_finding("E14", path, line_no, None, "targets 不是对象",
                                             "把 targets 写成 {问题键: {选项id: 概率}} 的对象"))
                targets_raw = {}

            leak_key = _leak_key(rec)
            if leak_key is not None and leak_key_key(leak_key) in against:
                other = against[leak_key_key(leak_key)]
                rec_findings.append(_finding("E17", path, line_no, None,
                                             f"与 {other} 的 state 和图片完全相同（train/dev 泄漏）",
                                             "把这条记录从其中一个 split 里删掉"))

            questions = []
            for key, spec in questions_raw.items():
                qres = {"key": key, "spec": spec, "row": None, "options": [], "target": None, "type": None,
                        "length": None, "findings": [], "trainable": False}
                if not key:
                    qres["findings"].append(_finding(
                        "W03", path, line_no, None, "问题键为空字符串，无法在报告和 targets 里稳定引用",
                        "给问题一个非空键名（训练器仍会训练它，但报告和对齐会失去稳定引用）"))
                target = targets_raw.get(key)
                _check_question(qres, spec, target, placed, len(image_raw), image_tokens, path, line_no,
                                settings, tokenizer, role, model_folder)
                if isinstance(spec, dict):
                    raw_type = spec.get("type", "choice")
                    raw_type = "noul" if raw_type == "bool" else raw_type
                    criteria = spec.get("criteria", spec.get("options"))
                    if raw_type == "noul" and isinstance(criteria, list):
                        qres["findings"].append(_finding("W03", path, line_no, key,
                                                         "noul 的 criteria 列表会被渲染器丢弃",
                                                         "noul 只需要 instructions，true/false 自动生成"))
                    elif raw_type == "noul" and isinstance(criteria, dict):
                        extra = [k for k in criteria if k not in ("true", "false")]
                        if extra:
                            qres["findings"].append(_finding("W03", path, line_no, key,
                                                             f"noul 的 criteria 里这些键会被渲染器丢弃：{extra!r}",
                                                             "noul 只会读取 true/false 两个键"))
                if qres["row"] is not None:
                    sig = (qres["type"], qres["row"]["instructions"], tuple(qres["options"]))
                    context["signatures"].setdefault(key, []).append(
                        {"file": path, "line": line_no, "signature": sig})
                    if isinstance(target, dict):
                        context["target_keys"].update((key, option) for option in target)
                        for option in qres["options"]:
                            context["option_lines"].setdefault((key, option), []).append((path, line_no, key))
                if qres["type"]:
                    type_counts[qres["type"]] = type_counts.get(qres["type"], 0) + 1
                questions.append(qres)

            rec_blocked = any(f["kind"] == "E" for f in rec_findings)
            for qres in questions:
                qres["trainable"] = (not rec_blocked) and not any(f["kind"] == "E" for f in qres["findings"])
            result["records"].append(_record_result(path, line_no, rec, text, placed,
                                                    {"raw": image_raw, "sizes": image_sizes, "tokens": image_tokens},
                                                    questions, rec_findings))

    if role == "dev":
        for kind, count in sorted(type_counts.items()):
            if count < 200:
                result["findings"].append(_finding("W04", path, None, None,
                                                   f"dev 集里 {kind} 问题只有 {count} 个（<200）",
                                                   "增加该类型的 dev 问题，否则校准会回退到合并温度"))

    if state_lengths:
        median = statistics.median(state_lengths)
        if median > 0:
            for rec_res in result["records"]:
                if rec_res["record"] is not None and len(rec_res["state_text"]) > 4 * median:
                    result["findings"].append(_finding(
                        "W03", path, rec_res["line"], None,
                        f"state 长度 {len(rec_res['state_text'])} 超过中位数 {median:g} 的 4 倍",
                        "确认这条超长 state 是有意的"))

    for rec_res in result["records"]:
        result["findings"].extend(rec_res["findings"])
        for qres in rec_res["questions"]:
            result["findings"].extend(qres["findings"])

    questions_total = sum(len(r["questions"]) for r in result["records"])
    trainable = sum(1 for r in result["records"] for q in r["questions"] if q["trainable"])
    result["counts"] = {"records": len(result["records"]), "questions": questions_total,
                        "trainable": trainable, "blocked": questions_total - trainable,
                        "blocked_records": sum(1 for r in result["records"]
                                               if any(f["kind"] == "E" for f in r["findings"]))}
    return result


def _finding_line(finding):
    """终端里的一行发现项；消息本身已经以"问题"开头时不再重复问题键。"""
    where = finding["file"]
    if finding.get("line"):
        where += f":{finding['line']}"
    if finding.get("question") and not finding["message"].startswith("问题 "):
        where += f" 问题 {finding['question']!r}"
    return f"[{finding['rule']}] {where}: {finding['message']}"


def _record_result(path, line_no, rec, text, placed, images, questions, findings):
    return {"path": path, "line": line_no, "record": rec, "state_text": text, "placed": placed,
            "images": images, "questions": questions, "findings": findings}


def leak_key_key(key):
    return json.dumps(key, ensure_ascii=False, sort_keys=True)


def finalize_observations(context):
    """在全部文件读完后生成跨记录的 W01/W02。"""
    findings = []
    for key, seen in context["signatures"].items():
        distinct = {entry["signature"] for entry in seen}
        if len(distinct) > 1:
            lines = ", ".join(f"{entry['file']}:{entry['line']}" for entry in seen)
            first = seen[0]
            findings.append(_finding("W01", first["file"], first["line"], key,
                                     f"问题键 {key!r} 在不同记录里 type/instructions/选项不同，出现在 {lines}",
                                     "统一同一个问题键的规格，或给不同规格换一个键名"))
    for (key, option), seen in context["option_lines"].items():
        if (key, option) not in context["target_keys"]:
            first = seen[0]
            findings.append(_finding("W02", first[0], first[1], key,
                                     f"问题 {key!r} 的选项 {option!r} 从未作为答案出现",
                                     "确认这是有意给出的干扰项，否则补上标注"))
    return findings


# ---------------------------------------------------------------- 输出：canonical / render / report

def canonical_lines(results):
    """把通过校验的问题写成规范化记录。"""
    for file_result in results:
        for res in file_result["records"]:
            if res["record"] is None:
                continue
            if any(f["kind"] == "E" for f in res["findings"]):
                continue
            questions, targets = {}, {}
            for qres in res["questions"]:
                if not qres["trainable"] or qres["row"] is None:
                    continue
                row = qres["row"]
                key = qres["key"]
                questions[key] = {"type": row["type"], "instructions": row["instructions"],
                                  "options": [{"id": o["id"], "criterion": o.get("criterion")}
                                              for o in row["options"]]}
                targets[key] = {option: prob for option, prob in zip(qres["options"], qres["target"])}
            if not questions:
                continue
            images = [f"data:{mime or 'image/png'};base64,{base64.b64encode(raw).decode('ascii')}"
                      for raw, mime in res["images"]["raw"]]
            yield {"id": res["record"].get("id", f"line-{res['line']}"), "state": res["state_text"],
                   "images": images, "questions": questions, "targets": targets}


def _sample(records, n):
    if n >= len(records):
        return records[:n]
    if n <= 1:
        return records[:1]
    picks = []
    for i in range(n):
        index = int(round(i * (len(records) - 1) / (n - 1)))
        if index not in picks:
            picks.append(index)
    return [records[i] for i in picks]


def render_lines(results, n):
    """返回要打印的行；每行不带换行符。"""
    records = [res for file_result in results for res in file_result["records"] if res["record"] is not None]
    out = []
    for res in _sample(records, n):
        out.append(f"===== {res['path']}:{res['line']} 的模型视角 =====")
        for qres in res["questions"]:
            if qres["row"] is None:
                continue
            try:
                msgs = messages(qres["row"])
            except ValueError:
                continue
            out.append(f"--- 问题 {qres['key']!r}（{qres['type']}，{len(qres['options'])} 个选项）---")
            out.append("[system]")
            out.append(msgs[0]["content"])
            out.append("[user]")
            out.append(msgs[1]["content"].replace(SLOT, IMAGE_PLACEHOLDER))
        if res["images"]["sizes"]:
            out.append(f"图片：{len(res['images']['sizes'])} 张")
            for k, (size, tokens) in enumerate(zip(res["images"]["sizes"], res["images"]["tokens"]), 1):
                out.append(f"  #{k} 缩放后 {size[0]}x{size[1]}，{tokens} 个 token")
        out.append("")
    return out


def _unverified(tokenizer_path, against_given, image_unverified=False):
    """没有跑过的规则：E17 需要 --against，E18 的文本部分需要 --tokenizer，图片部分还需要可确定的边长。"""
    out = []
    if not against_given:
        out.append("E17")
    if not tokenizer_path or image_unverified:
        out.append("E18")
    return out


def _not_applicable(role):
    """本次运行不适用的规则。E17 不用 --against 时属于"未验证"，不在这里。"""
    return ["E16", "W04"] if role == "train" else []


def _model_image_factor(folder):
    """模型目录 config.json 里的图片 token 边长；读不出时返回 None。"""
    try:
        with open(os.path.join(str(folder), "config.json"), encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    vision = config.get("vision_config")
    if not isinstance(vision, dict):       # vision_settings 不读顶层键
        return None
    patch, merge = vision.get("patch_size", 16), vision.get("spatial_merge_size", 2)
    try:
        factor = int(patch) * int(merge)
    except (TypeError, ValueError):
        return None
    return factor if factor > 0 else None


def _usage_error(message):
    print(f"用法错误：{message}", file=sys.stderr)
    return 2


def build_report(results, observations, settings, role, tokenizer_path, commit, against_given=False,
                 image_unverified=False):
    """组装 --report 的 JSON 对象（键见模块 docstring）。"""
    findings = [f for res in results for f in res["findings"]]
    blocking = [f for f in findings if f["kind"] == "E"]
    observations = [f for f in findings if f["kind"] == "W"] + list(observations)
    files = [{"path": res["path"], "records": res["counts"]["records"],
              "questions": res["counts"]["questions"], "trainable": res["counts"]["trainable"],
              "blocked": res["counts"]["blocked"], "blocked_records": res["counts"]["blocked_records"]}
             for res in results]
    return {
        "checker_version": CHECKER_VERSION,
        "commit": commit,
        "defaults": Settings.defaults_dict(),
        "settings": settings.as_dict(),
        "overrides": settings.overrides(),
        "role": role,
        "mode": "tokenizer" if tokenizer_path else "plain",
        "length_verified": bool(tokenizer_path) and not image_unverified,
        "unverified": _unverified(tokenizer_path, against_given, image_unverified),
        "not_applicable": _not_applicable(role),
        "image_factor_source": settings.image_factor_source,
        "model_folder": tokenizer_path or None,
        "files": files,
        "findings": blocking,
        "observations": observations,
    }


# ---------------------------------------------------------------- --emit-eval：逐字镜像 to_eval_data

def _eval_error(path, line, message):
    return ValueError(f"{path}:{line}: {message}")


def _eval_gold_label(path, line, key, spec, target):
    if not isinstance(spec, dict):
        raise _eval_error(path, line, f'question {key!r}: the question must be an object')
    if not isinstance(target, dict) or not target:
        raise _eval_error(path, line, f"question {key!r}: target must be a non-empty object mapping option ids to "
                                      "probabilities")
    values = {}
    for option, probability in target.items():
        try:
            values[option] = float(probability)
        except (TypeError, ValueError):
            raise _eval_error(path, line, f"question {key!r}: target probabilities must be numbers, not "
                                          f"{probability!r}") from None
    top = max(values.values())
    tied = [option for option, probability in values.items() if top - probability <= TIE_EPS]
    if len(tied) > 1:
        raise _eval_error(path, line, f"question {key!r}: tied target probabilities for options "
                                      f"{', '.join(map(repr, tied))}")
    option = tied[0]
    kind = spec.get("type", "choice")
    if kind == "score":
        try:
            return int(option)
        except (TypeError, ValueError):
            raise _eval_error(path, line, f"question {key!r}: a score target option must be a level id 0..n-1, "
                                          f"not {option!r}") from None
    if kind in ("noul", "bool"):
        if option not in ("true", "false"):
            raise _eval_error(path, line, f"question {key!r}: a yes/no target option must be 'true' or 'false', "
                                          f"not {option!r}")
        return option
    return option


def emit_eval(inputs, out_path):
    """-> (写出条数, 跳过问题数, 丢弃记录数)；输入有问题时抛带 file:line 的 ValueError。"""
    requests, skipped, dropped = [], 0, 0
    for path in inputs:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise _eval_error(path, n, f"invalid JSON: {e}") from None
                if not isinstance(rec, dict):
                    raise _eval_error(path, n, "each line must be a JSON object")
                if "state" not in rec:
                    raise _eval_error(path, n, 'each line needs a "state"')
                if not isinstance(rec["state"], (str, dict)):
                    raise _eval_error(path, n, "state must be a string or an object")
                if not isinstance(rec.get("questions"), dict):
                    raise _eval_error(path, n, 'each line needs a "questions" object')
                targets = rec.get("targets")
                if targets is None:
                    targets = {}
                elif not isinstance(targets, dict):
                    raise _eval_error(path, n, "targets must be an object")
                questions, gold = {}, {}
                for key, spec in rec["questions"].items():
                    target = targets.get(key)
                    if target is None:
                        skipped += 1
                        continue
                    questions[key] = spec
                    gold[key] = _eval_gold_label(path, n, key, spec, target)
                if not questions:
                    dropped += 1
                    continue
                item = {"id": rec.get("id", f"line-{n}"), "state": rec["state"], "questions": questions,
                        "gold": gold}
                if "images" in rec:
                    item["images"] = rec["images"]
                requests.append(item)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in requests:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(requests), skipped, dropped


# ---------------------------------------------------------------- CLI

def _detect_commit(directory):
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True,
                              timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _load_tokenizer(path):
    from transformers import AutoTokenizer       # 只在 --tokenizer 模式按需导入
    return AutoTokenizer.from_pretrained(path)


def _parser():
    ap = argparse.ArgumentParser(
        description="StartLux-Decision 微调数据自检器（单文件，只依赖标准库和 Pillow）。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", metavar="FILE", help="要检查的 JSONL 文件（可给多个）")
    ap.add_argument("--render", type=int, metavar="N", help="打印 N 条记录的模型视角（system/user 正文与图片尺寸）")
    ap.add_argument("--canonical", metavar="OUT", help="把通过校验的记录写成规范化 JSONL")
    ap.add_argument("--report", metavar="PATH", help="把机器可读报告写成 JSON")
    ap.add_argument("--against", action="append", default=[], metavar="OTHER",
                    help="另一个 split；检查 state+图片 是否与本文件重叠（可重复）")
    ap.add_argument("--tokenizer", metavar="MODEL_DIR", help="模型目录：用真实分词器校验 prompt 有效长度")
    ap.add_argument("--role", choices=["train", "dev"], default="train", help="train 或 dev（默认 train）")
    ap.add_argument("--emit-eval", metavar="OUT", help="dev 角色：在原始记录上生成 eval JSONL")
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="有效 prompt 长度上限（默认 8192）")
    ap.add_argument("--min-pixels", type=int, default=MIN_PIXELS, help=f"图片最小像素数（默认 {MIN_PIXELS}）")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS, help=f"图片最大像素数（默认 {MAX_PIXELS}）")
    ap.add_argument("--image-factor", type=int, default=None,
                    help=f"图片 token 边长（默认 {FACTOR}；显式传入会记录为 cli 来源）")
    ap.add_argument("--max-images", type=int, default=MAX_IMAGES, help=f"每条记录的图片上限（默认 {MAX_IMAGES}）")
    ap.add_argument("--commit", help="写进报告的 commit；默认自动读取 git，拿不到时为 unknown")
    return ap


def run(args, tokenizer):
    explicit_factor = args.image_factor is not None
    factor_unknown = False
    if explicit_factor:
        image_factor, image_factor_source = args.image_factor, "cli"
    elif args.tokenizer:
        found = _model_image_factor(args.tokenizer)
        if found is None:
            image_factor, image_factor_source, factor_unknown = FACTOR, "default", True
        else:
            image_factor, image_factor_source = found, "model"
    else:
        image_factor, image_factor_source = FACTOR, "default"
    settings = Settings(max_length=args.max_length, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                        image_factor=image_factor, max_images=args.max_images,
                        image_factor_source=image_factor_source)
    if args.emit_eval and args.role != "dev":
        return _usage_error("--emit-eval 只能在 --role dev 下使用")
    if args.emit_eval and len(args.files) != 1:
        return _usage_error("--emit-eval 只接受一个输入文件")

    against = {}
    for other in args.against:
        try:
            with open(other, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    key = _leak_key(rec)
                    if key is not None:
                        against[leak_key_key(key)] = f"{other}:{n}"
        except OSError as e:
            return _usage_error(f"无法读取 --against 文件 {other}：{e}")

    context = {"signatures": {}, "option_lines": {}, "target_keys": set()}
    try:
        results = [check_file(path, settings=settings, tokenizer=tokenizer, role=args.role,
                              context=context, against=against, model_folder=args.tokenizer)
                   for path in args.files]
    except OSError as e:
        return _usage_error(f"无法读取输入文件：{e}")
    cross_observations = finalize_observations(context)
    observations = cross_observations + [f for res in results for f in res["findings"] if f["kind"] == "W"]

    if args.render:
        for line in render_lines(results, args.render):
            print(line)

    if args.canonical:
        try:
            with open(args.canonical, "w", encoding="utf-8") as f:
                for item in canonical_lines(results):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except OSError as e:
            return _usage_error(f"无法写出 --canonical 文件 {args.canonical}：{e}")

    findings = [f for res in results for f in res["findings"]]
    blocking = [f for f in findings if f["kind"] == "E"]

    emit_counts = None
    emit_error = None
    emit_skipped = False
    if args.emit_eval:
        if blocking:
            emit_skipped = True
        else:
            try:
                emit_counts = emit_eval(args.files, args.emit_eval)
            except OSError as e:
                return _usage_error(f"无法写出 --emit-eval 文件 {args.emit_eval}：{e}")
            except ValueError as e:
                emit_error = e

    for finding in blocking:
        print(_finding_line(finding))
        if finding["fix"]:
            print(f"    修复：{finding['fix']}")

    for obs in observations:
        print(_finding_line(obs))

    total_questions = sum(res["counts"]["questions"] for res in results)
    total_trainable = sum(res["counts"]["trainable"] for res in results)
    print(f"共 {len(results)} 个文件：{total_questions} 个问题，{total_trainable} 个可训练，"
          f"{total_questions - total_trainable} 个被阻断；{len(blocking)} 条阻断发现，"
          f"{len(observations)} 条观察项。")
    if tokenizer is None:
        print("长度未完全验证（未使用 --tokenizer）：E18 只检查了图片 token 下界，文本部分未检查。")
    has_images = any(res["images"]["tokens"] for file_result in results for res in file_result["records"])
    image_unverified = factor_unknown and has_images
    unverified = _unverified(args.tokenizer, bool(args.against), image_unverified)
    if unverified:
        print(f"未验证的规则：{', '.join(unverified)}")
    not_applicable = _not_applicable(args.role)
    if not_applicable:
        print(f"不适用（本运行）：{', '.join(not_applicable)}")
    if has_images:
        if factor_unknown:
            print("数据含图片，但无法从模型目录 config.json 确定图片 token 边长；"
                  "E18 的图片部分未验证（当前按默认 32 估算）。")
        elif settings.image_factor_source == "default":
            print(f"图片 token 下界按每 {FACTOR}x{FACTOR} 像素 1 个 token 估算（默认 --image-factor {FACTOR}）；"
                  "若训练用了别的值，请显式传入 --image-factor。")
    if emit_skipped:
        print(f"检测到阻断发现，未写出 eval 文件；{args.emit_eval} 如已存在则保持原样。", file=sys.stderr)
    if emit_error is not None:
        print(f"eval 转换失败：{emit_error}", file=sys.stderr)
    elif emit_counts is not None:
        wrote, skipped, dropped = emit_counts
        print(f"已写出 {wrote} 条 eval 请求（跳过 {skipped} 个问题，丢弃 {dropped} 条记录）到 {args.emit_eval}",
              file=sys.stderr)

    if args.report:
        commit = args.commit or _detect_commit(os.path.dirname(os.path.abspath(__file__)))
        report = build_report(results, cross_observations, settings, args.role, args.tokenizer, commit,
                              against_given=bool(args.against), image_unverified=image_unverified)
        try:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except OSError as e:
            return _usage_error(f"无法写出 --report 文件 {args.report}：{e}")

    return 0 if not blocking and emit_error is None else 1


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.image_factor is not None and args.image_factor < 1:
        return _usage_error("--image-factor 必须是 >= 1 的整数")
    if args.min_pixels < 1 or args.max_pixels < 1:
        return _usage_error("--min-pixels 和 --max-pixels 都必须 >= 1")
    if args.max_pixels < args.min_pixels:
        return _usage_error("--max-pixels 必须 >= --min-pixels")
    tokenizer = None
    if args.tokenizer:
        try:
            tokenizer = _load_tokenizer(args.tokenizer)
        except Exception as e:              # 模型目录/依赖问题，不是检查器 bug
            print(f"--tokenizer 无法加载模型目录 {args.tokenizer}：{e}", file=sys.stderr)
            return 2
    try:
        return run(args, tokenizer)
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except TokenizerError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        print("这是检查器的问题，请把这条记录发回来。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
