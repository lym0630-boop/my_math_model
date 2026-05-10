"""
light_rules.py — 轻规则清洗 + 特征提取

只做低风险清洗，不承担"理解数学"的任务。
三部分能力：A. 文本清洗  B. 直接丢弃判定  C. 数学感知特征提取
"""
from __future__ import annotations

import re
import html
from typing import Dict

from .config import PipelineConfig
from .schemas import Segment, LightRuleResult

# ===========================================================================
# 正则模式（参考旧 openwebmath_pipeline.py 中的成熟模式）
# ===========================================================================

# HTML 实体
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|lt|gt|amp|quot|apos|mdash|ndash|hellip|#\d{1,5}|#x[0-9a-fA-F]{1,4});", re.IGNORECASE)

# HTML 标签（残留的）
_HTML_TAG_RE = re.compile(
    r"</?(?:div|span|p|br|table|td|tr|th|ul|li|ol|h[1-6]|a|img|script|style|"
    r"header|footer|nav|section|article|aside|main|form|input|button|select|"
    r"textarea|iframe|embed|object|meta|link)[\s/>]",
    re.IGNORECASE,
)

# 控制字符（保留换行、制表符）
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# URL 模式
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# BBCode 标签（论坛常见）
# ---------------------------------------------------------------------------
# 有内容的 BBCode：[b]text[/b] → text
_BBCODE_CONTENT_RE = re.compile(
    r"\[(?:b|i|u|s|quote|code|color|size|font|center|left|right|indent|"
    r"spoiler|hide|highlight|sub|sup)\b[^\]]*\](.*?)\[/\1?\w*\]",
    re.IGNORECASE | re.DOTALL,
)
# 简化版 [sup]...[/sup] 等
_BBCODE_SUP_RE = re.compile(r"\[sup\](.*?)\[/sup\]", re.IGNORECASE)
_BBCODE_SUB_RE = re.compile(r"\[sub\](.*?)\[/sub\]", re.IGNORECASE)
_BBCODE_BOLD_RE = re.compile(r"\[b\](.*?)\[/b\]", re.IGNORECASE)
_BBCODE_ITALIC_RE = re.compile(r"\[i\](.*?)\[/i\]", re.IGNORECASE)
_BBCODE_UNDERLINE_RE = re.compile(r"\[u\](.*?)\[/u\]", re.IGNORECASE)

# 纯标签 BBCode：[url]...[/url] → 删除，[img]...[/img] → 删除
_BBCODE_URL_RE = re.compile(
    r'\[url(?:=[^\]]+)?\].*?\[/url\]|\[URL(?:="[^"]*")?\].*?\[/URL\]',
    re.IGNORECASE | re.DOTALL,
)
_BBCODE_IMG_RE = re.compile(r"\[img\b[^\]]*\].*?\[/img\]", re.IGNORECASE | re.DOTALL)

# 剩余的任何未匹配 BBCode 标签
_BBCODE_TAG_RE = re.compile(r"\[/?(?:url|img|quote|code|color|size|font|center|left|right|indent|spoiler|hide|highlight|media|video|attach)\b[^\]]*\]", re.IGNORECASE)

# ---------------------------------------------------------------------------
# 论坛/网页噪声行模式
# ---------------------------------------------------------------------------
# 用户名 + 时间戳行：如 "Kalli Hofmann 2020-04-15 10:28"、"• Commented Sep 18, 2018 at 9:25"
_FORUM_TIMESTAMP_RE = re.compile(
    r"^\s*(?:•\s*)?(?:Commented|Posted|Edited|Updated|Answered|Asked)?\s*"
    r"(?:[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}\s+(?:at\s+)?\d{1,2}:\d{2})"
    r"|^\s*\w[\w\s]{0,30}\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}",
    re.MULTILINE | re.IGNORECASE,
)

# 纯 URL 独立行（整行只有 URL，无有用文本）
_STANDALONE_URL_LINE_RE = re.compile(
    r"^\s*(?:https?://\S+|\[url\].*?\[/url\])\s*$",
    re.IGNORECASE,
)

# 附件/图片行
_ATTACHMENT_LINE_RE = re.compile(
    r"^\s*\d*\s*Attachment\(s\)\s*$|"
    r"^\s*(?:Save this PDF as|Size:\s*px|Start display at)",
    re.IGNORECASE,
)

# 面包屑导航行：如 "-   Topic Name (https://...)" 连续出现
_BREADCRUMB_RE = re.compile(
    r"^\s*-\s*(?:-\s*)*\w.*\(https?://[^\)]+\)\s*$",
    re.IGNORECASE,
)

# 站点名 + URL 行：如 "mersenneforum.org (https://...)"
_SITE_HEADER_RE = re.compile(
    r"^\s*[\w.-]+\.(?:org|com|net|edu|io)\s*\(https?://[^\)]+\)\s*$",
    re.IGNORECASE,
)

# "Originally Posted by ..." 引用头
_QUOTE_HEADER_RE = re.compile(
    r"^\s*(?:Originally Posted by|Quote:)\s*",
    re.IGNORECASE,
)

# 论坛用户名独立短行（1-2 个单词，非数学）
# 匹配: "rqeeb" / "Plato" / "neela" / "– JMP"
_FORUM_USERNAME_RE = re.compile(
    r"^\s*(?:–\s*)?\w{2,20}\s*$",
)

# 论坛 bullet + 日期时间："• Oct 11th 2011, 07:09 PM"
_FORUM_BULLET_DATE_RE = re.compile(
    r"^\s*•\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}",
    re.IGNORECASE,
)

# 论坛用户签名 / 页面时间
_FORUM_FOOTER_RE = re.compile(
    r"^\s*All times are (?:UTC|GMT|EST|PST|CET)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 论坛评论/互动行（• 开头 + Commented/said 等）
# ---------------------------------------------------------------------------
_FORUM_COMMENT_RE = re.compile(
    r"^\s*•\s*.{0,50}Commented\s+\w+\s+\d",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Q&A / 教育网站元数据噪声
# ---------------------------------------------------------------------------
# 用户名 + 角色行："justaguide | College Teacher | (Level 2) Distinguished Educator"
_QA_USER_LINE_RE = re.compile(
    r"^\s*\w[\w\s]{0,30}\s*\|\s*(?:College|High School|Middle School)?\s*(?:Teacher|Educator|Student|Professor|Tutor)",
    re.IGNORECASE,
)

# "Posted on" 独立行
_POSTED_ON_RE = re.compile(r"^\s*Posted on\s*$", re.IGNORECASE)

# "Updated: 日期" 独立行
_UPDATED_LINE_RE = re.compile(r"^\s*Updated:\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$", re.IGNORECASE)

# "Wiki User" / "14y ago" / "Answered by" 等
_QA_META_RE = re.compile(
    r"^\s*(?:Wiki User|Answered by|Asked by)\s*$|"
    r"^\s*\d+[ymdh]\s+ago\s*$",
    re.IGNORECASE,
)

# "Is there an error in this question or solution?" 类反馈行
_QA_FEEDBACK_RE = re.compile(
    r"^\s*Is there an error in this (?:question|solution)\??|"
    r"^\s*(?:Report Error|Flag as Inappropriate|Mark as Brainliest)\s*$",
    re.IGNORECASE,
)

# "APPEARS IN" + 教材引用块（如 "RD Sharma Class 11..." "Exercise 1.8 | Q 15"）
_TEXTBOOK_REF_RE = re.compile(
    r"^\s*(?:APPEARS?\s+IN|SOURCE|REFERENCE)\s*$",
    re.IGNORECASE,
)

# "Chapter X ..." / "Exercise X.X | Q N | Page N" 教材定位行
_EXERCISE_REF_RE = re.compile(
    r"^\s*(?:Chapter\s+\d+\s+\w+|Exercise\s+[\d.]+\s*\|\s*Q\s*\d+\s*\|\s*Page\s*\d+)\s*$",
    re.IGNORECASE,
)

# "Concept: XXX" 标签行
_CONCEPT_TAG_RE = re.compile(r"^\s*Concept:\s*\w+", re.IGNORECASE)

# 教育网站推广："As a Chegg Study subscriber" / "Slader's free" / "download our app"
_EDU_PROMO_RE = re.compile(
    r"(?:Chegg\s+Study|Slader|StudyBlue|Course\s+Hero)\s+(?:subscriber|free|solutions?)|"
    r"download\s+our\s+(?:homework\s+)?(?:help\s+)?app|"
    r"redefine\s+your\s+true\s+self|"
    r"Shed\s+the\s+societal",
    re.IGNORECASE,
)

# "Skip Nav" 等无障碍导航残留
_SKIP_NAV_RE = re.compile(r"^\s*Skip\s+(?:Nav|Navigation|to\s+(?:main\s+)?content)\s*$", re.IGNORECASE)

# 独立日期行："Apr 19, 2018" 或 "Oct 11th 2011, 07:09 PM"
_STANDALONE_DATE_RE = re.compile(
    r"^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"(?:,?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?)?\s*$",
    re.IGNORECASE,
)

# 论坛用户名独立行（短行，只有一个名字）："rqeeb" / "Plato" / "neela"
# 很难用正则精确匹配，但可以过滤 "Re: ..." 开头的回复标题
_FORUM_RE_TITLE_RE = re.compile(
    r"^\s*Re:\s+",
    re.IGNORECASE,
)

# "Save this PDF as:" / "Size: px" / "Start display at page:" 类 PDF 转换残留
_PDF_ARTIFACT_RE = re.compile(
    r"^\s*(?:Save this PDF as|Size:\s*px|Start display at\s+page)\s*:?\s*$",
    re.IGNORECASE,
)

# 教育页面末尾的版本/页码标记："People s Physics book 3e Ch 25-1"
_PAGE_MARKER_RE = re.compile(
    r"^\s*(?:People\s*s\s+Physics|Textbook|Source)\s+book\s+\w+\s+Ch\s+[\d-]+\s*$|"
    r"^\s*\d+\s+People\s+s\s+Physics",
    re.IGNORECASE,
)

# 邮件/账户偏好提示
_EMAIL_PREFS_RE = re.compile(
    r"you\s+can\s+change\s+email\s+preferences|"
    r"account\s+settings",
    re.IGNORECASE,
)

# "Privacy FAQs" / "Main Topics" 等独立导航行
_NAV_HEADING_RE = re.compile(
    r"^\s*(?:#{1,4}\s+)?(?:Privacy\s+FAQs?|Main\s+Topics|Table\s+of\s+Contents|"
    r"Quick\s+Links|Site\s+Map)\s*$",
    re.IGNORECASE,
)

# Boilerplate 行模式 — 整行匹配则删除
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"(?:accept|use|uses?)\s+cookies?", re.IGNORECASE),
    re.compile(r"subscribe\s+(?:to|for|now)", re.IGNORECASE),
    re.compile(r"sign\s*up\s+(?:for|to|now)", re.IGNORECASE),
    re.compile(r"(?:log\s*in|sign\s*in)\s+(?:to|with|using)", re.IGNORECASE),
    re.compile(r"(?:click\s+here|read\s+more|learn\s+more|see\s+also)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:menu|navigation|sidebar|footer|header|breadcrumb)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:previous|next)\s+(?:post|article|page)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:copyright|©|\(c\))\s+", re.IGNORECASE),
    re.compile(r"^\s*all\s+rights?\s+reserved", re.IGNORECASE),
    re.compile(r"^\s*(?:posted|published)\s+(?:on|in|by)\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:tags?|categories?|filed\s+under)\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*(?:views?|likes?|shares?|retweets?)\s*$", re.IGNORECASE),
    re.compile(r"(?:facebook|twitter|linkedin|pinterest|reddit|instagram)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:advertisement|sponsored|ad)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:get\s+(?:a\s+)?free|start\s+(?:your|a)\s+free|free\s+trial)", re.IGNORECASE),
    re.compile(r"^\s*(?:Rate\s+this|Was\s+this\s+helpful|Helpful\s*\?)", re.IGNORECASE),
    re.compile(r"^\s*(?:share\s+this|share\s+on|tweet|pin\s+it)\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:Disclaimer|Terms\s+of\s+(?:Service|Use)|Privacy\s+Policy)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:About\s+the\s+(?:Author|Creator))\s*$", re.IGNORECASE),
    re.compile(r"^\s*Not\s+the\s+answer\s+you.re\s+looking\s+for", re.IGNORECASE),
    re.compile(r"^\s*Browse\s+other\s+questions?\s+tagged", re.IGNORECASE),
    re.compile(r"^\s*(?:Report\s+(?:an?\s+)?(?:error|issue|bug|problem))\s*$", re.IGNORECASE),
]

# 尾部截断模式 — 从匹配位置起截断整个后续内容
_TAIL_TRUNCATE_PATTERNS = [
    re.compile(r"^\s*#{0,4}\s*(?:Responses?\s+to|Leave\s+a\s+(?:Reply|Comment)|Comments?\s*(?:\(\d+\))?)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*#{0,4}\s*(?:Related\s+(?:Posts?|Articles?|Entries|Problems?|Questions?|Topics?|Resources?))\s*:?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*#{0,4}\s*(?:You\s+(?:might|may)\s+(?:also\s+)?(?:like|enjoy|be\s+interested))", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*#{0,4}\s*(?:Recommended\s+(?:for\s+you|problems?|reading))", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Showing\s+results?\s+for\s*$", re.MULTILINE | re.IGNORECASE),
    # 教育网站推广段
    re.compile(r"^\s*As a Chegg Study subscriber", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Now is the time to redefine your true self", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Can you find your fundamental truth", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*You can (?:also )?(?:download|check) our", re.MULTILINE | re.IGNORECASE),
]

# 数学相关正则
_MATH_SYMBOL_RE = re.compile(r"[+\-*/=≠≤≥≈∞∑∏∫∂∇±×÷∈∉⊂⊃∪∩∧∨¬⇒⇔∀∃]")
_LATEX_CMD_RE = re.compile(r"\\(?:frac|int|sum|prod|lim|sqrt|partial|nabla|infty|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|pi|phi|psi|begin|end|left|right|text|mathbb|mathcal|mathbf|mathrm|operatorname)\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?%?")
_VARIABLE_RE = re.compile(r"\b([a-zA-Z])(?:_\{?[a-zA-Z0-9]+\}?)?\b")
_MATH_KEYWORDS = [
    "theorem", "lemma", "proposition", "corollary", "definition",
    "proof", "example", "solution", "exercise", "problem",
    "equation", "formula", "integral", "derivative", "function",
    "matrix", "vector", "polynomial", "inequality", "sequence",
    "convergence", "limit", "continuous", "differentiable",
    # 中文关键词
    "定理", "引理", "命题", "推论", "定义",
    "证明", "例题", "解题", "练习", "题目",
    "方程", "公式", "积分", "导数", "函数",
]
_MATH_KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _MATH_KEYWORDS if k.isascii()) + r")\b"
    + r"|(?:" + "|".join(re.escape(k) for k in _MATH_KEYWORDS if not k.isascii()) + r")",
    re.IGNORECASE,
)

# 题目/解答信号
_PROBLEM_SOLUTION_RE = re.compile(
    r"(?:problem|exercise|question|example)\s*[\d.:)]|"
    r"(?:solution|answer|proof)\s*[:.]|"
    r"题目|解[：:]|解答|证明[：:]",
    re.IGNORECASE,
)

# 证明信号
_PROOF_RE = re.compile(
    r"(?:proof|prove|Q\.E\.D\.|∎|□)|"
    r"证明|证毕",
    re.IGNORECASE,
)

# 常见英文停用词（用于变量提取时过滤）
_STOP_WORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "is", "it",
    "by", "as", "or", "if", "so", "no", "be", "do", "we", "he", "me",
    "up", "am", "my",
})


# ===========================================================================
# A. 文本清洗
# ===========================================================================

def clean_text(text: str) -> str:
    """
    低风险文本清洗。

    - 替换 HTML 实体
    - 删除残留 HTML 标签
    - 清洗 BBCode 标签（论坛来源）
    - 删除控制字符
    - 删除论坛元数据行（用户名+时间戳、纯URL行、面包屑导航等）
    - 统一多余空白和换行
    - 删除 boilerplate 行
    - 尾部截断
    - 删除相邻重复行
    """
    # HTML 实体解码
    text = html.unescape(text)
    # 残余的 HTML 实体模式
    text = _HTML_ENTITY_RE.sub(" ", text)

    # 删除残留 HTML 标签
    text = _HTML_TAG_RE.sub("", text)

    # --- BBCode 清洗 ---
    # 先处理带 URL 的 BBCode（整个删除，因为 URL 不是数学内容）
    text = _BBCODE_URL_RE.sub("", text)
    text = _BBCODE_IMG_RE.sub("", text)
    # 保留内容的 BBCode：[sup]2[/sup] → ^{2}，[sub]n[/sub] → _{n}
    text = _BBCODE_SUP_RE.sub(r"^{\1}", text)
    text = _BBCODE_SUB_RE.sub(r"_{\1}", text)
    # [b]text[/b] → text，[i]text[/i] → text
    text = _BBCODE_BOLD_RE.sub(r"\1", text)
    text = _BBCODE_ITALIC_RE.sub(r"\1", text)
    text = _BBCODE_UNDERLINE_RE.sub(r"\1", text)
    # 清除剩余的 BBCode 标签
    text = _BBCODE_TAG_RE.sub("", text)

    # 删除控制字符
    text = _CONTROL_CHAR_RE.sub("", text)

    # 尾部截断：从 boilerplate 边界处截断
    for pat in _TAIL_TRUNCATE_PATTERNS:
        m = pat.search(text)
        if m:
            text = text[:m.start()]

    # 逐行处理：删除各类噪声行
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过空行（后面统一处理）
        if not stripped:
            cleaned_lines.append("")
            continue

        # 检查是否匹配 boilerplate 模式
        is_noise = False
        for pat in _BOILERPLATE_LINE_PATTERNS:
            if pat.search(stripped):
                is_noise = True
                break

        # 论坛/网页噪声行
        if not is_noise:
            if _STANDALONE_URL_LINE_RE.match(stripped):
                is_noise = True
            elif _ATTACHMENT_LINE_RE.match(stripped):
                is_noise = True
            elif _BREADCRUMB_RE.match(stripped):
                is_noise = True
            elif _SITE_HEADER_RE.match(stripped):
                is_noise = True
            elif _FORUM_FOOTER_RE.match(stripped):
                is_noise = True
            elif _FORUM_TIMESTAMP_RE.match(stripped):
                is_noise = True
            elif _FORUM_COMMENT_RE.match(stripped):
                is_noise = True
            elif _QA_USER_LINE_RE.match(stripped):
                is_noise = True
            elif _POSTED_ON_RE.match(stripped):
                is_noise = True
            elif _UPDATED_LINE_RE.match(stripped):
                is_noise = True
            elif _QA_META_RE.match(stripped):
                is_noise = True
            elif _QA_FEEDBACK_RE.match(stripped):
                is_noise = True
            elif _TEXTBOOK_REF_RE.match(stripped):
                is_noise = True
            elif _EXERCISE_REF_RE.match(stripped):
                is_noise = True
            elif _CONCEPT_TAG_RE.match(stripped):
                is_noise = True
            elif _EDU_PROMO_RE.search(stripped):
                is_noise = True
            elif _SKIP_NAV_RE.match(stripped):
                is_noise = True
            elif _STANDALONE_DATE_RE.match(stripped):
                is_noise = True
            elif _FORUM_RE_TITLE_RE.match(stripped):
                is_noise = True
            elif _PDF_ARTIFACT_RE.match(stripped):
                is_noise = True
            elif _PAGE_MARKER_RE.match(stripped):
                is_noise = True
            elif _NAV_HEADING_RE.match(stripped):
                is_noise = True
            elif _QUOTE_HEADER_RE.match(stripped):
                is_noise = True
            elif _FORUM_BULLET_DATE_RE.match(stripped):
                is_noise = True

        if not is_noise:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 删除行内独立 URL（不在数学上下文中的）
    # 保守策略：只删除独立出现的 URL（前后有空格或行首行尾），不删嵌在句子里的
    text = re.sub(r"(?<=\s)https?://\S+(?=\s|$)", "", text)
    text = re.sub(r"^https?://\S+\s*", "", text, flags=re.MULTILINE)

    # 删除相邻重复行
    lines = text.split("\n")
    deduped = []
    prev = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev:
            continue
        deduped.append(line)
        prev = stripped
    text = "\n".join(deduped)

    # 统一多余换行（3+空行 → 2空行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 统一多余空格
    text = re.sub(r" {3,}", " ", text)
    # 行尾空格
    text = re.sub(r" +\n", "\n", text)

    # --- 后处理 ---
    # 删除段首独立数字行（如 PDF 页码残留 "0\n\n" 或 "5\n"）
    text = re.sub(r"^\s*\d{1,3}\s*\n", "", text)

    # 删除页脚标记（如 "People s Physics book 3e Ch 25-1"）
    text = re.sub(
        r"\n\s*(?:\d+\s+)?(?:People\s*s\s+Physics|Textbook)\s+book\s+\w+\s+Ch\s+[\d-]+\s*$",
        "", text, flags=re.IGNORECASE,
    )

    # 删除尾部文本表情
    text = re.sub(r"\s*[:;][\-]?[)D(P/\\|]\s*$", "", text)

    return text.strip()


# ===========================================================================
# C. 特征提取（放在 B 之前，因为 B 依赖特征）
# ===========================================================================

def extract_features(text: str) -> Dict[str, float | int | bool]:
    """
    从文本中提取数学感知特征。
    """
    char_len = len(text)

    # URL / 链接比例
    urls = _URL_RE.findall(text)
    url_chars = sum(len(u) for u in urls)
    link_ratio = url_chars / max(char_len, 1)

    # 重复行比例
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        unique_lines = set(lines)
        repeat_ratio = 1.0 - len(unique_lines) / len(lines)
    else:
        repeat_ratio = 0.0

    # Boilerplate 命中数
    boilerplate_hits = 0
    for line in lines:
        for pat in _BOILERPLATE_LINE_PATTERNS:
            if pat.search(line):
                boilerplate_hits += 1
                break

    # 数学符号
    symbol_count = len(_MATH_SYMBOL_RE.findall(text))

    # LaTeX 命令
    latex_cmd_count = len(_LATEX_CMD_RE.findall(text))

    # 数字
    number_count = len(_NUMBER_RE.findall(text))

    # 变量
    var_matches = _VARIABLE_RE.findall(text)
    variable_like_count = len([v for v in var_matches if v.lower() not in _STOP_WORDS])

    # 数学关键词
    math_keyword_count = len(_MATH_KEYWORD_RE.findall(text))

    # 题目/解答信号
    has_problem_solution_signal = bool(_PROBLEM_SOLUTION_RE.search(text))

    # 证明信号
    has_proof_signal = bool(_PROOF_RE.search(text))

    return {
        "char_len": char_len,
        "link_ratio": round(link_ratio, 4),
        "repeat_ratio": round(repeat_ratio, 4),
        "boilerplate_hits": boilerplate_hits,
        "symbol_count": symbol_count,
        "latex_cmd_count": latex_cmd_count,
        "number_count": number_count,
        "variable_like_count": variable_like_count,
        "math_keyword_count": math_keyword_count,
        "has_problem_solution_signal": has_problem_solution_signal,
        "has_proof_signal": has_proof_signal,
    }


# ===========================================================================
# B. 直接丢弃判定
# ===========================================================================

def should_hard_drop(features: Dict, config: PipelineConfig) -> bool:
    """
    极确定是垃圾时才丢弃，宁可漏杀不可误杀。
    """
    char_len = features.get("char_len", 0)
    link_ratio = features.get("link_ratio", 0)
    repeat_ratio = features.get("repeat_ratio", 0)
    boilerplate_hits = features.get("boilerplate_hits", 0)
    latex_cmd_count = features.get("latex_cmd_count", 0)
    math_keyword_count = features.get("math_keyword_count", 0)
    symbol_count = features.get("symbol_count", 0)
    has_any_math = (
        latex_cmd_count > 0
        or math_keyword_count > 0
        or symbol_count >= 2
        or features.get("number_count", 0) >= 3
        or features.get("variable_like_count", 0) >= 3
    )

    # 极短且无数学信号
    if char_len < config.min_text_chars and not has_any_math:
        return True

    # 链接比例极高
    if link_ratio > config.max_link_ratio:
        return True

    # boilerplate 命中极强且几乎无数学内容
    if boilerplate_hits > config.max_boilerplate_for_drop and not has_any_math:
        return True

    # 重复率极高且内容空洞
    if repeat_ratio > config.max_repeat_ratio and char_len < 100:
        return True

    return False


def _has_strong_math_signal(features: Dict) -> bool:
    """判断是否有强数学信号"""
    return (
        features.get("latex_cmd_count", 0) >= 2
        or features.get("math_keyword_count", 0) >= 2
        or features.get("has_problem_solution_signal", False)
        or features.get("has_proof_signal", False)
        or features.get("symbol_count", 0) >= 5
    )


def _needs_llm(features: Dict) -> bool:
    """判断是否需要 LLM 清理（有噪声迹象但也有数学内容）"""
    has_noise = (
        features.get("boilerplate_hits", 0) >= 2
        or features.get("link_ratio", 0) >= 0.1
        or features.get("repeat_ratio", 0) >= 0.15
    )
    has_math = (
        features.get("latex_cmd_count", 0) >= 1
        or features.get("math_keyword_count", 0) >= 1
        or features.get("symbol_count", 0) >= 3
    )
    return has_noise and has_math


# ===========================================================================
# 主入口
# ===========================================================================

def apply_light_rules(
    segment: Segment,
    config: PipelineConfig | None = None,
) -> LightRuleResult:
    """
    对单个段落执行轻规则清洗 + 特征提取 + 丢弃判定。

    返回 LightRuleResult，包含清洗后文本、是否丢弃、特征字典等。

    注意：特征提取在原始文本上进行（用于丢弃判定），
    但输出的 cleaned_text 是清洗后的版本。
    """
    if config is None:
        config = PipelineConfig()

    # 先在原始文本上提取特征（用于丢弃判定和路由）
    raw_features = extract_features(segment.text)

    # 丢弃判定（基于原始特征）
    hard_drop = should_hard_drop(raw_features, config)

    # 清洗
    cleaned = clean_text(segment.text)

    # 如果清洗后文本为空或极短，也标记为丢弃
    if not cleaned or len(cleaned) < config.min_text_chars:
        has_any_math = (
            raw_features.get("latex_cmd_count", 0) > 0
            or raw_features.get("math_keyword_count", 0) > 0
            or raw_features.get("symbol_count", 0) > 2
        )
        if not has_any_math:
            hard_drop = True

    # 在清洗后文本上也提取特征（供 scorer 使用）
    cleaned_features = extract_features(cleaned)

    # 合并特征：丢弃判定用原始特征，其他用清洗后特征
    # 但保留原始的 boilerplate_hits 作为额外信号
    features = cleaned_features.copy()
    features["raw_boilerplate_hits"] = raw_features["boilerplate_hits"]
    features["raw_char_len"] = raw_features["char_len"]

    # 数学信号（基于清洗后特征）
    strong_math = _has_strong_math_signal(features)

    # 是否需要 LLM
    needs_llm = _needs_llm(features)

    return LightRuleResult(
        cleaned_text=cleaned,
        hard_drop=hard_drop,
        strong_math_signal=strong_math,
        needs_llm_cleanup=needs_llm,
        feature_dict=features,
    )
