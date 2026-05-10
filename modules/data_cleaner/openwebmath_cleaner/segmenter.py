"""
segmenter.py — 段级切分

将整篇文档切成适合评分和 LLM 处理的段落。
算法：空行粗切 → 标题拆分 → 合并短块 → 拆分长块 → 公式保护
"""
from __future__ import annotations

import re
from typing import List

from .config import PipelineConfig
from .schemas import Segment

# ---------------------------------------------------------------------------
# 正则模式
# ---------------------------------------------------------------------------

# 连续两个及以上空行（段落分隔）
_BLANK_LINE_SPLIT = re.compile(r"\n\s*\n")

# Markdown 标题行
_HEADING_RE = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)

# 独立 display math 块的开始/结束
_DISPLAY_MATH_START = re.compile(r"^\s*(?:\$\$|\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\})", re.MULTILINE)
_DISPLAY_MATH_END = re.compile(r"(?:\$\$|\\end\{(?:equation|align|gather|multline|eqnarray)\*?\})\s*$", re.MULTILINE)

# 句号 / 中文句号，用于超长块的二次切分
_SENTENCE_END = re.compile(r"[.。!！?？]\s+")


# ---------------------------------------------------------------------------
# 公式保护：标记公式区间，切分时避开
# ---------------------------------------------------------------------------

def _find_math_spans(text: str) -> list[tuple[int, int]]:
    """
    找到文本中的 display math 区间（$$...$$ 和 \\begin...\\end）。
    返回 (start, end) 列表，用于切分时避开公式内部。
    """
    spans = []

    # $$ ... $$
    for m in re.finditer(r"\$\$(.*?)\$\$", text, re.DOTALL):
        spans.append((m.start(), m.end()))

    # \begin{env} ... \end{env}
    for m in re.finditer(
        r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", text, re.DOTALL
    ):
        spans.append((m.start(), m.end()))

    # 合并重叠区间
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _is_in_math(pos: int, spans: list[tuple[int, int]]) -> bool:
    """判断某个位置是否在公式区间内"""
    for s, e in spans:
        if s <= pos < e:
            return True
        if s > pos:
            break
    return False


# ---------------------------------------------------------------------------
# 核心切分逻辑
# ---------------------------------------------------------------------------

def _split_by_blank_lines(text: str) -> list[str]:
    """按连续空行切分"""
    blocks = _BLANK_LINE_SPLIT.split(text)
    return [b.strip() for b in blocks if b.strip()]


def _split_at_headings(blocks: list[str]) -> list[str]:
    """
    如果一个块内包含 Markdown 标题行，在标题处拆分。
    标题行归到下一个块。
    """
    result = []
    for block in blocks:
        lines = block.split("\n")
        current_lines: list[str] = []
        for line in lines:
            if _HEADING_RE.match(line) and current_lines:
                # 标题前的内容作为一个块
                chunk = "\n".join(current_lines).strip()
                if chunk:
                    result.append(chunk)
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            chunk = "\n".join(current_lines).strip()
            if chunk:
                result.append(chunk)
    return result


def _merge_short_blocks(
    blocks: list[str],
    min_chars: int,
    max_chars: int = 0,
) -> list[str]:
    """
    合并短块：如果块长度 < min_chars，向前合并到上一个块。
    合并后不超过 max_chars（如果设了的话）。

    多轮合并，直到没有短块可合并为止。
    """
    if not blocks:
        return blocks

    changed = True
    while changed:
        changed = False
        merged = [blocks[0]]
        for block in blocks[1:]:
            prev_len = len(merged[-1])
            curr_len = len(block)
            combined_len = prev_len + curr_len + 2  # +2 for "\n\n"

            # 如果当前块或上一块太短，且合并后不会超限，就合并
            should_merge = (prev_len < min_chars or curr_len < min_chars)
            if max_chars > 0 and combined_len > max_chars:
                should_merge = False

            if should_merge:
                merged[-1] = merged[-1] + "\n\n" + block
                changed = True
            else:
                merged.append(block)

        blocks = merged

    return blocks


def _split_long_block(block: str, hard_max: int, ideal_max: int) -> list[str]:
    """
    拆分超长块。

    策略：
    1. 优先在公式区间外的空行处切
    2. 其次在句号后切
    3. 最后强制在 hard_max 位置切
    """
    if len(block) <= hard_max:
        return [block]

    math_spans = _find_math_spans(block)
    result = []
    remaining = block

    while len(remaining) > hard_max:
        # 在 ideal_max 附近找切分点
        search_end = min(hard_max, len(remaining))

        best_pos = -1

        # 策略1: 找空行
        for m in re.finditer(r"\n\s*\n", remaining[:search_end]):
            pos = m.end()
            if pos >= ideal_max // 2 and not _is_in_math(m.start(), math_spans):
                best_pos = pos
                if pos >= ideal_max * 0.7:
                    break

        # 策略2: 找句号
        if best_pos < ideal_max // 2:
            for m in _SENTENCE_END.finditer(remaining[:search_end]):
                pos = m.end()
                if pos >= ideal_max // 2 and not _is_in_math(m.start(), math_spans):
                    best_pos = pos
                    if pos >= ideal_max * 0.7:
                        break

        # 策略3: 强制切
        if best_pos < ideal_max // 3:
            best_pos = search_end

        chunk = remaining[:best_pos].strip()
        if chunk:
            result.append(chunk)
        remaining = remaining[best_pos:].strip()

        # 更新 math_spans 偏移量
        offset = best_pos
        math_spans = [(s - offset, e - offset) for s, e in math_spans if e > offset]

    if remaining.strip():
        result.append(remaining.strip())

    return result


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def segment_document(
    doc_id: str,
    text: str,
    config: PipelineConfig | None = None,
) -> list[Segment]:
    """
    将文档文本切分为段落列表。

    参数:
        doc_id: 文档 ID
        text: 文档原始文本
        config: 配置（None 时使用默认值）

    返回:
        Segment 列表
    """
    if config is None:
        config = PipelineConfig()

    if not text or not text.strip():
        return []

    text = text.strip()

    # 第一步：按空行粗切
    blocks = _split_by_blank_lines(text)

    # 第二步：在标题处进一步拆分
    blocks = _split_at_headings(blocks)

    # 第三步：合并短块（用 ideal_min_chars 做阈值，合并后不超过 hard_max）
    blocks = _merge_short_blocks(blocks, config.ideal_min_chars, config.hard_max_chars)

    # 第四步：拆分超长块
    final_blocks = []
    for block in blocks:
        if len(block) > config.hard_max_chars:
            final_blocks.extend(
                _split_long_block(block, config.hard_max_chars, config.ideal_max_chars)
            )
        else:
            final_blocks.append(block)

    # 最终再做一次短块合并（拆分可能产生新的短块），用较宽松的阈值
    final_blocks = _merge_short_blocks(
        final_blocks, config.min_segment_chars, config.hard_max_chars
    )

    # 构造 Segment 对象
    segments = []
    for i, block in enumerate(final_blocks):
        seg_id = f"{doc_id}_seg_{i:04d}"
        segments.append(Segment(
            doc_id=doc_id,
            segment_id=seg_id,
            text=block,
        ))

    return segments
