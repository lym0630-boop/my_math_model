"""
llm_processor.py — LLM 统一处理

LLM 只处理灰区样本，限定操作范围：去噪、可恢复补全、分步重写。
提供三个实现：NoOp、RegexAssist、RealLLM（预留）。
"""
from __future__ import annotations

import abc
import re
from typing import Dict, Optional

from .schemas import LLMDecision, LLMProcessResult

# 复用 light_rules 中的 boilerplate 模式
from .light_rules import _BOILERPLATE_LINE_PATTERNS, _TAIL_TRUNCATE_PATTERNS


# ===========================================================================
# 基类
# ===========================================================================

class BaseLLMProcessor(abc.ABC):
    """LLM 处理器基类"""

    @abc.abstractmethod
    def process(self, text: str, meta: Dict | None = None) -> LLMProcessResult:
        """
        处理灰区文本。

        参数:
            text: 轻清洗后的文本
            meta: 可选的元信息（如特征字典）

        返回:
            LLMProcessResult
        """
        ...


# ===========================================================================
# NoOpProcessor — 原样返回
# ===========================================================================

class NoOpProcessor(BaseLLMProcessor):
    """什么都不做，原样返回"""

    def process(self, text: str, meta: Dict | None = None) -> LLMProcessResult:
        return LLMProcessResult(
            text=text,
            decision=LLMDecision.KEEP_AS_IS,
            metadata={"processor": "noop"},
        )


# ===========================================================================
# RegexAssistProcessor — 最小可运行版本
# ===========================================================================

class RegexAssistProcessor(BaseLLMProcessor):
    """
    在无真实 LLM 时提供的最小清理版本。

    只做：
    1. 残余 boilerplate 行删除
    2. 多余空白压缩
    3. 完全重复的相邻句清理
    4. 尾部残余噪声截断
    不做内容补全或重写。
    """

    def process(self, text: str, meta: Dict | None = None) -> LLMProcessResult:
        original_len = len(text)
        cleaned = text

        # 1. 尾部截断
        for pat in _TAIL_TRUNCATE_PATTERNS:
            m = pat.search(cleaned)
            if m:
                cleaned = cleaned[:m.start()]

        # 2. 删除 boilerplate 行
        lines = cleaned.split("\n")
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered_lines.append("")
                continue
            is_bp = False
            for pat in _BOILERPLATE_LINE_PATTERNS:
                if pat.search(stripped):
                    is_bp = True
                    break
            if not is_bp:
                filtered_lines.append(line)
        cleaned = "\n".join(filtered_lines)

        # 3. 删除相邻重复句
        sentences = re.split(r"(?<=[.。!！?？])\s+", cleaned)
        if len(sentences) > 1:
            deduped = [sentences[0]]
            for s in sentences[1:]:
                if s.strip() != deduped[-1].strip():
                    deduped.append(s)
            cleaned = " ".join(deduped)

        # 4. 多余空白压缩
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r" {3,}", " ", cleaned)
        cleaned = re.sub(r" +\n", "\n", cleaned)
        cleaned = cleaned.strip()

        # 判断是否做了实质性修改
        changed = abs(len(cleaned) - original_len) > 10 or cleaned != text.strip()
        decision = LLMDecision.LIGHT_CLEAN if changed else LLMDecision.KEEP_AS_IS

        return LLMProcessResult(
            text=cleaned,
            decision=decision,
            metadata={
                "processor": "regex_assist",
                "chars_removed": original_len - len(cleaned),
            },
        )


# ===========================================================================
# RealLLMProcessor — 预留真实 LLM 接口
# ===========================================================================

# LLM 系统 prompt（供真实 LLM 使用）
_SYSTEM_PROMPT = """你是一个数学文本清洗助手。你的任务是对数学预训练语料做保守清理。

你只能做以下三件事：
1. **去噪**：删除残余网页噪声、无关引导语、冗余重复说明
2. **可恢复补全**：补充局部明显缺失但可从当前片段安全恢复的上下文（如被截断的题设、缺失但上下文明确的符号说明）
3. **分步重写**：把零散、跳跃的解答整理成逻辑更连贯、步骤更清晰的形式

**严格禁止**：
- 不改公式含义、不改数字、不改变量名
- 不添加不存在的推导步骤
- 不把精确信息改成模糊表述
- 不脑补新条件、新假设、新结论

请先判断这段文本需要什么处理，然后给出你的决策和处理后的文本。

决策选项：
- keep_as_is: 文本质量已足够好，无需修改
- light_clean: 只需删除少量噪声
- restore_context: 需要补充缺失的上下文
- rewrite_steps: 需要整理解答步骤
- drop: 文本没有数学价值，建议丢弃

请按以下格式输出：
DECISION: <决策>
TEXT:
<处理后的文本>
"""


class RealLLMProcessor(BaseLLMProcessor):
    """
    真实 LLM 处理器（预留接口）。

    支持 OpenAI 兼容 API 和本地模型。
    """

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model_name: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def process(self, text: str, meta: Dict | None = None) -> LLMProcessResult:
        """
        调用真实 LLM API 处理文本。

        TODO: 实现真实 API 调用
        """
        try:
            response_text = self._call_api(text)
            decision, processed = self._parse_response(response_text)
            return LLMProcessResult(
                text=processed,
                decision=decision,
                metadata={"processor": "real_llm", "model": self.model_name},
            )
        except Exception as e:
            # 出错时回退到原文
            return LLMProcessResult(
                text=text,
                decision=LLMDecision.KEEP_AS_IS,
                metadata={"processor": "real_llm", "error": str(e)},
            )

    def _call_api(self, text: str) -> str:
        """
        调用 LLM API。

        预留实现，需要安装 openai 或 requests。
        """
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            raise ImportError(
                "使用 RealLLMProcessor 需要安装 openai: pip install openai"
            )

    def _parse_response(self, response: str) -> tuple[LLMDecision, str]:
        """解析 LLM 返回的决策和文本"""
        # 提取 DECISION 行
        decision = LLMDecision.KEEP_AS_IS
        decision_match = re.search(r"DECISION:\s*(\w+)", response)
        if decision_match:
            decision_str = decision_match.group(1).strip().lower()
            try:
                decision = LLMDecision(decision_str)
            except ValueError:
                decision = LLMDecision.KEEP_AS_IS

        # 提取 TEXT 部分
        text_match = re.search(r"TEXT:\s*\n(.*)", response, re.DOTALL)
        if text_match:
            processed = text_match.group(1).strip()
        else:
            # 没有标准格式，整段作为输出
            processed = response.strip()

        return decision, processed


# ===========================================================================
# 工厂函数
# ===========================================================================

def create_processor(processor_type: str = "noop", **kwargs) -> BaseLLMProcessor:
    """
    创建 LLM 处理器实例。

    参数:
        processor_type: "noop", "regex", "real"
        **kwargs: 传给具体处理器的参数
    """
    if processor_type == "noop":
        return NoOpProcessor()
    elif processor_type == "regex":
        return RegexAssistProcessor()
    elif processor_type == "real":
        return RealLLMProcessor(
            api_base=kwargs.get("api_base", ""),
            api_key=kwargs.get("api_key", ""),
            model_name=kwargs.get("model_name", ""),
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.1),
        )
    else:
        raise ValueError(f"未知的处理器类型: {processor_type}")
