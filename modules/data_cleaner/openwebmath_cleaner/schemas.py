"""
schemas.py — 所有数据结构定义

纯 dataclass，零外部依赖。供所有模块统一使用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 段级切分
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """文档切分后的单个段落"""
    doc_id: str
    segment_id: str        # 格式: {doc_id}_seg_{序号:04d}
    text: str
    char_len: int = 0

    def __post_init__(self):
        if self.char_len == 0:
            self.char_len = len(self.text)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 轻规则清洗
# ---------------------------------------------------------------------------

@dataclass
class LightRuleResult:
    """轻规则清洗的输出"""
    cleaned_text: str                   # 清洗后的文本
    hard_drop: bool = False             # 是否被规则直接丢弃
    strong_math_signal: bool = False    # 是否有强数学信号
    needs_llm_cleanup: bool = False     # 是否需要 LLM 进一步处理
    feature_dict: dict = field(default_factory=dict)  # 提取的特征

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# LLM 处理
# ---------------------------------------------------------------------------

class LLMDecision(str, Enum):
    """LLM 输出的决策类型"""
    KEEP_AS_IS = "keep_as_is"
    LIGHT_CLEAN = "light_clean"
    RESTORE_CONTEXT = "restore_context"
    REWRITE_STEPS = "rewrite_steps"
    DROP = "drop"


@dataclass
class LLMProcessResult:
    """LLM 处理的输出"""
    text: str                                   # 处理后的文本
    decision: LLMDecision = LLMDecision.KEEP_AS_IS
    metadata: dict = field(default_factory=dict)  # 附加信息

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


# ---------------------------------------------------------------------------
# 数学感知校验
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """校验结果"""
    validation_pass: bool = True
    number_recall: float = 1.0
    variable_recall: float = 1.0
    symbol_recall: float = 1.0
    length_ratio: float = 1.0
    bracket_balance_ok: bool = True
    latex_balance_ok: bool = True
    score_raw: float = 0.0
    score_post: float = 0.0
    fail_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 路由标记
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """段落在 pipeline 中的最终路径"""
    KEPT_LIGHT_CLEANED = "kept_light_cleaned"
    KEPT_LLM_PROCESSED = "kept_llm_processed"
    FALLBACK_TO_LIGHT_CLEANED = "fallback_to_light_cleaned"
    DROPPED = "dropped"


# ---------------------------------------------------------------------------
# 输出记录
# ---------------------------------------------------------------------------

@dataclass
class TrainOutputRecord:
    """主训练输出 — 只有 id, text, quality_score"""
    id: str
    text: str
    quality_score: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "quality_score": round(self.quality_score, 4),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class AuditRecord:
    """审计日志记录 — 过程信息"""
    id: str
    route: str
    classifier_score_raw: float = 0.0
    classifier_score_post: float = 0.0
    llm_decision: str = ""
    validation_pass: Optional[bool] = None
    fail_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class RunSummary:
    """全局运行统计"""
    total_documents: int = 0
    total_segments: int = 0
    rule_dropped: int = 0
    direct_kept: int = 0
    sent_to_llm: int = 0
    llm_passed_validation: int = 0
    llm_failed_fallback: int = 0
    final_kept: int = 0
    final_dropped: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
