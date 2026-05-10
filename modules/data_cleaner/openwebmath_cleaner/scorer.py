"""
scorer.py — 质量打分与路由

提供统一的评分接口，用 openwebmath-classifier 风格的分数做前后控制。
先实现 DummyScorer（启发式），预留 HFClassifierScorer（真实模型）。
"""
from __future__ import annotations

import abc
from typing import Dict, List

from .config import PipelineConfig


# ===========================================================================
# 基类
# ===========================================================================

class BaseScorer(abc.ABC):
    """评分器基类"""

    @abc.abstractmethod
    def score(self, text: str, features: Dict | None = None) -> float:
        """对单条文本打分，返回 0-5 的分数"""
        ...

    def score_batch(self, texts: List[str], features_list: List[Dict] | None = None) -> List[float]:
        """批量打分，默认逐条调用"""
        if features_list is None:
            return [self.score(t) for t in texts]
        return [self.score(t, f) for t, f in zip(texts, features_list)]


# ===========================================================================
# DummyScorer — 基于特征的启发式评分
# ===========================================================================

class DummyScorer(BaseScorer):
    """
    启发式评分器，基于轻规则特征返回 0-5 的模拟分数。
    用于本地跑通流程，不需要 GPU 或模型。
    """

    def score(self, text: str, features: Dict | None = None) -> float:
        if features is None:
            # 没有预提取特征时，做简单打分
            from .light_rules import extract_features
            features = extract_features(text)

        base = 2.5

        # 正向信号
        latex_cmd = features.get("latex_cmd_count", 0)
        if latex_cmd >= 5:
            base += 1.0
        elif latex_cmd >= 3:
            base += 0.8
        elif latex_cmd >= 1:
            base += 0.4

        math_kw = features.get("math_keyword_count", 0)
        if math_kw >= 4:
            base += 0.6
        elif math_kw >= 2:
            base += 0.5
        elif math_kw >= 1:
            base += 0.2

        if features.get("has_problem_solution_signal", False):
            base += 0.4

        if features.get("has_proof_signal", False):
            base += 0.3

        number_count = features.get("number_count", 0)
        if number_count >= 10:
            base += 0.3
        elif number_count >= 5:
            base += 0.2

        symbol_count = features.get("symbol_count", 0)
        if symbol_count >= 10:
            base += 0.3
        elif symbol_count >= 5:
            base += 0.2

        variable_count = features.get("variable_like_count", 0)
        if variable_count >= 8:
            base += 0.2

        # 负向信号
        boilerplate = features.get("boilerplate_hits", 0)
        if boilerplate >= 5:
            base -= 0.8
        elif boilerplate >= 3:
            base -= 0.5
        elif boilerplate >= 1:
            base -= 0.2

        link_ratio = features.get("link_ratio", 0)
        if link_ratio >= 0.3:
            base -= 0.6
        elif link_ratio >= 0.2:
            base -= 0.4
        elif link_ratio >= 0.1:
            base -= 0.2

        repeat_ratio = features.get("repeat_ratio", 0)
        if repeat_ratio >= 0.4:
            base -= 0.5
        elif repeat_ratio >= 0.3:
            base -= 0.3

        # 文本太短扣分
        char_len = features.get("char_len", 0)
        if char_len < 80:
            base -= 0.3
        elif char_len < 150:
            base -= 0.1

        # 长度合理的文本加分（OpenWebMath 文本通常较长）
        if char_len >= 500:
            base += 0.2
        if char_len >= 1000:
            base += 0.1

        # 纯文字数学补偿：有数学关键词/数字/变量但无 LaTeX 的文本
        # 避免对网页数学（无 LaTeX 但有实质内容）打分过低
        if latex_cmd == 0 and (math_kw >= 1 or number_count >= 3):
            if variable_count >= 3:
                base += 0.3
            if number_count >= 5:
                base += 0.2

        return max(0.0, min(5.0, round(base, 4)))


# ===========================================================================
# HFClassifierScorer — 预留真实模型接口
# ===========================================================================

class HFClassifierScorer(BaseScorer):
    """
    基于 Hugging Face 模型的评分器（如 openwebmath-classifier）。

    预留接口，需要安装 transformers 和 torch 才能使用。
    支持本地路径和远程模型名。
    """

    def __init__(self, model_name_or_path: str, device: str = "auto", batch_size: int = 32):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self._pipeline = None

    def _load_model(self):
        """延迟加载模型"""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "使用 HFClassifierScorer 需要安装 transformers 和 torch: "
                "pip install transformers torch"
            )

        self._pipeline = hf_pipeline(
            "text-classification",
            model=self.model_name_or_path,
            device=self.device,
            batch_size=self.batch_size,
            truncation=True,
            max_length=512,
        )

    def score(self, text: str, features: Dict | None = None) -> float:
        self._load_model()
        assert self._pipeline is not None
        result = self._pipeline(text[:2048])  # 截断过长文本
        # 假设模型输出 label 格式为 "LABEL_0" ~ "LABEL_5" 或直接分数
        if result and isinstance(result, list):
            item = result[0]
            label = item.get("label", "")
            score_val = item.get("score", 0.0)
            # 尝试从 label 中提取数字
            if label.startswith("LABEL_"):
                try:
                    return float(label.split("_")[1])
                except (ValueError, IndexError):
                    pass
            return score_val * 5.0  # 归一化到 0-5
        return 0.0

    def score_batch(self, texts: List[str], features_list: List[Dict] | None = None) -> List[float]:
        self._load_model()
        assert self._pipeline is not None
        truncated = [t[:2048] for t in texts]
        results = self._pipeline(truncated)
        scores = []
        for item in results:
            if isinstance(item, list):
                item = item[0]
            label = item.get("label", "")
            score_val = item.get("score", 0.0)
            if label.startswith("LABEL_"):
                try:
                    scores.append(float(label.split("_")[1]))
                    continue
                except (ValueError, IndexError):
                    pass
            scores.append(score_val * 5.0)
        return scores


# ===========================================================================
# 路由函数
# ===========================================================================

def route_segment(
    score: float,
    features: Dict,
    config: PipelineConfig,
) -> str:
    """
    根据评分和特征决定段落的处理路径。

    返回:
        "keep" — 直接保留轻清洗文本
        "drop" — 直接丢弃
        "llm"  — 送 LLM 处理
    """
    strong_math = (
        features.get("latex_cmd_count", 0) >= 2
        or features.get("math_keyword_count", 0) >= 2
        or features.get("has_problem_solution_signal", False)
        or features.get("has_proof_signal", False)
        or features.get("symbol_count", 0) >= 5
    )

    # 高分直接保留
    if score >= config.clf_keep:
        return "keep"

    # 高分 + 强数学信号也直接保留
    if score >= config.clf_keep_with_math and strong_math:
        return "keep"

    # 低分且数学信号弱 → 丢弃
    if score < config.clf_drop and not strong_math:
        return "drop"

    # 其余灰区 → LLM
    return "llm"


# ===========================================================================
# 工厂函数
# ===========================================================================

def create_scorer(scorer_type: str = "dummy", **kwargs) -> BaseScorer:
    """
    创建评分器实例。

    参数:
        scorer_type: "dummy" 或 "hf"
        **kwargs: 传给具体评分器的参数
    """
    if scorer_type == "dummy":
        return DummyScorer()
    elif scorer_type == "hf":
        model_path = kwargs.get("model_name_or_path", "")
        if not model_path:
            raise ValueError("HFClassifierScorer 需要 model_name_or_path 参数")
        return HFClassifierScorer(
            model_name_or_path=model_path,
            device=kwargs.get("device", "auto"),
            batch_size=kwargs.get("batch_size", 32),
        )
    else:
        raise ValueError(f"未知的评分器类型: {scorer_type}")
