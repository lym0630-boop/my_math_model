"""
config.py — 配置管理

从 YAML 文件加载配置，缺失字段使用默认值。
无 pyyaml 时降级为纯默认配置。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

# 尝试导入 yaml，不可用时降级
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class PipelineConfig:
    """pipeline 全局配置 — 所有阈值集中管理"""

    # === 段切分 ===
    min_segment_chars: int = 40
    ideal_min_chars: int = 200
    ideal_max_chars: int = 1500
    hard_max_chars: int = 2200

    # === classifier 阈值 ===
    clf_keep: float = 4.2
    clf_keep_with_math: float = 3.8
    clf_llm_low: float = 3.0
    clf_drop: float = 2.8

    # === 校验阈值 ===
    length_ratio_min: float = 0.45
    length_ratio_max: float = 1.20
    number_recall_min: float = 0.90
    variable_recall_min: float = 0.85
    symbol_recall_min: float = 0.80
    max_clean_score_drop: float = 0.20

    # === 轻规则层 ===
    min_text_chars: int = 30
    max_link_ratio: float = 0.50
    max_repeat_ratio: float = 0.60
    max_boilerplate_for_drop: int = 5

    # === 回退阈值 ===
    fallback_min_raw_score: float = 3.5
    min_post_score_absolute: float = 3.6


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """
    从 YAML 文件加载配置。

    - path 为 None 时返回纯默认配置
    - YAML 中缺失的字段使用默认值
    - 无 pyyaml 库时忽略文件，返回默认配置
    """
    config = PipelineConfig()

    if path is None:
        return config

    if not _HAS_YAML:
        import warnings
        warnings.warn(
            f"pyyaml 未安装，忽略配置文件 {path}，使用默认配置",
            stacklevel=2,
        )
        return config

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 获取 dataclass 字段名集合
    valid_fields = {fld.name for fld in fields(PipelineConfig)}

    for key, value in raw.items():
        if key in valid_fields:
            setattr(config, key, value)

    return config
