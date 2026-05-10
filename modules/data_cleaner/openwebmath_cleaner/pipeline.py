"""
pipeline.py — 主流程编排

实现完整决策链：
段级切分 → 轻规则清洗 → classifier 初筛 → LLM 处理 → 数学感知校验 → 复评分 → 保留/回退/丢弃
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .config import PipelineConfig
from .schemas import (
    AuditRecord,
    LLMDecision,
    Route,
    RunSummary,
    Segment,
    TrainOutputRecord,
)
from .segmenter import segment_document
from .light_rules import apply_light_rules
from .scorer import BaseScorer, route_segment
from .llm_processor import BaseLLMProcessor
from .validator import validate

log = logging.getLogger("openwebmath_cleaner.pipeline")


def process_segment(
    segment: Segment,
    config: PipelineConfig,
    scorer: BaseScorer,
    processor: BaseLLMProcessor,
) -> Tuple[TrainOutputRecord | None, AuditRecord]:
    """
    处理单个段落，返回训练输出（可能为空）和审计记录。
    """
    seg_id = segment.segment_id

    # ---- 1. 轻规则清洗 ----
    rule_result = apply_light_rules(segment, config)

    # ---- 2. 规则直接丢弃 ----
    if rule_result.hard_drop or not rule_result.cleaned_text.strip():
        audit = AuditRecord(
            id=seg_id,
            route=Route.DROPPED.value,
            fail_reasons=["hard_drop_by_rule"],
        )
        return None, audit

    cleaned_text = rule_result.cleaned_text
    features = rule_result.feature_dict

    # ---- 3. classifier 初筛 ----
    raw_score = scorer.score(cleaned_text, features)

    # ---- 4. 路由决策 ----
    route = route_segment(raw_score, features, config)

    if route == "keep":
        # 直接保留轻清洗文本
        output = TrainOutputRecord(
            id=seg_id,
            text=cleaned_text,
            quality_score=raw_score,
        )
        audit = AuditRecord(
            id=seg_id,
            route=Route.KEPT_LIGHT_CLEANED.value,
            classifier_score_raw=raw_score,
        )
        return output, audit

    if route == "drop":
        # 直接丢弃
        audit = AuditRecord(
            id=seg_id,
            route=Route.DROPPED.value,
            classifier_score_raw=raw_score,
            fail_reasons=["low_score_weak_math"],
        )
        return None, audit

    # ---- 5. 灰区 → LLM 处理 ----
    try:
        llm_result = processor.process(cleaned_text, meta=features)
    except Exception as e:
        log.warning("LLM 处理异常 [%s]: %s，回退到轻清洗文本", seg_id, e)
        # LLM 出错，根据原始分数决定回退还是丢弃
        if raw_score >= config.fallback_min_raw_score:
            output = TrainOutputRecord(
                id=seg_id,
                text=cleaned_text,
                quality_score=raw_score,
            )
            audit = AuditRecord(
                id=seg_id,
                route=Route.FALLBACK_TO_LIGHT_CLEANED.value,
                classifier_score_raw=raw_score,
                llm_decision="error",
                fail_reasons=[f"llm_error: {e}"],
            )
            return output, audit
        else:
            audit = AuditRecord(
                id=seg_id,
                route=Route.DROPPED.value,
                classifier_score_raw=raw_score,
                llm_decision="error",
                fail_reasons=[f"llm_error: {e}"],
            )
            return None, audit

    # LLM 决定丢弃
    if llm_result.decision == LLMDecision.DROP:
        audit = AuditRecord(
            id=seg_id,
            route=Route.DROPPED.value,
            classifier_score_raw=raw_score,
            llm_decision=llm_result.decision.value,
            fail_reasons=["llm_decided_drop"],
        )
        return None, audit

    # ---- 6. 数学感知校验 + 复评分 ----
    val_result = validate(
        raw_text=cleaned_text,
        processed_text=llm_result.text,
        score_raw=raw_score,
        scorer=scorer,
        config=config,
    )

    if val_result.validation_pass:
        # 校验通过，保留 LLM 结果
        output = TrainOutputRecord(
            id=seg_id,
            text=llm_result.text,
            quality_score=val_result.score_post,
        )
        audit = AuditRecord(
            id=seg_id,
            route=Route.KEPT_LLM_PROCESSED.value,
            classifier_score_raw=raw_score,
            classifier_score_post=val_result.score_post,
            llm_decision=llm_result.decision.value,
            validation_pass=True,
        )
        return output, audit

    # ---- 7. 校验失败 → 回退或丢弃 ----
    if raw_score >= config.fallback_min_raw_score:
        # 回退到轻清洗文本
        output = TrainOutputRecord(
            id=seg_id,
            text=cleaned_text,
            quality_score=raw_score,
        )
        audit = AuditRecord(
            id=seg_id,
            route=Route.FALLBACK_TO_LIGHT_CLEANED.value,
            classifier_score_raw=raw_score,
            classifier_score_post=val_result.score_post,
            llm_decision=llm_result.decision.value,
            validation_pass=False,
            fail_reasons=val_result.fail_reasons,
        )
        return output, audit
    else:
        # 丢弃
        audit = AuditRecord(
            id=seg_id,
            route=Route.DROPPED.value,
            classifier_score_raw=raw_score,
            classifier_score_post=val_result.score_post,
            llm_decision=llm_result.decision.value,
            validation_pass=False,
            fail_reasons=val_result.fail_reasons,
        )
        return None, audit


def process_document(
    doc_id: str,
    text: str,
    config: PipelineConfig,
    scorer: BaseScorer,
    processor: BaseLLMProcessor,
) -> Tuple[List[TrainOutputRecord], List[AuditRecord], Dict[str, int]]:
    """
    处理一篇文档，返回训练输出列表、审计记录列表和统计信息。

    参数:
        doc_id: 文档 ID
        text: 文档原始文本
        config: 配置
        scorer: 评分器
        processor: LLM 处理器

    返回:
        (outputs, audits, stats)
        stats 包含本文档的各路径计数
    """
    outputs: List[TrainOutputRecord] = []
    audits: List[AuditRecord] = []
    stats = {
        "segments": 0,
        "rule_dropped": 0,
        "direct_kept": 0,
        "sent_to_llm": 0,
        "llm_passed": 0,
        "llm_failed_fallback": 0,
        "final_kept": 0,
        "final_dropped": 0,
    }

    # 段级切分
    segments = segment_document(doc_id, text, config)
    stats["segments"] = len(segments)

    for segment in segments:
        try:
            output, audit = process_segment(segment, config, scorer, processor)
        except Exception as e:
            log.error("处理段落异常 [%s]: %s", segment.segment_id, e)
            audit = AuditRecord(
                id=segment.segment_id,
                route=Route.DROPPED.value,
                fail_reasons=[f"unexpected_error: {e}"],
            )
            output = None

        audits.append(audit)

        if output is not None:
            outputs.append(output)
            stats["final_kept"] += 1
        else:
            stats["final_dropped"] += 1

        # 统计路径
        route = audit.route
        if route == Route.DROPPED.value:
            if "hard_drop" in str(audit.fail_reasons):
                stats["rule_dropped"] += 1
            elif audit.llm_decision:
                stats["sent_to_llm"] += 1
            else:
                stats["rule_dropped"] += 1
        elif route == Route.KEPT_LIGHT_CLEANED.value:
            stats["direct_kept"] += 1
        elif route == Route.KEPT_LLM_PROCESSED.value:
            stats["sent_to_llm"] += 1
            stats["llm_passed"] += 1
        elif route == Route.FALLBACK_TO_LIGHT_CLEANED.value:
            stats["sent_to_llm"] += 1
            stats["llm_failed_fallback"] += 1

    return outputs, audits, stats


def run_pipeline(
    documents: list[dict],
    config: PipelineConfig,
    scorer: BaseScorer,
    processor: BaseLLMProcessor,
) -> Tuple[List[TrainOutputRecord], List[AuditRecord], RunSummary]:
    """
    对一批文档执行完整 pipeline。

    参数:
        documents: 文档列表，每个 dict 至少包含 "id" 和 "text"
        config: 配置
        scorer: 评分器
        processor: LLM 处理器

    返回:
        (all_outputs, all_audits, summary)
    """
    all_outputs: List[TrainOutputRecord] = []
    all_audits: List[AuditRecord] = []
    summary = RunSummary()

    summary.total_documents = len(documents)

    for doc in documents:
        doc_id = doc.get("id", "unknown")
        text = doc.get("text", "")

        if not text.strip():
            log.debug("跳过空文档: %s", doc_id)
            continue

        outputs, audits, stats = process_document(
            doc_id=doc_id,
            text=text,
            config=config,
            scorer=scorer,
            processor=processor,
        )

        all_outputs.extend(outputs)
        all_audits.extend(audits)

        # 累计统计
        summary.total_segments += stats["segments"]
        summary.rule_dropped += stats["rule_dropped"]
        summary.direct_kept += stats["direct_kept"]
        summary.sent_to_llm += stats["sent_to_llm"]
        summary.llm_passed_validation += stats["llm_passed"]
        summary.llm_failed_fallback += stats["llm_failed_fallback"]
        summary.final_kept += stats["final_kept"]
        summary.final_dropped += stats["final_dropped"]

    log.info(
        "Pipeline 完成: %d 文档, %d 段落, 保留 %d, 丢弃 %d",
        summary.total_documents,
        summary.total_segments,
        summary.final_kept,
        summary.final_dropped,
    )

    return all_outputs, all_audits, summary
