"""
cli.py — 命令行入口

支持：
    python -m openwebmath_cleaner.cli --input data/input.jsonl --output data/cleaned.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import PipelineConfig, load_config
from .io_utils import read_jsonl, write_jsonl_line, load_processed_ids, write_summary
from .pipeline import process_document
from .scorer import create_scorer
from .llm_processor import create_processor
from .schemas import RunSummary

log = logging.getLogger("openwebmath_cleaner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="openwebmath_cleaner",
        description="通用数学语料二次清洗 pipeline",
    )
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="输出训练数据 JSONL 路径")
    parser.add_argument("--summary", default=None, help="统计摘要 JSON 输出路径")
    parser.add_argument("--audit", default=None, help="审计日志 JSONL 输出路径")
    parser.add_argument("--config", default=None, help="配置 YAML 文件路径")
    parser.add_argument(
        "--scorer", default="dummy", choices=["dummy", "hf"],
        help="评分器类型 (默认: dummy)",
    )
    parser.add_argument("--scorer-model", default="", help="HF 评分模型路径（仅 --scorer hf 时需要）")
    parser.add_argument(
        "--processor", default="regex", choices=["noop", "regex", "real"],
        help="LLM 处理器类型 (默认: regex)",
    )
    parser.add_argument("--llm-api-base", default="", help="LLM API 地址（仅 --processor real 时需要）")
    parser.add_argument("--llm-api-key", default="", help="LLM API Key")
    parser.add_argument("--llm-model", default="", help="LLM 模型名")
    parser.add_argument("--limit", type=int, default=None, help="最多处理文档数")
    parser.add_argument("--resume", action="store_true", help="断点续跑（跳过已处理的文档）")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 加载配置
    config = load_config(args.config)
    log.info("配置加载完成")

    # 创建评分器
    scorer = create_scorer(
        args.scorer,
        model_name_or_path=args.scorer_model,
    )
    log.info("评分器: %s", type(scorer).__name__)

    # 创建处理器
    processor = create_processor(
        args.processor,
        api_base=args.llm_api_base,
        api_key=args.llm_api_key,
        model_name=args.llm_model,
    )
    log.info("处理器: %s", type(processor).__name__)

    # 断点续跑
    processed_doc_ids = set()
    if args.resume:
        processed_doc_ids = load_processed_ids(args.output)
        if processed_doc_ids:
            log.info("断点续跑: 已跳过 %d 个文档", len(processed_doc_ids))

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.audit:
        Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)

    # 打开输出文件
    output_mode = "a" if args.resume else "w"
    f_output = open(args.output, output_mode, encoding="utf-8")
    f_audit = None
    if args.audit:
        f_audit = open(args.audit, output_mode, encoding="utf-8")

    summary = RunSummary()
    doc_count = 0
    start_time = time.time()

    try:
        for doc in read_jsonl(args.input, limit=args.limit):
            doc_id = doc.get("id", f"doc_{doc_count:06d}")
            text = doc.get("text", "")

            # 跳过已处理
            if doc_id in processed_doc_ids:
                continue

            if not text.strip():
                log.debug("跳过空文档: %s", doc_id)
                continue

            doc_count += 1
            summary.total_documents += 1

            try:
                outputs, audits, stats = process_document(
                    doc_id=doc_id,
                    text=text,
                    config=config,
                    scorer=scorer,
                    processor=processor,
                )
            except Exception as e:
                log.error("处理文档异常 [%s]: %s", doc_id, e)
                continue

            # 写训练输出
            for rec in outputs:
                write_jsonl_line(f_output, rec.to_dict())

            # 写审计日志
            if f_audit:
                for audit in audits:
                    write_jsonl_line(f_audit, audit.to_dict())

            # 累计统计
            summary.total_segments += stats["segments"]
            summary.rule_dropped += stats["rule_dropped"]
            summary.direct_kept += stats["direct_kept"]
            summary.sent_to_llm += stats["sent_to_llm"]
            summary.llm_passed_validation += stats["llm_passed"]
            summary.llm_failed_fallback += stats["llm_failed_fallback"]
            summary.final_kept += stats["final_kept"]
            summary.final_dropped += stats["final_dropped"]

            # 定期打印进度
            if doc_count % 100 == 0:
                elapsed = time.time() - start_time
                log.info(
                    "进度: %d 文档, %d 段保留 / %d 段总计, 耗时 %.1fs",
                    doc_count, summary.final_kept, summary.total_segments, elapsed,
                )

    except KeyboardInterrupt:
        log.warning("用户中断，正在保存已处理结果...")
    finally:
        f_output.close()
        if f_audit:
            f_audit.close()

    # 写统计摘要
    elapsed = time.time() - start_time
    if args.summary:
        summary_dict = summary.to_dict()
        summary_dict["elapsed_seconds"] = round(elapsed, 2)
        write_summary(args.summary, summary_dict)

    # 打印最终统计
    print("\n" + "=" * 60)
    print("清洗完成")
    print(f"  文档总数:        {summary.total_documents}")
    print(f"  段落总数:        {summary.total_segments}")
    print(f"  规则直接丢弃:    {summary.rule_dropped}")
    print(f"  直接保留:        {summary.direct_kept}")
    print(f"  送 LLM:          {summary.sent_to_llm}")
    print(f"  LLM 通过校验:    {summary.llm_passed_validation}")
    print(f"  LLM 失败回退:    {summary.llm_failed_fallback}")
    print(f"  最终保留:        {summary.final_kept}")
    print(f"  最终丢弃:        {summary.final_dropped}")
    print(f"  耗时:            {elapsed:.1f}s")
    print(f"  输出文件:        {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
