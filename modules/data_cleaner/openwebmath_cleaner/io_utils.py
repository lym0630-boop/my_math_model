"""
io_utils.py — JSONL 读写工具

支持逐行流式读写、断点续跑、错误容忍。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Set

log = logging.getLogger("openwebmath_cleaner.io")


def read_jsonl(path: str, limit: int | None = None) -> Iterator[dict]:
    """
    逐行读取 JSONL 文件。

    参数:
        path: 文件路径
        limit: 最多读取条数（None 表示全部）

    Yields:
        解析后的 dict
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                yield record
                count += 1
                if limit is not None and count >= limit:
                    return
            except json.JSONDecodeError as e:
                log.warning("JSONL 解析错误 [行 %d]: %s", line_no, e)
                continue


def write_jsonl_line(f, record: dict) -> None:
    """向文件对象写入一行 JSONL"""
    line = json.dumps(record, ensure_ascii=False)
    f.write(line + "\n")


def load_processed_ids(output_path: str) -> Set[str]:
    """
    从已有输出文件中加载已处理的 ID 集合。
    用于断点续跑：跳过已经处理过的文档。

    参数:
        output_path: 输出文件路径

    返回:
        已处理的 ID 集合
    """
    processed_ids: Set[str] = set()
    path = Path(output_path)

    if not path.exists():
        return processed_ids

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    seg_id = record.get("id", "")
                    if seg_id:
                        # 从段级 ID 提取文档级 ID
                        # 格式: doc_id_seg_XXXX → doc_id
                        parts = seg_id.rsplit("_seg_", 1)
                        if parts:
                            processed_ids.add(parts[0])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log.warning("读取已处理 ID 失败: %s", e)

    return processed_ids


def write_summary(path: str, summary_dict: dict) -> None:
    """写入 summary.json"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    log.info("统计摘要已写入: %s", path)
