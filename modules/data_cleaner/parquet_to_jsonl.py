"""
parquet_to_jsonl.py — 将 OpenWebMath parquet 转为 pipeline 输入的 JSONL

用法:
    python parquet_to_jsonl.py \
        --input "/path/to/openwebmath-4plus/train-00000-of-00064.parquet" \
        --output data/input.jsonl

    # 处理多个文件（glob）
    python parquet_to_jsonl.py \
        --input "/path/to/openwebmath-4plus/*.parquet" \
        --output data/input.jsonl

    # 只保留 int_score=5
    python parquet_to_jsonl.py \
        --input "..." --output data/input.jsonl --min-score 5

注意:
    核心只输出 id + text，不依赖 OpenWebMath 专有字段。
    其余字段可选保留到 JSONL 里但 pipeline 不使用。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="OpenWebMath parquet → pipeline JSONL")
    p.add_argument("--input", required=True,
                    help="parquet 路径或 glob 模式（如 *.parquet）")
    p.add_argument("--output", required=True, help="输出 JSONL 路径")
    p.add_argument("--min-score", type=int, default=None,
                    help="只保留 int_score >= 该值（可选）")
    p.add_argument("--min-tokens", type=int, default=50,
                    help="最小 token 数（默认 50）")
    p.add_argument("--max-tokens", type=int, default=8192,
                    help="最大 token 数（默认 8192）")
    p.add_argument("--limit", type=int, default=None,
                    help="最多输出条数")
    return p.parse_args()


def main():
    args = parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        print(f"未找到文件: {args.input}")
        sys.exit(1)

    print(f"找到 {len(files)} 个 parquet 文件")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = 0
    kept = 0
    skipped_score = 0
    skipped_short = 0
    skipped_long = 0

    with open(args.output, "w", encoding="utf-8") as f_out:
        for fi, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            print(f"[{fi+1}/{len(files)}] {fname} ...", end=" ", flush=True)

            df = pd.read_parquet(fpath)
            n = len(df)
            total += n
            local_kept = 0

            for idx in range(n):
                row = df.iloc[idx]

                # 过滤 int_score
                if args.min_score is not None:
                    if row.get("int_score", 0) < args.min_score:
                        skipped_score += 1
                        continue

                # 过滤 token 数
                token_count = row.get("token_count", 0)
                if token_count < args.min_tokens:
                    skipped_short += 1
                    continue
                if token_count > args.max_tokens:
                    skipped_long += 1
                    continue

                text = str(row["text"]).strip()
                if not text:
                    continue

                # 构造 ID：文件名 + 行号
                doc_id = f"{os.path.splitext(fname)[0]}_{idx}"

                record = {
                    "id": doc_id,
                    "text": text,
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                local_kept += 1

                if args.limit and kept >= args.limit:
                    print(f"保留 {local_kept}/{n}")
                    break

            print(f"保留 {local_kept}/{n}")

            if args.limit and kept >= args.limit:
                break

    print()
    print("=" * 50)
    print(f"文件数:          {len(files)}")
    print(f"总行数:          {total}")
    print(f"score 过滤:      {skipped_score}")
    print(f"太短过滤:        {skipped_short}")
    print(f"太长过滤:        {skipped_long}")
    print(f"最终输出:        {kept}")
    print(f"输出文件:        {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
