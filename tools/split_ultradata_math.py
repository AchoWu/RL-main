#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split UltraData-RL-Math-2609.jsonl into disjoint train/validation JSONL files.

``ResponseDataset`` (``data.dataset_name=ResponseDataset``) loads train and
validation from two separate paths and does not implement the
``validation_source=train_holdout`` holdout that ``DAPOMath17KProcessed``
provides, so the split has to be materialized on disk up front.

The shuffle is seeded, so re-running this reproduces the exact same partition
and the two splits stay strictly disjoint.

Usage:
    python tools/split_ultradata_math.py \
        --input UltraData-RL-Math-2609.jsonl \
        --out-dir . \
        --val-num-samples 500 \
        --seed 42
"""

import argparse
import json
import os
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="UltraData-RL-Math-2609.jsonl",
        help="Source JSONL with one record per line.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory the two split files are written to.",
    )
    parser.add_argument(
        "--prefix",
        default="ultradata_math",
        help="Basename prefix for the written files.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=500,
        help="Number of records held out for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the holdout partition shuffle.",
    )
    return parser.parse_args()


def write_jsonl(path: str, rows: list[dict]) -> None:
    # newline="" keeps the "\n" below from becoming CRLF on Windows, so the
    # files are byte-identical regardless of which platform split them.
    with open(path, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not 0 < args.val_num_samples < len(rows):
        raise ValueError(
            f"--val-num-samples must be in [1, {len(rows) - 1}]; "
            f"got {args.val_num_samples} for {len(rows)} records"
        )

    random.Random(args.seed).shuffle(rows)
    val_rows = rows[: args.val_num_samples]
    train_rows = rows[args.val_num_samples :]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, f"{args.prefix}_train.jsonl")
    val_path = os.path.join(args.out_dir, f"{args.prefix}_val.jsonl")
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    # Plain ASCII: a GBK console (default on Windows CN) cannot encode "✓".
    print(f"[ok] {len(train_rows)} train -> {train_path}")
    print(f"[ok] {len(val_rows)} val   -> {val_path}")


if __name__ == "__main__":
    main()
