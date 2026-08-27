"""Consolidate automodel DCP-sharded safetensors checkpoint into a standard HF dir."""

import argparse
import json
import os
import shutil

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files,
)
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-step-dir", required=True, help="e.g. .../step_45")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", required=True, help="base model path for tokenizer")
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    src = os.path.join(args.ckpt_step_dir, "policy", "weights", "model")
    meta = os.path.join(src, ".hf_metadata")
    with open(os.path.join(meta, "fqn_to_file_index_mapping.json")) as f:
        mapping = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    consolidate_safetensors_files(
        input_dir=src,
        output_dir=args.out_dir,
        fqn_to_index_mapping=mapping,
        num_threads=args.num_threads,
    )

    for name in ("config.json", "generation_config.json"):
        p = os.path.join(meta, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out_dir, name))

    AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True).save_pretrained(
        args.out_dir
    )
    print(f"Saved HF checkpoint to: {args.out_dir}")


if __name__ == "__main__":
    main()
