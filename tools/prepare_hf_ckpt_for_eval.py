"""Consolidate an OPD DCP checkpoint into an inference-ready HF dir.

Wraps tools/consolidate_opd_ckpt.py and additionally:
  - casts fp32 master weights to bfloat16 (halves size and load time; training
    ran in bf16 anyway and vLLM loads with dtype=bfloat16 regardless)
  - flips use_cache back to True (training writes False, inference needs KV cache)
"""

import argparse
import json
import os
import shutil

import torch
from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files,
)
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


def consolidate(ckpt_step_dir: str, out_dir: str, tokenizer: str, num_threads: int = 8) -> None:
    src = os.path.join(ckpt_step_dir, "policy", "weights", "model")
    meta = os.path.join(src, ".hf_metadata")
    with open(os.path.join(meta, "fqn_to_file_index_mapping.json")) as f:
        mapping = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    consolidate_safetensors_files(
        input_dir=src,
        output_dir=out_dir,
        fqn_to_index_mapping=mapping,
        num_threads=num_threads,
    )

    for name in ("config.json", "generation_config.json"):
        p = os.path.join(meta, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out_dir, name))

    AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True).save_pretrained(out_dir)


def to_bf16_and_fix_config(out_dir: str) -> None:
    weights = os.path.join(out_dir, "model-00001-of-00001.safetensors")
    sd = load_file(weights)
    cast = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    assert all(torch.isfinite(v).all() for v in cast.values()), "non-finite weights after cast"
    save_file(cast, weights, metadata={"format": "pt"})
    os.chmod(weights, 0o644)  # save_file defaults to 0600

    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        index["metadata"]["total_size"] = sum(v.numel() * v.element_size() for v in cast.values())
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    config_path = os.path.join(out_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["dtype"] = "bfloat16"
    config["torch_dtype"] = "bfloat16"  # old-style key, for vLLM/transformers back-compat
    config["use_cache"] = True
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-step-dir", required=True, help="e.g. .../step_45")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", required=True, help="base model path for tokenizer")
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    consolidate(args.ckpt_step_dir, args.out_dir, args.tokenizer, args.num_threads)
    to_bf16_and_fix_config(args.out_dir)
    print(f"Saved inference-ready HF checkpoint to: {args.out_dir}")


if __name__ == "__main__":
    main()
