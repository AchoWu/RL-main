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
"""Evaluate one HF checkpoint exactly as the OPD run's pre-training validation does.

    conda activate opd
    python eval_model.py                 # settings at the bottom of this file
    python eval_model.py policy.model_name=/path/to/hf_model   # or override here

Only inference: vLLM loads the checkpoint itself, no policy or teacher is built. Everything
that decides the score is shared with training rather than copied:

* hyperparameters: the training yaml (distillation_math_justrl2.yaml) is loaded as-is, plus
  the few eval-relevant overrides train_opd_justrl2_data.sh passes on its command line;
* data, prompt and grader: run_distillation_math.setup_data;
* the validation loop and its metrics: distillation.validate.

So the numbers match the run's ``val:accuracy`` / ``val:accuracy/<year>``. Training writes DCP
checkpoints; convert one to HF first with examples/converters/convert_dcp_to_hf.py.
"""

import json
import os
import sys
from pprint import pformat


def main(
    config_path: str,
    overrides: list[str],
    seed: int,
    save_dir: str | None,
    val_data_paths: dict[str, str] | None,
    val_repeat: int | None,
):
    # Imported here, not at the top: ray_actor_environment_registry reads
    # NEMO_RL_PY_EXECUTABLES_SYSTEM at import time, and __main__ sets it first.
    import ray
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader

    from examples.run_distillation_math import setup_data  # registers `mul` too
    from nemo_rl.algorithms.distillation import validate
    from nemo_rl.algorithms.utils import get_tokenizer, set_seed
    from nemo_rl.data.collate_fn import rl_collate_fn
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.models.generation.vllm import VllmGeneration
    from nemo_rl.utils.config import load_config, parse_hydra_overrides

    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    custom_val_set = val_data_paths is not None or val_repeat is not None
    if val_data_paths is not None:
        config["data"]["val_data_paths"] = val_data_paths
    if val_repeat is not None:
        config["data"]["val_repeat"] = val_repeat
    print("Final config:\n" + pformat(config))
    print(f"\n\nload model from {config['policy']['model_name']}\n\n")

    set_seed(seed)
    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    generation_config = configure_generation_config(
        config["policy"]["generation"], tokenizer, is_eval=True
    )
    generation_config["model_name"] = config["policy"]["model_name"]
    config["policy"]["generation"] = generation_config

    _, val_dataset, _, val_task_to_env = setup_data(
        tokenizer, config["data"], config["env"], seed
    )
    assert val_dataset is not None, "the data config produced no validation set"
    if custom_val_set:
        # validate() stops after max_val_samples // val_batch_size batches, and the yaml's 720
        # is sized for the AIME sets: cover the whole custom set instead.
        batch_size = config["distillation"]["val_batch_size"]
        config["distillation"]["max_val_samples"] = (
            -(-len(val_dataset) // batch_size) * batch_size
        )
    val_dataloader = StatefulDataLoader(
        val_dataset,
        batch_size=config["distillation"]["val_batch_size"],
        shuffle=False,
        collate_fn=rl_collate_fn,
    )

    cluster_config = config["cluster"]
    cluster = RayVirtualCluster(
        name="opd_eval_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    generation = VllmGeneration(cluster=cluster, config=generation_config)

    val_metrics, timings = validate(
        generation, val_dataloader, tokenizer, val_task_to_env, 0, config
    )

    result = {
        "model_name": config["policy"]["model_name"],
        "val_metrics": val_metrics,
        "total_validation_time": timings.get("total_validation_time"),
    }
    print("\n" + json.dumps(result, indent=2))
    if save_dir:
        # Named after the model actually evaluated, so a command-line policy.model_name=
        # override lands in its own directory rather than MODEL_PATH's.
        model_tag = os.path.basename(os.path.normpath(config["policy"]["model_name"]))
        save_dir = save_dir.format(model=model_tag)
        save_path = os.path.join(save_dir, f"{model_tag}.json")
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved metrics to {save_path}")

    generation.shutdown()
    for env in {id(e): e for e in val_task_to_env.values()}.values():
        ray.get(env.shutdown.remote())


if __name__ == "__main__":
    REPO_ROOT = "/group/40092/howu/RL-main"
    # ---- hyperparameters: the training yaml ----------------------------------
    CONFIG_PATH = "/group/40092/howu/RL-main/examples/configs/distillation_math_justrl2.yaml"

    # ---- what to evaluate ---------------------------------------------------------------
    # MODEL_PATH = "/dev/shm/llms/DeepSeek-R1-Distill-Qwen-1.5B/"
    MODEL_PATH = "/group/40143/howu/verl/results/justrl2_step_150/"
    # Metrics go to <SAVE_DIR>/<model dir name>.json; None prints only. {model} is filled in
    # with the dir name of the model actually evaluated (after command-line overrides).
    SAVE_DIR = os.path.join(REPO_ROOT, "results/eval-{model}")
    SEED = 42  # run_distillation_math.main passes 42 to setup_data too

    # ---- which sets to evaluate on ------------------------------------------------------
    # None: the training yaml's data.val_data_paths / data.val_repeat, i.e. AIME 2024/25/26,
    # each problem 8 times -- the run's own validation.
    # Or name -> parquet; each name gets its own accuracy/<name>. Files must be verl RL parquets
    # (a single user turn in `prompt`, label in reward_model.ground_truth), like the ones
    # verl06/recipe/justrl2/prepare_data.sh writes.
    # VAL_DATA_PATHS = None
    VAL_DATA_PATHS = {
        "aime-2024": "/group/40143/howu/verl/data/aime-2024.parquet",
        "aime-2025": "/group/40143/howu/verl/data/aime-2025.parquet",
        "aime-2026": "/group/40143/howu/verl/data/aime-2026.parquet",
    }
    VAL_REPEAT = 8  # copies of each problem (verl's val_kwargs.n); None = the yaml's 8

    # Only what train_opd_justrl2_data.sh sets on its command line (not in the yaml) and that
    # affects validation. Keep these in step with that script.
    OVERRIDES = [
        f"policy.model_name={MODEL_PATH}",
        "policy.max_total_sequence_length=16384",  # response budget and vLLM max_model_len
        "cluster.gpus_per_node=8",
        # Training shares the GPUs with student and teacher; alone, vLLM can take more KV cache.
        # Speed only, the samples are unchanged. Uncomment to use it.
        # "policy.generation.vllm_cfg.gpu_memory_utilization=0.85",
    ]
    # Anything on the command line is applied last, e.g. policy.model_name=<other model>.
    OVERRIDES += sys.argv[1:]

    # ---- environment (what train_opd_justrl2_data.sh exports) ---------------------------
    # Ray workers use this Python instead of building uv venvs.
    os.environ["NEMO_RL_PY_EXECUTABLES_SYSTEM"] = "1"
    # nemo_rl / examples, when not installed into the env: sys.path for this process,
    # PYTHONPATH for the Ray workers, which inherit the environment.
    sys.path.insert(0, REPO_ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (REPO_ROOT, os.environ.get("PYTHONPATH")) if p
    )

    main(CONFIG_PATH, OVERRIDES, SEED, SAVE_DIR, VAL_DATA_PATHS, VAL_REPEAT)

