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
"""JustRL-II's training and validation sets, read from the verl parquets that train it.

verl06/recipe/justrl2/prepare_data.sh writes them:

* ``ultradata-rl-math.parquet``: UltraData-RL-2609 (Math), 32,412 rows.
* ``aime-2024.parquet`` / ``aime-2025.parquet`` / ``aime-2026.parquet``: 30 problems each.

Each row is verl's RL schema: ``prompt`` is a single user turn with no system prompt, already
ending in "Please reason step by step, and put your final answer within \\boxed{}.", and the
label is ``reward_model.ground_truth``. The user turn is taken as-is, so run with
``data.prompt_file=null``; a prompt template would wrap it a second time.

``val_data_paths`` maps a name to each validation parquet (``aime-2024`` -> ...); the name
becomes the rows' ``task_name``, which ``validate`` uses to report one accuracy per year as
verl's ``val-core/<data_source>`` does.
Every validation problem is repeated ``val_repeat`` times, the counterpart of verl's
``val_kwargs.n``.
"""

from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


def format_verl_row(data: dict[str, Any], task_name: str) -> dict[str, Any]:
    """One verl RL row -> NeMo RL's [user, assistant=ground_truth] message pair."""
    assert len(data["prompt"]) == 1 and data["prompt"][0]["role"] == "user", (
        f"expected a single user turn, got roles {[m['role'] for m in data['prompt']]}"
    )
    return {
        "messages": [
            {"role": "user", "content": data["prompt"][0]["content"]},
            {"role": "assistant", "content": data["reward_model"]["ground_truth"]},
        ],
        "task_name": task_name,
    }


def _load_parquet(path: str, task_name: str) -> Dataset:
    ds = load_dataset("parquet", data_files=path, split="train")
    return ds.map(
        format_verl_row,
        remove_columns=ds.column_names,
        fn_kwargs={"task_name": task_name},
    )


class JustRL2MathDataset(RawDataset):
    def __init__(
        self,
        train_data_path: str,
        val_data_paths: dict[str, str],
        val_repeat: int,
        seed: int = 42,
    ) -> None:
        """Load JustRL-II's training parquet and its AIME validation parquets.

        Args:
            train_data_path: verl parquet of the training set.
            val_data_paths: validation set name -> verl parquet; the name keys its accuracy.
            val_repeat: copies of each validation problem (verl's ``val_kwargs.n``).
            seed: shuffle seed for the training set.
        """
        self.task_name = "JustRL2Math"
        train_ds = _load_parquet(train_data_path, self.task_name).shuffle(seed=seed)
        val_ds = concatenate_datasets(
            [_load_parquet(path, name) for name, path in val_data_paths.items()]
        )
        val_ds = val_ds.select(
            [i for i in range(len(val_ds)) for _ in range(val_repeat)]
        )
        self.formatted_ds = {"train": train_ds, "validation": val_ds}
