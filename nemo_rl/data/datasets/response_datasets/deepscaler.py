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


from typing import Any

from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


def format_math(
    data: dict[str, str | float | int], task_name: str = "DeepScaler"
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data["answer"],
            },
        ],
        "task_name": task_name,
    }


def prepare_deepscaler_dataset(
    seed: int = 42,
    task_name: str = "DeepScaler",
    validation_source: str = "aime_2024",
    validation_num_samples: int = 1000,
    validation_seed: int = 42,
) -> dict[str, Dataset | None]:
    """Load DeepScaler and build its configured validation set.

    ``validation_source="train_holdout"`` creates a reproducible random
    partition with ``validation_seed``. Validation examples are removed from
    the training dataset, so train and validation remain strictly disjoint.
    """
    # Load the original dataset for training
    train_ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")

    if validation_source == "aime_2024":
        val_ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif validation_source == "train_holdout":
        if not 0 < validation_num_samples < len(train_ds):
            raise ValueError(
                "validation_num_samples must be in [1, len(train_ds) - 1] when "
                f"validation_source='train_holdout'; got {validation_num_samples} "
                f"for {len(train_ds)} training samples"
            )
        partitioned_ds = train_ds.shuffle(seed=validation_seed)
        val_ds = partitioned_ds.select(range(validation_num_samples))
        train_ds = partitioned_ds.select(
            range(validation_num_samples, len(partitioned_ds))
        )
    else:
        raise ValueError(
            "Unknown DeepScaler validation_source "
            f"{validation_source!r}; expected 'aime_2024' or 'train_holdout'"
        )

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(
        format_math,
        remove_columns=train_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )
    val_formatted = val_ds.map(
        format_math,
        remove_columns=val_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )

    if validation_source == "aime_2024":
        # Preserve the historical DeepScaleR evaluation behavior for existing recipes.
        val_repeated = []
        for _ in range(16):
            val_repeated.extend(val_formatted)
        val_formatted = val_formatted.from_list(val_repeated)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class DeepScalerDataset(RawDataset):
    def __init__(
        self,
        seed: int = 42,
        validation_source: str = "aime_2024",
        validation_num_samples: int = 1000,
        validation_seed: int = 42,
    ) -> None:
        """Initialize the DeepScaler dataset with train/test split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.task_name = "DeepScaler"
        self.formatted_ds = prepare_deepscaler_dataset(
            seed=seed,
            task_name=self.task_name,
            validation_source=validation_source,
            validation_num_samples=validation_num_samples,
            validation_seed=validation_seed,
        )
