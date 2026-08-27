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


def format_dapo_math_17k(
    data: dict[str, str | float | int],
    task_name: str = "DAPOMath17K",
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["prompt"][0]["content"],
            },
            {
                "role": "assistant",
                "content": data["reward_model"]["ground_truth"],
            },
        ],
        "task_name": task_name,
    }


def prepare_dapo_math_17k_dataset(
    seed: int = 42, task_name: str = "DAPOMath17K"
) -> dict[str, Dataset | None]:
    """Load and split the DeepScaler dataset into train and test sets."""
    # Load the original dataset for training
    train_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

    # Load hendrydong/aime24 dataset for validation
    val_ds = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(
        format_dapo_math_17k,
        remove_columns=train_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )
    val_formatted = val_ds.map(
        format_dapo_math_17k,
        remove_columns=val_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class DAPOMath17KDataset(RawDataset):
    def __init__(self, seed: int = 42) -> None:
        """Initialize the DAPO Math 17K dataset with train split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.task_name = "DAPOMath17K"
        self.formatted_ds = prepare_dapo_math_17k_dataset(
            seed=seed, task_name=self.task_name
        )


def format_dapo_math_processed(
    data: dict[str, Any],
    task_name: str = "DAPOMath17KProcessed",
) -> dict[str, list[Any] | str]:
    """Format open-r1/DAPO-Math-17k-Processed, whose ``prompt`` is a plain string.

    This differs from ``format_dapo_math_17k`` above: the BytedTsinghua-SIA
    release stores ``prompt`` as a chat list, the open-r1 one as a bare string.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": data["prompt"],
            },
            {
                "role": "assistant",
                "content": data["reward_model"]["ground_truth"],
            },
        ],
        "task_name": task_name,
    }


def prepare_dapo_math_processed_dataset(
    seed: int = 42,
    task_name: str = "DAPOMath17KProcessed",
    config_name: str = "en",
    validation_source: str = "train_holdout",
    validation_num_samples: int = 500,
    validation_seed: int = 42,
) -> dict[str, Dataset | None]:
    """Load open-r1/DAPO-Math-17k-Processed and build its validation set.

    ``validation_source="train_holdout"`` carves a reproducible random
    partition out of train using ``validation_seed``; those rows are removed
    from train so the two splits stay strictly disjoint.
    """
    train_ds = load_dataset(
        "open-r1/DAPO-Math-17k-Processed", config_name, split="train"
    )

    if validation_source == "aime_2024":
        val_ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        val_formatted = val_ds.map(
            format_math_processed_aime,
            remove_columns=val_ds.column_names,
            fn_kwargs={"task_name": task_name},
        )
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
        val_formatted = val_ds.map(
            format_dapo_math_processed,
            remove_columns=val_ds.column_names,
            fn_kwargs={"task_name": task_name},
        )
    else:
        raise ValueError(
            "Unknown DAPOMath17KProcessed validation_source "
            f"{validation_source!r}; expected 'aime_2024' or 'train_holdout'"
        )

    train_ds = train_ds.shuffle(seed=seed)
    train_formatted = train_ds.map(
        format_dapo_math_processed,
        remove_columns=train_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


def format_math_processed_aime(
    data: dict[str, Any],
    task_name: str = "DAPOMath17KProcessed",
) -> dict[str, list[Any] | str]:
    """Format HuggingFaceH4/aime_2024 rows to match the processed DAPO schema."""
    return {
        "messages": [
            {"role": "user", "content": data["problem"]},
            {"role": "assistant", "content": data["answer"]},
        ],
        "task_name": task_name,
    }


class DAPOMath17KProcessedDataset(RawDataset):
    def __init__(
        self,
        seed: int = 42,
        config_name: str = "en",
        validation_source: str = "train_holdout",
        validation_num_samples: int = 500,
        validation_seed: int = 42,
    ) -> None:
        """Initialize open-r1/DAPO-Math-17k-Processed with a train/val split.

        Args:
            seed: Random seed for shuffling train
            config_name: Dataset config: "all", "cn", or "en"
            validation_source: "train_holdout" or "aime_2024"
            validation_num_samples: Holdout size when using "train_holdout"
            validation_seed: Random seed for the holdout partition
        """
        self.task_name = "DAPOMath17KProcessed"
        self.formatted_ds = prepare_dapo_math_processed_dataset(
            seed=seed,
            task_name=self.task_name,
            config_name=config_name,
            validation_source=validation_source,
            validation_num_samples=validation_num_samples,
            validation_seed=validation_seed,
        )
