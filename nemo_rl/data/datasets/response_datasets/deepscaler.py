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

from datasets import Dataset, concatenate_datasets, load_dataset

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
                "content": str(data["answer"]),
            },
        ],
        "task_name": task_name,
    }


# AIME year -> HF dataset with `problem` and `answer` columns (30 problems each).
AIME_VAL_SOURCES = {
    2024: "HuggingFaceH4/aime_2024",
    2025: "MathArena/aime_2025",
    2026: "MathArena/aime_2026",
}


def prepare_deepscaler_dataset(
    seed: int = 42,
    task_name: str = "DeepScaler",
    val_aime_years: list[int] | None = None,
    val_repeat: int = 16,
) -> dict[str, Dataset | None]:
    """Load and split the DeepScaler dataset into train and test sets.

    Args:
        seed: Shuffle seed for the training set.
        task_name: Task name of the training rows.
        val_aime_years: AIME years to validate on, each tagged with its own task
            name ``aime<year>`` so validation reports one accuracy per year. None
            keeps the original single AIME 2024 set under ``task_name``.
        val_repeat: Samples per validation problem (16 = the DeepScaleR setting).
    """
    # Load the original dataset for training
    train_ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(
        format_math,
        remove_columns=train_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )

    val_sets = (
        [(2024, task_name)]
        if val_aime_years is None
        else [(year, f"aime{year}") for year in val_aime_years]
    )
    val_parts = []
    for year, val_task_name in val_sets:
        if year not in AIME_VAL_SOURCES:
            raise ValueError(
                f"No AIME {year} validation source; known: {sorted(AIME_VAL_SOURCES)}"
            )
        val_ds = load_dataset(AIME_VAL_SOURCES[year], split="train")
        val_parts.append(
            val_ds.map(
                format_math,
                remove_columns=val_ds.column_names,
                fn_kwargs={"task_name": val_task_name},
            )
        )
    val_formatted = concatenate_datasets(val_parts)

    # Compute accuracy val_repeat times per sample (16 matches the DeepScaleR evaluation setting)
    val_repeated = []
    for _ in range(val_repeat):
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
        val_aime_years: list[int] | None = None,
        val_repeat: int = 16,
    ) -> None:
        """Initialize the DeepScaler dataset with train/test split.

        Args:
            seed: Random seed for reproducible splitting
            val_aime_years: AIME years to validate on; None = AIME 2024 only.
            val_repeat: Samples per validation problem.
        """
        self.task_name = "DeepScaler"
        self.formatted_ds = prepare_deepscaler_dataset(
            seed=seed,
            task_name=self.task_name,
            val_aime_years=val_aime_years,
            val_repeat=val_repeat,
        )
