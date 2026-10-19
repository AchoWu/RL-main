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
"""JustRL-II's math grader (miles ``--rm-type math``), vendored from verl06/recipe/justrl2.

Last ``\boxed{}`` graded 0/1 by mathd normalisation + sympy, with a math-verify fallback that
can only flip 0 -> 1. Selected with ``env.math.math_verify_impl=justrl2_math`` so validation
accuracy is scored exactly as JustRL-II scores its AIME sets.
"""

from nemo_rl.environments.justrl2_math.math_utils import (
    extract_answer,
    grade_answer_union,
)

__all__ = ["extract_answer", "grade_answer_union"]
