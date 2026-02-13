#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
from typing import List, TypedDict


class AiReviewConfig(TypedDict):
    model: str
    diff: str
    extra_prompt: str
    prompt_chunk_size: int
    comments_text: str
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int


def check_required_env_vars():
    """Check required environment variables."""
    required_env_vars = [
        "GEMINI_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_PULL_REQUEST_NUMBER",
        "GIT_COMMIT_HASH",
    ]
    # if running locally, we only check gemini api key
    if os.getenv("LOCAL") is not None:
        required_env_vars = [
            "GEMINI_API_KEY",
        ]

    for required_env_var in required_env_vars:
        if os.getenv(required_env_var) is None:
            raise ValueError(f"{required_env_var} is not set")
