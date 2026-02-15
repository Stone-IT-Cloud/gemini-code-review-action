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
from typing import List, Optional


def chunk_string(input_string: str, chunk_size: int) -> List[str]:
    """Chunk a string into pieces of at most *chunk_size* characters."""
    chunked_inputs = []
    for i in range(0, len(input_string), chunk_size):
        chunked_inputs.append(input_string[i:i + chunk_size])
    return chunked_inputs


def _extract_model_text(response) -> Optional[str]:
    """Best-effort extraction of text from a Gemini SDK response."""
    if response is None:
        return None
    return getattr(response, "text", None)


def _safe_str(obj) -> str:
    """Best-effort stringify without raising."""
    try:
        return str(obj)
    except Exception:  # pylint: disable=broad-exception-caught
        return repr(obj)


def _get_usage_metadata(response) -> dict:
    """Extract usage metadata from a Gemini SDK response (best-effort)."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}

    # SDK may return an object with attributes; keep this defensive.
    prompt_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    # Some SDK versions may expose dict-like usage metadata.
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_token_count")
        output_tokens = usage.get("candidates_token_count")
        total_tokens = usage.get("total_token_count")

    out: dict = {}
    if prompt_tokens is not None:
        out["prompt_tokens"] = int(prompt_tokens)
    if output_tokens is not None:
        out["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        out["total_tokens"] = int(total_tokens)
    return out


def create_suggestion_fence(suggestion: str) -> str:
    """Create a suggestion code block with dynamic fence sizing.

    Ensures the fence cannot be terminated early by backticks in the
    suggestion content. Uses a fence size that exceeds the longest
    backtick run found in the suggestion.

    Args:
        suggestion: The code suggestion to wrap

    Returns:
        A formatted suggestion block with safe fence delimiters
    """
    import re

    # Find the longest sequence of backticks in the suggestion
    backtick_runs = re.findall(r'`+', suggestion)
    max_backticks = max((len(run) for run in backtick_runs), default=0)

    # Use at least 3 backticks, but more if needed to exceed max found
    fence_size = max(3, max_backticks + 1)
    fence = '`' * fence_size

    return f"\n{fence}suggestion\n{suggestion}\n{fence}"
