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
from src.review_parser import REVIEW_SYSTEM_PROMPT


def get_review_prompt(extra_prompt: str = "") -> str:
    """Get a prompt template for structured JSON code review."""
    parts = [REVIEW_SYSTEM_PROMPT]
    if extra_prompt.strip():
        parts.append(extra_prompt.strip())
    parts.append(
        "This is a pull request or part of a pull request if the pull request is very large.\n"
        "Suppose you review this PR as an excellent software engineer and an excellent security engineer.\n"
        "Analyze the differences and provide review comments for any major issues found.\n"
        "In the next messages I will be sending you the difference between the GitHub file codes, okay?"
    )
    return "\n\n".join(parts)


def get_summarize_prompt() -> str:
    """Get a prompt template for summarizing chunked reviews."""
    template = """
    Can you summarize this for me?
    It would be good to stick to highlighting pressing issues and providing code suggestions to improve the pull request.
    Here's what you need to summarize:
    """
    return template
