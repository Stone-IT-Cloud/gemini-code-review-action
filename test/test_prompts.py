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
from src.prompts import get_review_prompt
from src.review_parser import REVIEW_SYSTEM_PROMPT


class TestGetReviewPrompt:
    def test_includes_system_prompt(self):
        prompt = get_review_prompt()
        assert REVIEW_SYSTEM_PROMPT in prompt

    def test_includes_extra_prompt(self):
        prompt = get_review_prompt(extra_prompt="Focus on security")
        assert "Focus on security" in prompt
        assert REVIEW_SYSTEM_PROMPT in prompt

    def test_empty_extra_prompt_not_included(self):
        prompt = get_review_prompt(extra_prompt="  ")
        # Should not have a blank extra section but still include system prompt
        assert REVIEW_SYSTEM_PROMPT in prompt
