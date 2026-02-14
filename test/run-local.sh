#!/bin/bash
# Run the Gemini code review locally without Docker.
#
# Prerequisites:
#   pip install -r requirements.txt
#   export GEMINI_API_KEY="your-api-key"
#
# Usage:
#   bash test/run-local.sh [path-to-diff] [review-level]
#
# If no diff path is provided, the bundled test/long-diff.txt is used.
# If no review-level is provided, defaults to IMPORTANT.
#
# Examples:
#   bash test/run-local.sh                           # Use default diff and IMPORTANT level
#   bash test/run-local.sh /tmp/my.diff              # Use custom diff, IMPORTANT level
#   bash test/run-local.sh /tmp/my.diff CRITICAL     # Use custom diff and CRITICAL level
#   bash test/run-local.sh "" TRIVIAL                # Use default diff and TRIVIAL level

set -euo pipefail

DIFF_FILE="${1:-test/long-diff.txt}"
REVIEW_LEVEL="${2:-IMPORTANT}"

export LOCAL=1

python -m src.main \
    --diff-file="${DIFF_FILE}" \
    --model="gemini-2.5-flash" \
    --extra-prompt="Please write your review in English as an experienced software engineer." \
    --temperature=0.7 \
    --top-p=1 \
    --diff-chunk-size=2000000 \
    --review-level="${REVIEW_LEVEL}"
