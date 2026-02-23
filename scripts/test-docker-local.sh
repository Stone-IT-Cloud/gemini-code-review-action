#!/usr/bin/env bash
# Test the code review action using this repo: build the image and run it
# with the last commit's diff. Requires GEMINI_API_KEY in the environment
# (e.g. set in your zshrc).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="${PWD}"
IMAGE_TAG="${IMAGE_TAG:-gemini-code-review-action:test}"

echo "==> Building image ${IMAGE_TAG}..."
docker build -t "${IMAGE_TAG}" .

echo "==> Creating diff from last commit (HEAD~1..HEAD)..."
git diff HEAD~1 HEAD > /tmp/pr.diff
echo "    $(wc -l < /tmp/pr.diff) lines in /tmp/pr.diff"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set. Export it in your shell and run again."
  exit 1
fi

echo "==> Running code review (LOCAL=1 so output is printed, no GitHub posting)..."
docker run --rm \
  -v /tmp:/tmp \
  -e GEMINI_API_KEY \
  -e LOCAL=1 \
  "${IMAGE_TAG}" \
  --diff-file=/tmp/pr.diff \
  --log-level=INFO \
  --review-level=TRIVIAL

echo "==> Done."
