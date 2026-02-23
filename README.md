# gemini-code-review-action
GitHub Action that uses Google Gemini to automatically review Pull Requests.

If the diff exceeds the model's context window, the Action splits it into chunks and requests feedback per chunk, then produces a final summary and posts a single PR review. The Action also reads existing PR discussion (general comments, inline review comments, and past review summaries) and feeds them to Gemini as context so the review considers prior feedback.

## Pre-requisites
- Set a repository secret `GEMINI_API_KEY` with your Gemini API key.
  - Get an API key: [GET MY API KEY](https://makersuite.google.com/app/apikey)
- The workflow should grant `contents: read` and `pull-requests: write` permissions (see example below).

## Inputs

- `gemini_api_key` (required): Gemini API key (secret recommended).
- `github_token` (required): GitHub token. Use the default `${{ secrets.GITHUB_TOKEN }}`.
- `dockerhub_username` (optional): Docker Hub username to authenticate pulls during the action image build (helps avoid Docker Hub rate limits).
- `dockerhub_token` (optional): Docker Hub access token/password (secret recommended).
- `github_repository` (required): Target repository (defaults to `${{ github.repository }}`).
- `github_pull_request_number` (required): PR number (defaults to `${{ github.event.pull_request.number }}`).
- `git_commit_hash` (required): PR head SHA (defaults to `${{ github.event.pull_request.head.sha }}`).
- `model` (required): Gemini model name (e.g., `gemini-2.5-flash`, `gemini-2.5-pro`).
- `extra_prompt` (optional): Additional system guidance to steer the review.
- `temperature` (optional): Sampling temperature.
- `top_p` (optional): Nucleus sampling.
- `pull_request_diff_file` (required): Path to the PR diff file to review.
- `pull_request_chunk_size` (optional): Diff chunk size to fit model limits.
- `log_level` (optional): Logging level (e.g., `INFO`, `DEBUG`).
- `review_level` (optional): Minimum severity level to post comments (defaults to `IMPORTANT`).
  - `TRIVIAL`: Include all comments (style issues, formatting, minor refactoring, etc.)
  - `IMPORTANT`: Include important and critical issues (logic errors, bugs, performance issues)
  - `CRITICAL`: Only include critical issues (security vulnerabilities, crashes, data loss risks)

### Environment Variables for Quota Management

The following optional environment variables can be set to configure quota tracking and fail-fast behavior:

- `GEMINI_FAIL_FAST_ON_NO_QUOTA` (default: `1`): When set to `1`, the action will immediately fail when daily quota exhaustion is detected, rather than continuing to retry. Set to `0` to disable fail-fast.
- `GEMINI_QUOTA_RPM` (optional): Your Gemini API requests-per-minute quota limit. When provided, the action logs estimated remaining RPM after each request.
- `GEMINI_QUOTA_TPM` (optional): Your Gemini API tokens-per-minute quota limit. When provided, the action logs estimated remaining TPM after each request.
- `GEMINI_QUOTA_RPD` (optional): Your Gemini API requests-per-day quota limit. When provided, the action logs estimated remaining RPD after each request.

**Example with quota tracking:**
```yaml
env:
  GEMINI_FAIL_FAST_ON_NO_QUOTA: "1"
  GEMINI_QUOTA_RPM: "60"
  GEMINI_QUOTA_TPM: "32000"
  GEMINI_QUOTA_RPD: "1500"
```

## Features
- Splits large diffs and aggregates per-chunk analysis.
- Summarizes into a single review comment posted to the PR.
- Reads existing PR comments and reviews using PyGithub and injects them into the AI context so prior feedback is respected.
- Configurable prompt, model, and chunk size.
- **Severity-based filtering** to reduce noise from trivial suggestions (see [Severity Filtering](#severity-filtering)).

As you might know, a model of Gemini has limitation of the maximum number of input tokens.
So we have to split the diff of a pull request into multiple chunks, if the size of the diff is over the limitation.
We can tune the chunk size based on the model we use.

## Severity Filtering

To reduce review noise, the action classifies comments by severity and allows you to filter which ones get posted. This is useful for focusing on important issues while avoiding trivial style suggestions.

### Severity Levels

- **TRIVIAL**: Style issues, formatting, minor refactoring, missing docstrings
- **IMPORTANT**: Logic errors, potential bugs, performance inefficiencies (e.g., O(n²)), bad practices
- **CRITICAL**: Security vulnerabilities (SQLi, XSS), potential crashes, breaking changes, data loss risks

### Configuration

**In GitHub Actions workflows:**

```yaml
- uses: Stone-IT-Cloud/gemini-code-review-action@1.1.4
  with:
    review_level: IMPORTANT  # Default: filters out TRIVIAL comments
    # Other parameters...
```

**When running locally:**

```bash
# Option 1: Use CLI parameter
python -m src.main \
    --diff-file=/tmp/my-changes.diff \
    --review-level=CRITICAL

# Option 2: Use environment variable
export REVIEW_LEVEL=CRITICAL
python -m src.main --diff-file=/tmp/my-changes.diff
```

**Behavior:**
- `review_level: TRIVIAL` → Posts all comments (including style suggestions)
- `review_level: IMPORTANT` (default) → Posts IMPORTANT + CRITICAL only
- `review_level: CRITICAL` → Posts only CRITICAL security/stability issues

Comments that don't meet the threshold are logged but not posted to the PR.

## Example usage
The workflow below writes the PR diff to a file and passes its path to the Action. The Action will also fetch PR comments automatically via the provided `github_token` and include them in the review context.

As a result, a single consolidated review is posted to the PR.
![An example comment of the code review](./docs/images/example.png)

```yaml
name: "Code Review by Gemini AI"

on:
  pull_request:

# This configuration limits concurrency per pull request, allowing different PRs in the same repository to run in parallel.
concurrency:
  group: gemini-review-${{ github.repository }}-${{ github.event.pull_request.number }}
  cancel-in-progress: true
env:
  GEMINI_MODEL: "gemini-2.5-flash"

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    env:
      PR_DIFF_PATH: pull-request.diff
      # Optional: Configure quota tracking and fail-fast behavior
      GEMINI_FAIL_FAST_ON_NO_QUOTA: "1"
      GEMINI_QUOTA_RPM: "60"
      GEMINI_QUOTA_TPM: "32000"
      GEMINI_QUOTA_RPD: "1500"
    steps:
      - uses: actions/checkout@v4
      - name: "Get diff of the pull request"
        id: get_diff
        shell: bash
        env:
          PULL_REQUEST_HEAD_REF: "${{ github.event.pull_request.head.ref }}"
          PULL_REQUEST_BASE_REF: "${{ github.event.pull_request.base.ref }}"
        run: |-
          git fetch origin "${{ env.PULL_REQUEST_HEAD_REF }}"
          git fetch origin "${{ env.PULL_REQUEST_BASE_REF }}"
          git checkout "${{ env.PULL_REQUEST_HEAD_REF }}"
          git diff "origin/${{ env.PULL_REQUEST_BASE_REF }}" > "${{ env.PR_DIFF_PATH }}"
      - uses: Stone-IT-Cloud/gemini-code-review-action@1.1.4
        name: "Code Review by Gemini AI"
        id: review
        with:
          gemini_api_key: ${{ secrets.GEMINI_API_KEY }}
          github_token: ${{ secrets.GITHUB_TOKEN }}
          # Optional: authenticate Docker Hub pulls to avoid rate limits
          # dockerhub_username: ${{ secrets.DOCKERHUB_USERNAME }}
          # dockerhub_token: ${{ secrets.DOCKERHUB_TOKEN }}
          github_repository: ${{ github.repository }}
          github_pull_request_number: ${{ github.event.pull_request.number }}
          git_commit_hash: ${{ github.event.pull_request.head.sha }}
          model: gemini-2.5-flash
          pull_request_diff_file: ${{ env.PR_DIFF_PATH }}
          pull_request_chunk_size: "100000"
          extra_prompt: |
            Please review as a senior Python and security engineer. Be concise and actionable.
          log_level: INFO
          # Optional: Set minimum severity level for comments (TRIVIAL, IMPORTANT, CRITICAL)
          # Default is IMPORTANT (filters out trivial style suggestions)
          # review_level: CRITICAL
```

## How it works
1. The workflow produces a unified diff file of the PR and provides it to the Action.
2. The Action loads the diff, splits it into chunks according to `pull_request_chunk_size`.
3. For each chunk, it makes a single Gemini request containing:
   - The review instruction prompt,
   - The diff chunk,
   - Existing PR comments (when available) as additional context.
4. If there are multiple chunks, the Action automatically summarizes the chunk-level feedback and posts a single review on the PR.

## Avoiding Gemini rate limits (recommended)
Gemini quotas are shared across your project/account. If multiple workflows run in parallel using the same `GEMINI_API_KEY`, they can compete for the same quota.

- Use workflow `concurrency` (example above) to serialize runs per pull request (avoid overlapping runs for the same PR).
- If you still hit rate limits, reduce `pull_request_chunk_size` and/or avoid running reviews for every PR update (e.g., only when specific labels are added using an `if:` condition such as `if: contains(github.event.pull_request.labels.*.name, 'needs-review')`, or by triggering the workflow manually via `workflow_dispatch`).

## Permissions and security
- Uses the default `GITHUB_TOKEN` to read PR metadata and post reviews.
- Requires `pull-requests: write` to create a PR review.
- Keep your `GEMINI_API_KEY` in repository secrets; never commit it.

## Local testing
Set the `LOCAL` environment variable to any value to prevent posting comments and log the review output instead.

### Running as a Pre-Commit Hook (Recommended)

You can use this action as a [pre-commit](https://pre-commit.com/) hook to get AI-powered code reviews **before** you commit. This enables "Shift Left" development practices by catching issues early in your development workflow.

**Prerequisites:**
- Install pre-commit: `pip install pre-commit`
- Set your Gemini API key in your shell environment

**1. Add to your `.pre-commit-config.yaml`:**

```yaml
repos:
  - repo: https://github.com/Stone-IT-Cloud/gemini-code-review-action
    rev: v1.1.4  # Use the latest release tag
    hooks:
      - id: gemini-code-review
        # Optional: customize the review level
        # args: ['--review-level=CRITICAL']
```

**2. Set your API key:**

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

**3. Install the hook:**

```bash
pre-commit install
```

**4. Use it:**

Now when you run `git commit`, the hook will automatically:
- Analyze your staged changes using Gemini AI
- Display findings in a colorized, human-readable format
- Block the commit if **CRITICAL** issues are found
- Allow the commit if only IMPORTANT or TRIVIAL issues are found

**Example output:**

```
================================================================================
🤖 Gemini AI Code Review
================================================================================

Found 2 issue(s):
  🔴 1 CRITICAL (blocking)
  🟡 1 IMPORTANT

🔴 Issue #1  CRITICAL
   📄 src/auth.py:45
   💬 Comment:
      SQL injection vulnerability detected. User input is directly
      concatenated into SQL query without sanitization.
   💡 Suggested Fix:
   ──────────────────────────────────────────────────────────────────────────
   -  query = f"SELECT * FROM users WHERE id = {user_id}"
   +  query = "SELECT * FROM users WHERE id = ?"
   +  cursor.execute(query, (user_id,))
   ──────────────────────────────────────────────────────────────────────────
```

**Customizing the Review Level:**

By default, only CRITICAL issues block commits. You can customize this in your `.pre-commit-config.yaml`:

```yaml
- id: gemini-code-review
  args: ['--review-level=IMPORTANT']  # Show IMPORTANT and CRITICAL (only CRITICAL blocks)
```

**Bypassing the hook (when needed):**

```bash
git commit --no-verify  # Skip all pre-commit hooks
```

### Running locally without Docker

You can run the review tool directly from your machine using Python. This is useful for testing diffs before pushing to GitHub.

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Set environment variables:**

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export LOCAL=1
```

**3. Generate a diff and run the review:**

```bash
# Generate a diff from your branch
git diff main > /tmp/my-changes.diff

# Run the review
python -m src.main \
    --diff-file=/tmp/my-changes.diff \
    --model=gemini-2.5-flash \
    --extra-prompt="Review as a senior Python engineer." \
    --temperature=0.7 \
    --top-p=1 \
    --diff-chunk-size=2000000 \
    --review-level=IMPORTANT
```

**Review Level Options:**
- `--review-level=TRIVIAL`: Show all comments including style suggestions
- `--review-level=IMPORTANT` (default): Show important and critical issues only
- `--review-level=CRITICAL`: Show only critical security and stability issues

Or use the bundled helper script:

```bash
bash test/run-local.sh /tmp/my-changes.diff
```

You can also set the review level via environment variable:

```bash
export REVIEW_LEVEL=CRITICAL
bash test/run-local.sh /tmp/my-changes.diff
```

## Project structure

The source code follows the **Single Responsibility Principle (SRP)**, with each module handling one concern:

```
src/
├── main.py              # CLI entry point and orchestration
├── config.py            # Configuration and environment validation
├── gemini_client.py     # Gemini AI API interactions
├── github_client.py     # GitHub API interactions (comments, PR data)
├── prompts.py           # Prompt templates for the AI model
├── quota.py             # Rate limiting and quota tracking
├── review_formatter.py  # Formatting review output for GitHub
├── review_parser.py     # Parsing structured JSON from AI responses
└── utils.py             # General-purpose utilities
```

## Notes
- This Action fetches PR comments using [PyGithub](https://github.com/PyGithub/PyGithub) and includes them in the model context to avoid redundant feedback and align with ongoing discussion.
