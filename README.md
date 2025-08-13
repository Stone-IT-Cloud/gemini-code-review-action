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
- `github_repository` (required): Target repository (defaults to `${{ github.repository }}`).
- `github_pull_request_number` (required): PR number (defaults to `${{ github.event.pull_request.number }}`).
- `git_commit_hash` (required): PR head SHA (defaults to `${{ github.event.pull_request.head.sha }}`).
- `model` (required): Gemini model name (e.g., `gemini-1.5-pro-latest`).
- `extra_prompt` (optional): Additional system guidance to steer the review.
- `temperature` (optional): Sampling temperature.
- `top_p` (optional): Nucleus sampling.
- `pull_request_diff_file` (required): Path to the PR diff file to review.
- `pull_request_chunk_size` (optional): Diff chunk size to fit model limits.
- `log_level` (optional): Logging level (e.g., `INFO`, `DEBUG`).

## Features
- Splits large diffs and aggregates per-chunk analysis.
- Summarizes into a single review comment posted to the PR.
- Reads existing PR comments and reviews using PyGithub and injects them into the AI context so prior feedback is respected.
- Configurable prompt, model, and chunk size.

As you might know, a model of Gemini has limitation of the maximum number of input tokens.
So we have to split the diff of a pull request into multiple chunks, if the size of the diff is over the limitation.
We can tune the chunk size based on the model we use.

## Example usage
The workflow below writes the PR diff to a file and passes its path to the Action. The Action will also fetch PR comments automatically via the provided `github_token` and include them in the review context.

As a result, a single consolidated review is posted to the PR.
![An example comment of the code review](./docs/images/example.png)

```yaml
name: "Code Review by Gemini AI"

on:
  pull_request:
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
      - uses: Stone-IT-Cloud/gemini-code-review-action@1.0.3
        name: "Code Review by Gemini AI"
        id: review
        with:
          gemini_api_key: ${{ secrets.GEMINI_API_KEY }}
          github_token: ${{ secrets.GITHUB_TOKEN }}
          github_repository: ${{ github.repository }}
          github_pull_request_number: ${{ github.event.pull_request.number }}
          git_commit_hash: ${{ github.event.pull_request.head.sha }}
          model: gemini-2.5-flash
          pull_request_diff_file: ${{ env.PR_DIFF_PATH }}
          pull_request_chunk_size: "100000"
          extra_prompt: |
            Please review as a senior Python and security engineer. Be concise and actionable.
          log_level: INFO
```

## How it works
1. The workflow produces a unified diff file of the PR and provides it to the Action.
2. The Action loads the diff, splits it into chunks according to `pull_request_chunk_size`.
3. For each chunk, it sends:
   - A review instruction prompt, then the diff chunk,
   - Then a message containing the existing PR comments and reviews (via PyGithub),
   - Then asks Gemini to produce the final feedback for that chunk.
4. The Action summarizes the chunk-level feedback and posts a single review on the PR.

## Permissions and security
- Uses the default `GITHUB_TOKEN` to read PR metadata and post reviews.
- Requires `pull-requests: write` to create a PR review.
- Optional but recommended: `issues: read` to allow fetching general PR discussion comments.
- If `issues: read` or other permissions are unavailable, the Action will skip the unavailable comment types and proceed with the rest.
- Keep your `GEMINI_API_KEY` in repository secrets; never commit it.

## Local testing
Set the `LOCAL` environment variable to any value to prevent posting comments and log the review output instead.

## Notes
- This Action fetches PR comments using [PyGithub](https://github.com/PyGithub/PyGithub) and includes them in the model context to avoid redundant feedback and align with ongoing discussion.
