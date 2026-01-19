# Harbor Task Validator - GitHub Action

Check if PRs in your repository can become [Harbor](https://github.com/laude-institute/harbor) tasks for LLM training and evaluation.

## Installation

**1. Run this command** in your repo:

```bash
curl -fL --create-dirs -o .github/workflows/harbor-check.yml https://raw.githubusercontent.com/abundant-ai/swe-gen/main/action/harbor-check.yml
```

**2. Add secrets to GitHubs** (`Settings` → `Secrets and variables` → `Actions`):
- `CLAUDE_CODE_OAUTH_TOKEN` (or `ANTHROPIC_API_KEY`)
- `OPENAI_API_KEY`

**3. Commit and push:**

```bash
git add .github/workflows/harbor-check.yml && git commit -m "Added Harbor task check" && git push
```

## What Makes a PR Eligible?

| Requirement | Why |
|-------------|-----|
| Substantial changes | Not just docs, formatting, or version bumps |
| Includes test changes | Tests validate the fix works |
| 3-10 source files modified | Multi-component fixes make better tasks |

Most PRs won't be eligible—and that's fine!

## Outputs

| Output | Description |
|--------|-------------|
| `eligible` | `true` or `false` |
| `reason` | Why the PR is/isn't eligible |
| `task_id` | Task ID like `owner__repo-123` |

## What Happens Next?

When a PR passes validation:
- ✅ Job Summary shows validation results
- 📦 Task artifact is uploaded to the workflow run
- 📤 **One-click submission** to [Task Bank](https://github.com/abundant-ai/task-bank) for review

### Submitting to Task Bank

Click the **"Submit to Task Bank"** button in the Job Summary to contribute your task. This opens a pre-filled issue that triggers automatic import. Maintainers review submissions before adding them to the training set.
