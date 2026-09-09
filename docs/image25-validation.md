# Offline validation — 2026-09-09

This change is an engineering integration, with no real native image calls or paid provider
calls. The 24-case model evaluation remains planned; no quality, latency or cost is claimed.

Targeted regression command:

```bash
python -B -m pytest -q -rs -p no:cacheprovider \
  tests/test_image25 tests/test_config.py tests/test_providers/test_registry.py \
  tests/test_features.py tests/test_agents/test_visualizer.py tests/test_utils.py \
  tests/test_cli.py tests/test_final_selection.py tests/test_pipeline/test_output_format.py \
  tests/test_pipeline/test_metadata_contract.py
```

- Clean feature worktree: **171 passed, 3 skipped** (Python 3.11, OpenAI SDK 2.30.0).
- Compatibility copy with the user's existing Gemini/Vertex source patch applied: **173 passed,
  3 skipped**. The patch applied cleanly; the original working directory and its diff digest
  remained unchanged. These existing uncommitted Gemini changes are not part of this commit.
- The three skips are optional `fastmcp` tests because that package is not installed.
- Ruff check/format passed for the new modules/tests/evaluation helper and rewritten OpenAI provider.
- The empty evaluation report stays pending, with 12 missing cases per candidate/split and null
  latency/cost. Mock evidence is excluded from real aggregates.

Coverage includes model aliases/snapshots, quality/size validation and precedence, real SDK
serialization over mocked HTTP, actual multipart reference/mask encoding, output decoding,
bounded retry, terminal/ambiguous errors, cancellation, output reuse, OpenAI Critic failure
handling, and preservation of Gemini provider/configuration/Polish/batch behavior.

A pre-existing legacy-size test expected GPT Image 1.x sizing while instantiating the default
GPT Image 2 provider. It now explicitly selects GPT Image 1.5; modern size contracts are tested
separately. This corrects the baseline test mismatch without changing the provider default.

No whole-suite or remote service acceptance is implied. Review real artifacts and usage evidence
under `evaluations/image25/README.md` before changing default model or quality settings.
