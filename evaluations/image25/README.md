# Image 2.5 evaluation protocol

Status: **planned, no live calls or measured model results**. This folder does not launch API
requests. It supplies 24 fixed cases, a scoring contract and an offline report generator.

## Freeze before running

`cases.json` has six categories × four cases: bilingual text, structure, data fidelity, masked
edit, multiple references and transparent assets. Each category has two tuning and two holdout
cases. All numbers in prompts are illustrative, not experimental or clinical findings.

Four candidate profiles compare Sunburst/Flare at high/xhigh quality using dated requested model
IDs. The full matrix would be 96 image requests plus failures/retries. This is a proposed matrix,
not spending authorization. Start with the authorized subset, record every attempted case and
retain the full planned denominator. Do not change prompts after looking at holdout outputs.

Before paid evaluation, prepare the specified edit references/masks, verify mask dimensions and
alpha, and freeze their files and SHA-256 hashes in an input manifest. Freeze the case manifest,
source commit, SDK version and candidate settings as well. These reference assets have not been
created by the offline engineering work. Preserve original inputs and output artifacts locally;
never invent input hashes or fill missing outputs with stand-ins in real records.

## Optimization and acceptance

1. On the tuning split, compare instruction adherence and semantic errors first; then compare
   measured latency and actual cost among candidates that meet the quality floor. Sunburst's
   editing emphasis and Flare's speed emphasis are hypotheses from product positioning.
2. Use the lowest quality/cost setting that passes the task's rubric. Test xhigh only as a named
   candidate; max is supported but is not the default. Keep size and prompts fixed when comparing.
3. Freeze the chosen profile and rationale before inspecting holdout scores. Review outputs under
   anonymized candidate labels; record reviewer identity separately from the generator.
4. Score each criterion 0–4: 0 unusable, 1 major defects, 2 needs material edits, 3 usable with minor
   issues, 4 meets the prompt. Criteria: instruction, exact text/numbers, layout, semantics; edits
   additionally require reference preservation. Pass requires every applicable score >=3 and no
   critical error (invented data, altered meaning, wrong label/arrow, or unauthorized region edits).
5. Review all six holdout categories. Keep failed, missing, unknown and unreviewed cases visible;
   never exclude them to increase the reported pass rate. Repeat runs are separate experiments,
   not replacements for failures. Default/provider changes require a reviewed real report.

Latency includes retry/backoff, with sample count. Costs require billing/usage evidence; do not
estimate an absent cost as zero. A median on partial observations is not an end-to-end benchmark.
No latency/cost budget has been supplied, so this protocol does not invent a budget pass/fail gate.

## Recording and summarizing

Records are a JSON array, one row per candidate/case. `records.example.json` is deliberately empty.
Each real row includes `candidate`, `case`, `evidence: "real"`, `requested_model`, `quality`,
`status` (`success`, `failed`, `result_unknown`), `review_status` (`UNREVIEWED`, `REVIEWED`),
`elapsed_seconds` and `cost_usd` (null if unknown). Successful rows also include a relative
`artifact` path and its `artifact_sha256`. Keep requested versus reported model separately in
raw request metadata; reported model is null if absent.

Unreviewed rows have `scores: null`. Reviewed rows need `reviewer`, `critical_errors` (an array,
empty when none) and integer scores under `instruction`, `text`, `layout`, `semantic`, plus
`reference_preservation` for edits. Bind edit rows to the separately frozen input manifest.
Use `evidence: "mock"` only for synthetic software tests; those rows never enter real aggregates.

```bash
python evaluations/image25/summarize.py --records evaluations/image25/records.example.json \
  --artifact-root ./evaluation-artifacts
```

The report checks candidate identity, artifact hashes/decodability/dimensions, review completeness,
invalid numeric values and duplicate records. It reports tuning/holdout separately with all planned
cases in denominators. It is a consistency checker of supplied records, not proof that a claimed
real API call happened; preserve request/usage receipts for independent review. Its decision stays
`PENDING_HUMAN_REVIEW`. Missing latency/cost remains null. An empty report is not a passed evaluation.
