import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "image25_summary", ROOT / "evaluations/image25/summarize.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MANIFEST = json.loads((ROOT / "evaluations/image25/cases.json").read_text())


def test_fixed_suite_balances_categories_and_splits():
    assert len(MANIFEST["cases"]) == 24
    assert len({c["category"] for c in MANIFEST["cases"]}) == 6
    for category in {c["category"] for c in MANIFEST["cases"]}:
        assert [c["split"] for c in MANIFEST["cases"] if c["category"] == category].count(
            "holdout"
        ) == 2


def test_missing_and_mock_evidence_cannot_pass(tmp_path):
    row = {
        "candidate": "sunburst-high",
        "case": "structure-01",
        "evidence": "mock",
        "status": "success",
    }
    result = MODULE.summarize(MANIFEST, [row], tmp_path)
    assert result["excluded_mock"] == 1
    assert result["decision"] == "PENDING_HUMAN_REVIEW"
    for group in result["groups"]:
        assert group["missing"] == 12
        assert group["quality_passes"] == 0
        assert group["median_seconds"] is None
        assert group["total_cost_usd"] is None


def make_record(tmp_path):
    path = tmp_path / "synthetic-fixture.png"
    Image.new("RGB", (1536, 864)).save(path)
    return {
        "candidate": "sunburst-high",
        "case": "structure-01",
        "evidence": "real",
        "requested_model": MANIFEST["candidates"][0]["model"],
        "quality": "high",
        "status": "success",
        "artifact": path.name,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "review_status": "UNREVIEWED",
        "scores": None,
        "elapsed_seconds": 3.0,
        "cost_usd": None,
    }


def test_unreviewed_is_not_quality_pass_and_cost_is_unknown(tmp_path):
    result = MODULE.summarize(MANIFEST, [make_record(tmp_path)], tmp_path)
    group = result["groups"][0]
    assert group["generation_successes"] == 1
    assert group["quality_passes"] == 0
    assert group["success_rate_all_planned"] == 1 / 12
    assert group["median_seconds"] == 3
    assert group["observed_cost_usd"] is None


@pytest.mark.parametrize(
    "mutation", ["bad_hash", "bad_model", "nan_cost", "missing_scores", "duplicate"]
)
def test_invalid_evidence_rejected(tmp_path, mutation):
    row = make_record(tmp_path)
    records = [row]
    if mutation == "bad_hash":
        row["artifact_sha256"] = "0" * 64
    elif mutation == "bad_model":
        row["requested_model"] = "gpt-image-2"
    elif mutation == "nan_cost":
        row["cost_usd"] = float("nan")
    elif mutation == "missing_scores":
        row["review_status"] = "REVIEWED"
    else:
        records.append(row.copy())
    with pytest.raises(ValueError):
        MODULE.summarize(MANIFEST, records, tmp_path)


def test_complete_review_and_failure_share_full_denominator(tmp_path):
    row = make_record(tmp_path)
    row.update(
        review_status="REVIEWED",
        scores={name: 3 for name in MODULE.CRITERIA},
        reviewer="synthetic-reviewer",
        critical_errors=[],
        cost_usd=0.1,
    )
    failed = {
        **row,
        "case": "structure-02",
        "status": "failed",
        "review_status": "UNREVIEWED",
        "scores": None,
    }
    group = MODULE.summarize(MANIFEST, [row, failed], tmp_path)["groups"][0]
    assert group["quality_pass_rate_all_planned"] == 1 / 12
    assert group["failed"] == 1
    assert group["observed_cost_usd"] == 0.2
    assert group["total_cost_usd"] is None
