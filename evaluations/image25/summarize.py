"""Summarize recorded evidence only. Never dispatches generation or fills missing scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from PIL import Image

CRITERIA = ("instruction", "text", "layout", "semantic")


def summarize(manifest: dict, records: list[dict], artifact_root: Path) -> dict:
    cases = {case["id"]: case for case in manifest["cases"]}
    candidates = {candidate["id"]: candidate for candidate in manifest["candidates"]}
    if len(cases) != len(manifest["cases"]) or len(candidates) != len(manifest["candidates"]):
        raise ValueError("Duplicate manifest IDs")
    seen = set()
    verified = []
    excluded = 0
    for row in records:
        key = (row["candidate"], row["case"])
        if key in seen or key[0] not in candidates or key[1] not in cases:
            raise ValueError("Duplicate or unknown evaluation record")
        seen.add(key)
        if row.get("evidence") not in {"real", "mock"}:
            raise ValueError("Evidence must be explicitly real or mock")
        if row.get("status") not in {"success", "failed", "result_unknown"}:
            raise ValueError("Invalid execution status")
        if row.get("evidence") == "mock":
            excluded += 1
            continue
        candidate, case = candidates[key[0]], cases[key[1]]
        for field in ("requested_model", "quality"):
            expected = candidate["model" if field == "requested_model" else field]
            if row.get(field) != expected:
                raise ValueError(f"Mismatched candidate {field}")
        for field in ("elapsed_seconds", "cost_usd"):
            value = row.get(field)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"Invalid {field}")
        if row["status"] == "success":
            path = (artifact_root / row["artifact"]).resolve()
            if not path.is_relative_to(artifact_root.resolve()):
                raise ValueError("Artifact must stay within the artifact directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("artifact_sha256"):
                raise ValueError("Artifact hash mismatch")
            with Image.open(path) as image:
                image.load()
                if f"{image.width}x{image.height}" != case["size"]:
                    raise ValueError("Artifact size mismatch")
                if case.get("background") == "transparent" and "A" not in image.getbands():
                    raise ValueError("Missing alpha channel")
        scores = row.get("scores")
        required = {*CRITERIA, *(["reference_preservation"] if case["operation"] == "edit" else [])}
        if row.get("review_status") == "REVIEWED":
            if (
                row["status"] != "success"
                or not isinstance(scores, dict)
                or set(scores) != required
                or any(type(value) is not int or not 0 <= value <= 4 for value in scores.values())
                or not isinstance(row.get("critical_errors"), list)
                or not row.get("reviewer")
            ):
                raise ValueError(
                    "Reviewed evidence requires successful output and a complete rubric"
                )
        elif row.get("review_status") != "UNREVIEWED" or scores is not None:
            raise ValueError("Missing review must stay UNREVIEWED with null scores")
        verified.append(row)
    groups = []
    for candidate in candidates:
        for split in ("tuning", "holdout"):
            expected = [case for case in cases.values() if case["split"] == split]
            observed = [
                row
                for row in verified
                if row["candidate"] == candidate and cases[row["case"]]["split"] == split
            ]
            successful = [row for row in observed if row["status"] == "success"]
            reviewed = [row for row in observed if row["review_status"] == "REVIEWED"]
            passed = [
                row
                for row in reviewed
                if not row["critical_errors"] and min(row["scores"].values()) >= 3
            ]
            latencies = [
                row["elapsed_seconds"] for row in observed if row.get("elapsed_seconds") is not None
            ]
            costs = [row["cost_usd"] for row in observed if row.get("cost_usd") is not None]
            groups.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "planned": len(expected),
                    "observed_real": len(observed),
                    "missing": len(expected) - len(observed),
                    "generation_successes": len(successful),
                    "reviewed": len(reviewed),
                    "quality_passes": len(passed),
                    "failed": sum(row["status"] == "failed" for row in observed),
                    "result_unknown": sum(row["status"] == "result_unknown" for row in observed),
                    "success_rate_all_planned": len(successful) / len(expected)
                    if expected
                    else None,
                    "quality_pass_rate_all_planned": len(passed) / len(expected)
                    if expected
                    else None,
                    "latency_samples": len(latencies),
                    "median_seconds": median(latencies) if latencies else None,
                    "cost_samples": len(costs),
                    "observed_cost_usd": sum(costs) if costs else None,
                    "total_cost_usd": sum(costs)
                    if len(costs) == len(expected) and expected
                    else None,
                }
            )
    return {
        "schema_version": 1,
        "excluded_mock": excluded,
        "groups": groups,
        "decision": "PENDING_HUMAN_REVIEW",
        "dispatches_api_requests": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.manifest.read_text()),
        json.loads(args.records.read_text()),
        args.artifact_root,
    )
    print(json.dumps(result, indent=2))
