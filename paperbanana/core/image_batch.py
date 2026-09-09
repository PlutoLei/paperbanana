"""Local batch receipts: reuse verified outputs and retain ambiguous attempts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ImageBatch:
    def __init__(self, directory: Path):
        self.path = directory / "image-batch.json"
        self.records = {}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("version") != 1 or not isinstance(saved.get("items"), dict):
                raise ValueError("Invalid image batch receipt; preserve it for recovery")
            self.records = saved["items"]

    def decision(self, name: str, request_hash: str, output: Path, retry_unknown=False) -> str:
        previous = self.records.get(name, {})
        if previous.get("request_hash") != request_hash:
            return "run"
        if previous.get("status") in {"running", "result_unknown"} and not retry_unknown:
            return "result_unknown"
        if previous.get("status") == "complete" and output.is_file():
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            if digest == previous.get("output_hash"):
                try:
                    with Image.open(output) as image:
                        image.load()
                    return "reuse"
                except (OSError, ValueError):
                    pass
        return "run"

    def record(self, name: str, request_hash: str, status: str, output: Path | None = None):
        record = {"request_hash": request_hash, "status": status}
        if output is not None:
            record["output_hash"] = hashlib.sha256(output.read_bytes()).hexdigest()
        self.records[name] = record
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "items": self.records}, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)
