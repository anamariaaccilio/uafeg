"""Verify the installed canonical checkout against the local benchmark lock."""
import hashlib
import json
import subprocess
from pathlib import Path
import src.schemas.packet as canonical


def verify_benchmark(lock_path):
    lock = json.loads(Path(lock_path).read_text())
    repository = Path(canonical.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["git_commit"]:
        raise ValueError("Benchmark commit differs from lock")
    if canonical.SCHEMA_VERSION != lock["schema_version"]:
        raise ValueError("Benchmark schema version differs from lock")
    for relative, expected in lock["contract_files_sha256"].items():
        actual = hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Benchmark contract changed: {relative}")
    for name in ("manifest", "splits", "evidence_packets"):
        identifier, expected = lock[f"{name}_identifier"], lock[f"{name}_sha256"]
        if (identifier is None) != (expected is None):
            raise ValueError(f"Incomplete {name} lock")
        if identifier is not None:
            if hashlib.sha256((repository / identifier).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Benchmark {name} changed")
    return lock
