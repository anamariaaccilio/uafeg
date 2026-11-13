"""Verify the installed canonical checkout against the local benchmark lock.

The lock is enforced ONLY through SHA-256 hashes of the contract files and the
frozen artifacts. ``git_commit`` is informative provenance: committing an
unrelated change (docs, new code) in the canonical repository must not break
the contract, while any byte change in a locked file must.
"""
import hashlib
import json
import subprocess
import warnings
from pathlib import Path
import src.schemas.packet as canonical


def canonical_repository() -> Path:
    return Path(canonical.__file__).resolve().parents[2]


def current_commit(repository: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_benchmark(lock_path, repository: Path | None = None):
    lock = json.loads(Path(lock_path).read_text())
    repository = Path(repository) if repository is not None else canonical_repository()
    if canonical.SCHEMA_VERSION != lock["schema_version"]:
        raise ValueError("Benchmark schema version differs from lock")
    for relative, expected in lock["contract_files_sha256"].items():
        path = repository / relative
        if not path.exists():
            raise ValueError(f"Benchmark contract file missing: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"Benchmark contract changed: {relative}")
    for name in ("manifest", "splits", "evidence_packets"):
        identifier, expected = lock.get(f"{name}_identifier"), lock.get(f"{name}_sha256")
        if (identifier is None) != (expected is None):
            raise ValueError(f"Incomplete {name} lock")
        if identifier is not None and _sha256(repository / identifier) != expected:
            raise ValueError(f"Benchmark {name} changed")
    commit = current_commit(repository)
    if lock.get("git_commit") and commit and commit != lock["git_commit"]:
        warnings.warn(f"Informative only: canonical HEAD {commit[:12]} differs from pinned "
                      f"{lock['git_commit'][:12]}; hashes match, so the contract holds.", stacklevel=2)
    return lock
