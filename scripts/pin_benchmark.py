"""Pin actual committed canonical artifacts; never fabricate a version or hash."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import src.schemas.packet as canonical

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_lock():
    repository = Path(canonical.__file__).resolve().parents[2]
    status = subprocess.check_output(['git','-C',str(repository),'status','--porcelain'],text=True)
    if status:
        raise ValueError('Canonical repository must be committed and clean before pinning')
    artifact_dir = Path('data/manifests/benchmark_v1')
    metadata_path = repository/artifact_dir/'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['schema_version'] != canonical.SCHEMA_VERSION:
        raise ValueError('Artifact schema differs from installed canonical contract')
    old = json.loads((ROOT/'benchmark.lock.json').read_text())
    contract_paths = set(old['contract_files_sha256']) | {
        'configs/benchmark_v1.yaml', 'configs/benchmark_conditions.yaml',
        'src/data/duplicate_audit.py', 'src/data/claim_benchmark.py',
        str(artifact_dir/'metadata.json'),
    }
    for name, expected in metadata['artifacts_sha256'].items():
        relative = artifact_dir/name
        if digest(repository/relative) != expected:
            raise ValueError(f'Frozen artifact disagrees with metadata: {name}')
        subprocess.run(['git','-C',str(repository),'ls-files','--error-unmatch',str(relative)],
                       check=True, stdout=subprocess.DEVNULL)
        contract_paths.add(str(relative))
    lock = {
        'benchmark_repository': 'vehicle-claim-verification-baselines',
        'benchmark_version': metadata['benchmark_version'],
        'git_commit': subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip(),
        'schema_version': canonical.SCHEMA_VERSION,
        'availability': {'manifest':'AVAILABLE','splits':'AVAILABLE','frozen_evidence_packets':'AVAILABLE_WITH_SPECIALISTS_NULL'},
        'contract_files_sha256': {name:digest(repository/name) for name in sorted(contract_paths)},
    }
    for key, name in [('manifest','manifest.jsonl'),('splits','splits.json'),('evidence_packets','evidence_packets.jsonl')]:
        relative = artifact_dir/name
        lock[key+'_identifier'] = str(relative)
        lock[key+'_sha256'] = digest(repository/relative)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write',action='store_true')
    group.add_argument('--check',action='store_true')
    args = parser.parse_args()
    actual = generate_lock()
    target = ROOT/'benchmark.lock.json'
    if args.check:
        if actual != json.loads(target.read_text()):
            raise ValueError('Lock differs from actual canonical committed artifacts')
        print('PASS: lock matches actual canonical artifacts')
    else:
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(actual,indent=2)+'\n')
        temporary.replace(target)
        print('Pinned canonical commit', actual['git_commit'])


if __name__ == '__main__':
    main()
