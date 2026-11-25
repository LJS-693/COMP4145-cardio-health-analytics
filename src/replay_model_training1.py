"""Replay one archived training script without overwriting model training1.

The original scripts and metadata are kept byte-for-byte. A temporary logical
project view resolves their original data, config and baseline-reference paths.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'model training1'
SCRIPTS = [
    'train_logistic.py', 'train_tree_models.py', 'train_svm.py',
    'train_seven_feature_models.py', 'tune_random_forest.py',
    'tune_random_forest_round2.py', 'tune_xgboost.py',
    'tune_xgboost_round2.py', 'tune_xgboost_multimetric.py',
]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_archive():
    manifest = json.loads((ROOT / 'reports' / ARCHIVE / 'archive_manifest.json').read_text())
    for record in [*manifest['files'], *manifest['snapshots']]:
        path = ROOT / record['archived']
        if not path.is_file() or checksum(path) != record['sha256']:
            raise RuntimeError(f'Archive file missing or changed: {path}')
    for record in manifest['data_references']:
        path = ROOT / record['path']
        if not path.is_file() or checksum(path) != record['sha256']:
            raise RuntimeError(f'Historical input data changed: {path}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify archive hashes and historical input data; no training.')
    parser.add_argument('--script', choices=SCRIPTS)
    parser.add_argument('--output-root', type=Path, help='New empty output directory; old artifacts are never overwritten.')
    args = parser.parse_args()
    if args.check:
        manifest = check_archive()
        print(f'Archive verified: {len(manifest["files"])} historical files and {len(manifest["snapshots"])} snapshots. No training performed.')
        return
    if not args.script or not args.output_root:
        parser.error('Replaying requires both --script and --output-root. Use --check for read-only verification.')
    output = args.output_root.expanduser().resolve()
    for protected in [ROOT / folder / ARCHIVE for folder in ['models', 'reports', 'src', 'outputs', 'config']]:
        if output == protected or protected in output.parents:
            parser.error('Output must be outside the archived model training1 folders.')
    for protected in [ROOT / 'data', ROOT / '.git']:
        if output == protected or protected in output.parents:
            parser.error('Output must not be inside data or .git.')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output directory must be new or empty.')
    check_archive()
    with tempfile.TemporaryDirectory(prefix='cardio-training1-view-') as temporary:
        view = Path(temporary)
        for name, target in {
            'data': ROOT / 'data',
            'config': ROOT / 'config' / ARCHIVE,
            'src': ROOT / 'src' / ARCHIVE,
            'reports': ROOT / 'reports' / ARCHIVE,
            'models': ROOT / 'models' / ARCHIVE,
            'outputs': ROOT / 'outputs' / ARCHIVE,
        }.items():
            (view / name).symlink_to(target, target_is_directory=True)
        command = [sys.executable, str(ROOT / 'src' / ARCHIVE / args.script),
                   '--project-root', str(view), '--output-root', str(output)]
        print(f'Replaying {args.script}. New outputs: {output}', flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    print('Replay complete. The historical archive was preserved.')


if __name__ == '__main__':
    main()
