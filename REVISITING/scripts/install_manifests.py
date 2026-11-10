#!/usr/bin/env python3
"""Materialize the exact published cohorts using locally prepared annotations."""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.data import load_metadata
from revision.paths import metadata_dir, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['bdd100k', 'nuimages'], required=True)
    parser.add_argument('--metadata-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=output_dir())
    args = parser.parse_args()
    filename = f'{args.dataset}_n1000_seed20261029.json'
    reference = json.loads((ROOT / 'data' / 'manifests' / filename).read_text())
    rows, _ = load_metadata(args.dataset, str(args.metadata_dir or metadata_dir(args.dataset)), 'test_fixed')
    lookup = {row['image_id']: row for row in rows}
    missing = [key for key in reference['image_ids'] if key not in lookup]
    if missing:
        raise SystemExit(f'{len(missing)} reference images missing from fixed-test metadata. First IDs {missing[:5]}')
    selected = [copy.deepcopy(lookup[key]) for key in reference['image_ids']]
    if len(selected) != 1000 or len({row['image_id'] for row in selected}) != 1000:
        raise SystemExit('Reference cohort must contain exactly 1,000 unique images.')
    reference.pop('format', None)
    reference['rows'] = selected
    reference['created_by'] = 'scripts/install_manifests.py from released evaluation image IDs'
    destination = args.output_dir.resolve() / 'manifests' / filename
    payload = json.dumps(reference, indent=2, sort_keys=True) + '\n'
    if destination.exists() and json.loads(destination.read_text()) != reference:
        raise SystemExit(f'Different manifest exists at {destination}. Use a separate output directory.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload)
    print(f'Installed fixed 1,000-image cohort at {destination}')


if __name__ == '__main__':
    main()
