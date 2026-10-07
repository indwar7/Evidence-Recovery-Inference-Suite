#!/bin/bash
# finish.sh -- stop the measurement workers and ship the challenge.
#
# Everything after the workers is fast (well under two minutes), so this can be
# fired at any checkpoint: whatever articles are fully measured become the
# dataset, and any partly-measured article is dropped rather than shipped with
# holes.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "== stopping workers"
pkill -f "gen_atlas.py --articles" || true
pkill -f "worker_cpu.py" || true
sleep 2

echo "== merging the two workers"
$PY dataset/generator/merge_atlas.py

echo "== exporting the three shipped files"
$PY dataset/generator/export_raw.py

echo "== building public/private splits"
$PY prepare.py --raw dataset/raw --out prepared

echo "== measuring the difficulty band"
$PY baselines.py

echo "== packaging to Desktop"
$PY package.py
