#!/usr/bin/env bash
# Quick progress report: how many tasks succeeded / failed / are still running.
#
# Usage: bash hpc/status.sh [job_id]
#   If job_id is given, also shows live squeue output.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/hpc/logs"

if [[ $# -gt 0 ]]; then
    echo "=== squeue ==="
    squeue -j "$1" -o "%.8i %.4P %.12j %.8u %.2t %.10M %.6D %R" 2>/dev/null || true
    echo ""
fi

echo "=== Log summary (from hpc/logs/) ==="
TOTAL=0
DONE=0
FAIL=0

for f in "$LOG_DIR"/agrixel_*_*.out; do
    [[ -f "$f" ]] || continue
    (( TOTAL++ )) || true
    if grep -q "DONE:" "$f" 2>/dev/null; then
        (( DONE++ )) || true
    fi
    if grep -q "exit code [^0]" "$f" 2>/dev/null; then
        (( FAIL++ )) || true
    fi
done

RUNNING=$(( TOTAL - DONE ))
echo "  Total log files : $TOTAL"
echo "  Finished        : $DONE"
echo "  Failed          : $FAIL"
echo "  Still running   : $RUNNING"

echo ""
echo "=== Failed tasks ==="
for f in "$LOG_DIR"/agrixel_*_*.out; do
    [[ -f "$f" ]] || continue
    if grep -q "exit code [^0]" "$f" 2>/dev/null; then
        echo "  $f"
        grep "PARCEL\|exit code" "$f" | head -5
    fi
done
