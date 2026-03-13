#!/usr/bin/env bash
# Run all rmeshvk tests: Rust (cargo) + Python (autograd).
#
# Usage:
#   ./scripts/run_tests.sh          # run from rmeshvk/
#   ./scripts/run_tests.sh --python # Python tests only
#   ./scripts/run_tests.sh --cargo  # Cargo tests only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$(dirname "$REPO_DIR")/.venv/bin/python}"

run_cargo=true
run_python=true

if [[ "${1:-}" == "--python" ]]; then
    run_cargo=false
elif [[ "${1:-}" == "--cargo" ]]; then
    run_python=false
fi

failed=0

if $run_cargo; then
    echo "=== Cargo tests ==="
    if (cd "$REPO_DIR" && cargo test --workspace 2>&1); then
        echo "--- Cargo tests: PASS ---"
    else
        echo "--- Cargo tests: FAIL ---"
        failed=1
    fi
    echo
fi

if $run_python; then
    echo "=== Python tests ==="

    # Rebuild the .so if needed
    if ! "$PYTHON" -c "from rmesh_wgpu import RMeshRenderer" 2>/dev/null; then
        echo "Rebuilding rmesh-python .so ..."
        (cd "$REPO_DIR/crates/rmesh-python" && "$PYTHON" -m pip show maturin >/dev/null 2>&1 || pip install maturin)
        (cd "$REPO_DIR/crates/rmesh-python" && maturin develop 2>&1)
    fi

    if "$PYTHON" "$REPO_DIR/crates/rmesh-python/tests/test_autograd.py"; then
        echo "--- Python tests: PASS ---"
    else
        echo "--- Python tests: FAIL ---"
        failed=1
    fi
    echo
fi

if [[ $failed -eq 0 ]]; then
    echo "All tests passed."
else
    echo "Some tests failed."
    exit 1
fi
