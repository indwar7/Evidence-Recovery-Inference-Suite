#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install the rewritten engine over the shipped one.
cp "${SCRIPT_DIR}/store.py" /app/src/store.py

echo "installed rewritten /app/src/store.py"
