#!/usr/bin/env bash
# Step 1: setup environment for ClinFuseDiff

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==========================================="
echo "ClinFuseDiff • Step 1: Setup Environment"
echo "==========================================="
echo
echo "This step creates the 'clinfusediff' conda environment"
echo "and installs all required dependencies."
echo

if [[ ! -f "scripts/setup_env.sh" ]]; then
  echo "ERROR: scripts/setup_env.sh not found. Please check your repository clone." >&2
  exit 1
fi

bash scripts/setup_env.sh

echo
echo "==========================================="
echo "Environment setup complete."
echo "Activate with:"
echo "  conda activate clinfusediff"
echo "==========================================="
