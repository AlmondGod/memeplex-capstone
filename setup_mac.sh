#!/bin/bash
# ===========================================================================
# setup_mac.sh — Set up SMAC v2 on macOS (Blizzard-installed StarCraft II)
#
# This script is macOS-specific. Linux users: see install_sc2.sh + SETUP.md.
#
# Prerequisites:
#   - StarCraft II installed via Blizzard launcher (usually /Applications/StarCraft II)
#   - Python 3.9 with torch & numpy (brew install python@3.9 + pip install torch numpy)
#
# Usage:
#   bash setup_mac.sh
#   # Then add the printed SC2PATH export to your ~/.zshrc or ~/.bashrc
# ===========================================================================
set -euo pipefail

# ---------- Platform check ------------------------------------------------
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script is for macOS only."
    echo "Linux users: run install_sc2.sh and follow SETUP.md instead."
    exit 1
fi

# ---------- Detect the Python interpreter that has torch ------------------
PYTHON=""
for candidate in python3.9 python3.10 python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        if "$candidate" -c "import torch" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Could not find a Python interpreter with 'torch' installed."
    echo "Install torch first:  pip3 install torch"
    exit 1
fi

echo ">> Using Python: $PYTHON ($($PYTHON --version))"

# ---------- SC2 path detection --------------------------------------------
DEFAULT_SC2_PATH="/Applications/StarCraft II"

if [ -z "${SC2PATH:-}" ]; then
    if [ -d "$DEFAULT_SC2_PATH" ]; then
        export SC2PATH="$DEFAULT_SC2_PATH"
    else
        echo "ERROR: StarCraft II not found at '$DEFAULT_SC2_PATH'."
        echo "Please install it via the Blizzard launcher, then re-run, or set:"
        echo "  export SC2PATH=/path/to/StarCraftII"
        exit 1
    fi
fi
echo "SC2PATH=$SC2PATH"

# ---------- SMAC v2 maps --------------------------------------------------
MAP_DIR="$SC2PATH/Maps/SMAC_Maps"
if [ ! -d "$MAP_DIR" ] || [ -z "$(ls -A "$MAP_DIR" 2>/dev/null)" ]; then
    echo ">> Downloading SMAC v2 maps (~500 KB compressed)..."
    curl -L -o /tmp/SMAC_Maps.zip \
        https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
    mkdir -p "$MAP_DIR"
    unzip -o /tmp/SMAC_Maps.zip -d /tmp/smac_maps_tmp
    # Handle both flat and nested zip layouts
    mv /tmp/smac_maps_tmp/*.SC2Map "$MAP_DIR/" 2>/dev/null || true
    if [ -d /tmp/smac_maps_tmp/SMAC_Maps ]; then
        mv /tmp/smac_maps_tmp/SMAC_Maps/*.SC2Map "$MAP_DIR/" 2>/dev/null || true
    fi
    rm -rf /tmp/SMAC_Maps.zip /tmp/smac_maps_tmp
    echo ">> Maps installed to $MAP_DIR"
else
    echo ">> SMAC maps already present at $MAP_DIR"
fi

# ---------- Python dependencies -------------------------------------------
echo ""
echo ">> Installing Python dependencies via $PYTHON -m pip..."
"$PYTHON" -m pip install --quiet "pysc2>=4.0.0"
"$PYTHON" -m pip install --quiet "smacv2 @ git+https://github.com/oxwhirl/smacv2.git@main"
"$PYTHON" -m pip install --quiet "tqdm>=4.65.0" "matplotlib>=3.7.0" \
    "gymnasium>=0.28.0" six
echo ">> Python dependencies installed."

# ---------- Verify --------------------------------------------------------
echo ""
echo "=== Verification ==="
echo "SC2PATH : $SC2PATH"
echo "SC2 app : $SC2PATH/Versions/"
ls "$SC2PATH/Versions/" 2>/dev/null || echo "  (no Versions dir found)"
echo "Maps    : $MAP_DIR"
ls "$MAP_DIR"/*.SC2Map 2>/dev/null | wc -l | xargs echo "  map count:"
echo ""
echo ">> Testing Python imports..."
"$PYTHON" -c "
import pysc2; print('  pysc2 OK')
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper; print('  smacv2 OK')
import torch; print(f'  torch {torch.__version__} OK')
import numpy; print(f'  numpy {numpy.__version__} OK')
"

echo ""
echo "=== Done! ==="
echo ""
echo "Add this to your ~/.zshrc (or ~/.bashrc) to persist SC2PATH:"
echo ""
echo "  export SC2PATH=\"$SC2PATH\""
echo ""
echo "Then run a smoke test:"
echo "  export SC2PATH=\"$SC2PATH\""
echo "  $PYTHON run_smacv2_mappo.py --mode test --test-episodes 1"
