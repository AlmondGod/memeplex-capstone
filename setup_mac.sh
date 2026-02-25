#!/bin/bash
# ===========================================================================
# setup_mac.sh — Set up SMAC v2 on macOS (Blizzard-installed StarCraft II)
#
# Prerequisites:
#   - StarCraft II installed via Blizzard launcher (usually /Applications/StarCraft II)
#   - Python 3 with pip3  (brew or system; needs torch & numpy)
#
# Usage:
#   bash setup_mac.sh
#   # Then add the printed SC2PATH export to your ~/.zshrc or ~/.bashrc
# ===========================================================================
set -euo pipefail

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
    echo ">> Downloading SMAC v2 maps (~177 MB)..."
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
echo ">> Installing Python dependencies via pip3..."
pip3 install --quiet "pysc2>=4.0.0"
pip3 install --quiet "smacv2 @ git+https://github.com/oxwhirl/smacv2.git@main"
pip3 install --quiet "torch>=2.0.0" numpy "tqdm>=4.65.0" "matplotlib>=3.7.0" \
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
python3 -c "
import pysc2; print('  pysc2 OK')
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper; print('  smacv2 OK')
import torch; print(f'  torch {torch.__version__} OK')
"

echo ""
echo "=== Done! ==="
echo ""
echo "Add this to your ~/.zshrc (or ~/.bashrc) to persist SC2PATH:"
echo ""
echo '  export SC2PATH="'"$SC2PATH"'"'
echo ""
echo "Then run a smoke test:"
echo '  export SC2PATH="'"$SC2PATH"'"'
echo "  python3 run_smacv2_mappo.py --mode test --test-episodes 1"
