#!/bin/bash
# ===========================================================================
# install_sc2.sh — Install StarCraft II and SMAC v2 maps (Linux)
#
# This script is idempotent: re-run it safely to pick up missing pieces.
#
# Env vars you can set BEFORE running:
#   SC2PATH  — where to put StarCraft II (default: ~/StarCraftII)
#
# Container / noexec note:
#   If the target path lands on a noexec filesystem (e.g. /dev/shm inside
#   Docker), SC2_x64 will not execute even with correct mode bits.
#   In that case, run with SC2PATH pointing to a normal filesystem:
#     SC2PATH=/workspace/StarCraftII bash install_sc2.sh
#   Or use the automatic fix step at the bottom of this script.
#
# Usage:
#   bash install_sc2.sh
# ===========================================================================
set -euo pipefail

# ---------- paths ----------------------------------------------------------
if [ -z "${SC2PATH:-}" ]; then
    export SC2PATH="$HOME/StarCraftII"
fi
echo "SC2PATH=$SC2PATH"

# ---------- StarCraft II client (Linux, v4.10) ----------------------------
if [ ! -d "$SC2PATH" ]; then
    echo ">> Downloading StarCraft II Linux client (v4.10) …"
    wget -q --show-progress -O /tmp/SC2.4.10.zip \
        http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    echo ">> Extracting …"
    unzip -q -P iagreetotheeula /tmp/SC2.4.10.zip -d "$(dirname "$SC2PATH")"
    rm -f /tmp/SC2.4.10.zip
    echo ">> StarCraft II installed to $SC2PATH"
else
    echo ">> StarCraft II already installed at $SC2PATH"
fi

# ---------- SMAC v2 maps ---------------------------------------------------
MAP_DIR="$SC2PATH/Maps/SMAC_Maps"
if [ ! -d "$MAP_DIR" ] || [ -z "$(ls -A "$MAP_DIR" 2>/dev/null)" ]; then
    echo ">> Downloading SMAC v2 maps …"
    wget -q --show-progress -O /tmp/SMAC_Maps.zip \
        https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
    mkdir -p "$MAP_DIR"
    unzip -q -o /tmp/SMAC_Maps.zip -d /tmp/smac_maps_tmp
    mv /tmp/smac_maps_tmp/*.SC2Map "$MAP_DIR/" 2>/dev/null || true
    # Some zips nest files in a SMAC_Maps/ subfolder
    if [ -d /tmp/smac_maps_tmp/SMAC_Maps ]; then
        mv /tmp/smac_maps_tmp/SMAC_Maps/*.SC2Map "$MAP_DIR/" 2>/dev/null || true
    fi
    rm -rf /tmp/SMAC_Maps.zip /tmp/smac_maps_tmp
    echo ">> Maps installed to $MAP_DIR"
else
    echo ">> SMAC maps already present at $MAP_DIR"
fi

# ---------- noexec filesystem fix ------------------------------------------
# If SC2PATH lives on a noexec-mounted filesystem (common in containers where
# /dev/shm has the noexec flag), copy the executables to /workspace/StarCraftII
# so pysc2 can actually launch them.
NEED_EXEC_FIX=false
if ! python3 -c "import os,sys; sys.exit(0 if os.access('$SC2PATH/Versions/Base75689/SC2_x64',os.X_OK) else 1)" 2>/dev/null && \
   ! python3.13 -c "import os,sys; sys.exit(0 if os.access('$SC2PATH/Versions/Base75689/SC2_x64',os.X_OK) else 1)" 2>/dev/null; then
    NEED_EXEC_FIX=true
fi

if [ "$NEED_EXEC_FIX" = true ]; then
    EXEC_PATH="/workspace/StarCraftII"
    echo ">> WARNING: $SC2PATH appears to be on a noexec filesystem."
    echo ">> Copying binaries to $EXEC_PATH (symlinks for large data dirs)..."
    mkdir -p "$EXEC_PATH"
    cp -r "$SC2PATH/Versions"   "$EXEC_PATH/"
    cp -r "$SC2PATH/Libs"       "$EXEC_PATH/"
    cp -r "$SC2PATH/Interfaces" "$EXEC_PATH/"
    cp    "$SC2PATH/.build.info" "$EXEC_PATH/" 2>/dev/null || true
    for d in SC2Data Maps Battle.net Replays; do
        ln -sfn "$SC2PATH/$d" "$EXEC_PATH/$d"
    done
    export SC2PATH="$EXEC_PATH"
    echo ">> Binaries copied. SC2PATH updated to $SC2PATH"
fi

# ---------- Verify ----------------------------------------------------------
echo ""
echo "=== Verification ==="
echo "SC2PATH:  $SC2PATH"
echo "Maps dir: $MAP_DIR"
echo "Map files:"
ls -1 "$MAP_DIR"/*.SC2Map 2>/dev/null || echo "  (none found — check for errors above)"
echo ""
echo "Done!  Make sure SC2PATH is exported in your shell:"
echo "  export SC2PATH=$SC2PATH"
