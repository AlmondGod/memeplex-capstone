#!/bin/bash
# ===========================================================================
# install_sc2.sh — Install StarCraft II and SMAC v2 maps (Linux)
#
# This script is idempotent: re-run it safely to pick up missing pieces.
#
# Env vars you can set BEFORE running:
#   SC2PATH  — where to put StarCraft II (default: ~/StarCraftII)
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
