#!/usr/bin/env bash
# Benchmark main-thread frame cost over central Vienna, the densest test
# case, in the headless compositor. Tiles and style come from MapTiler's
# OpenStreetMap map, with the key from MAPTILER_KEY or `.env`; STYLE and
# TILEJSON override either URL.
#
#     script/bench-vienna.sh
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
if [[ -z ${MAPTILER_KEY:-} && -f $root/.env ]]; then
    # shellcheck source=/dev/null
    source "$root/.env"
fi
style=${STYLE:-https://api.maptiler.com/maps/openstreetmap/style.json?key=${MAPTILER_KEY:?set MAPTILER_KEY or put it in .env}}
tilejson=${TILEJSON:-https://api.maptiler.com/tiles/v3-openmaptiles/tiles.json?key=$MAPTILER_KEY}

WAYLAND_DISPLAY=$("$root/script/headless-sway.sh")
export WAYLAND_DISPLAY
unset DISPLAY

VEC_MAP_KIT_BENCH=1 "$root/target/release/examples/map" "$tilejson" "$style" 14 16.3725 48.2083 2>&1 |
    grep -E '^(zoom|main-thread)'
