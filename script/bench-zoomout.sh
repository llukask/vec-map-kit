#!/usr/bin/env bash
# Count flicker while zooming out over central Vienna: frames with holes in
# the map and labels that vanish instead of fading. Same setup as
# `bench-vienna.sh`; runs RUNS times (default 3) since tile arrival timing
# varies.
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

for _ in $(seq "${RUNS:-3}"); do
    VEC_MAP_KIT_BENCH=zoomout "$root/target/release/examples/map" "$tilejson" "$style" 15.2 16.3725 48.2083 2>&1 |
        grep -E '^zoom-out'
done
