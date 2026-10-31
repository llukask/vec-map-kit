#!/usr/bin/env bash
# Run the `map` example in the headless compositor and save a screenshot.
#
#     script/screenshot.sh OUT.png TILEJSON_URL STYLE [ZOOM [LON LAT]]   (SETTLE=seconds)
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
out=$1 source=$2 style=$3 settle=${SETTLE:-4}
shift 3

WAYLAND_DISPLAY=$("$root/script/headless-sway.sh")
export WAYLAND_DISPLAY
unset DISPLAY
grim=$(nix build nixpkgs#grim --no-link --print-out-paths | tail -1)/bin/grim

"$root/target/release/examples/map" "$source" "$style" "$@" >"$root/tmp/map.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
sleep "$settle"
"$grim" "$out"
