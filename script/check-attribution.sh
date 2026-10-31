#!/usr/bin/env bash
# Click both links in the attribution overlay of the `map` example and check
# they open MapTiler's and OpenStreetMap's copyright pages. Runs in the
# headless compositor; a stand-in `xdg-open` records URLs instead of
# opening a browser, and the D-Bus portal fallback is cut off.
#
# Pointer input comes through wayvnc: headless sway has no pointer device,
# and wayvnc adds a virtual one for as long as a VNC client is connected.
# The click positions assume the 1600x1000 headless output and MapTiler's
# attribution text.
#
#     script/check-attribution.sh
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
wayvnc=$(nix build nixpkgs#wayvnc --no-link --print-out-paths | tail -1)/bin/wayvnc
vncdo=$(nix build nixpkgs#python3Packages.vncdotool --no-link --print-out-paths | tail -1)/bin/vncdo

fakebin=$root/tmp/fakebin
opened=$root/tmp/opened.log
mkdir -p "$fakebin"
printf '#!/bin/sh\necho "$@" >> %s\n' "$opened" >"$fakebin/xdg-open"
chmod +x "$fakebin/xdg-open"
rm -f "$opened"

"$wayvnc" 127.0.0.1 5917 >"$root/tmp/wayvnc.log" 2>&1 &
vnc=$!
env -u DBUS_SESSION_BUS_ADDRESS PATH="$fakebin:$PATH" \
    "$root/target/release/examples/map" "$tilejson" "$style" 15.2 16.3725 48.2083 >"$root/tmp/map.log" 2>&1 &
map=$!
trap 'kill $map $vnc 2>/dev/null || true' EXIT
sleep 6

# GPUI's `InteractiveText` registers its mouse-up handler on the frame after
# the press, so a press and release sent back to back never click; hold the
# button for 150 ms like a person would.
press() { echo "move $1 $2 pause 0.4 mousedown 1 pause 0.15 mouseup 1 pause 0.8"; }
# shellcheck disable=SC2046
timeout 40 "$vncdo" -s 127.0.0.1::5917 move 10 10 pause 0.5 $(press 1380 986) $(press 1520 986)

expected=$'https://www.maptiler.com/copyright/\nhttps://www.openstreetmap.org/copyright'
if [[ $(cat "$opened" 2>/dev/null) == "$expected" ]]; then
    echo "attribution links ok"
else
    echo "attribution links opened:" >&2
    cat "$opened" >&2 2>/dev/null || echo "(nothing)" >&2
    exit 1
fi
