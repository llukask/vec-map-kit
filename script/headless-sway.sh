#!/usr/bin/env bash
# Start a headless sway compositor so GUI probes and examples render on a
# real GPU without opening windows on (or stealing focus from) the user's
# desktop session. Prints the WAYLAND_DISPLAY to export; reuses a running
# instance if one is already up.
#
#     export WAYLAND_DISPLAY=$(script/headless-sway.sh); unset DISPLAY
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/tmp"
pidfile="$root/tmp/headless-sway.pid"
displayfile="$root/tmp/headless-sway.display"

if [[ -f $pidfile ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    cat "$displayfile"
    exit 0
fi

sway=$(nix build nixpkgs#sway --no-link --print-out-paths | tail -1)/bin/sway
printf 'output HEADLESS-1 resolution 1600x1000@60Hz\n' >"$root/tmp/sway-headless.conf"

before=$(find "$XDG_RUNTIME_DIR" -maxdepth 1 -name 'wayland-*' ! -name '*.lock' | sort)
env -u DISPLAY -u WAYLAND_DISPLAY \
    WLR_BACKENDS=headless WLR_LIBINPUT_NO_DEVICES=1 WLR_RENDERER=gles2 \
    "$sway" -c "$root/tmp/sway-headless.conf" >"$root/tmp/sway.log" 2>&1 &
echo $! >"$pidfile"

# sway picks the first free `wayland-N`; wait for a new socket to appear.
for _ in $(seq 50); do
    after=$(find "$XDG_RUNTIME_DIR" -maxdepth 1 -name 'wayland-*' ! -name '*.lock' | sort)
    socket=$(comm -13 <(echo "$before") <(echo "$after") | head -1)
    if [[ -n $socket ]]; then
        basename "$socket" | tee "$displayfile"
        exit 0
    fi
    sleep 0.1
done
echo "headless sway did not create a socket; see tmp/sway.log" >&2
exit 1
