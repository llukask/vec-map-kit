#!/usr/bin/env bash
# Materialise patched copies of the GPUI crates that `Cargo.toml`'s
# `[patch.crates-io]` section points at. Each crate is downloaded from
# crates.io at the exact version gpui-kit pins, unpacked into `vendor/`,
# and the matching `patches/<crate>-<version>-*.patch` files are applied.
#
# Idempotent: a crate is rebuilt only when its patch set changes. The dev
# shell runs this on entry, so `nix develop` always has them in place.
#
# The patches cut GPUI's per-vertex cost for paths (see patches/README.md);
# they are meant to go upstream, after which this script and the
# `[patch]` section can be deleted.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
vendor=$root/vendor
mkdir -p "$vendor"

for spec in gpui-pre:0.3.9 gpui-pre-wgpu:0.3.9; do
    crate=${spec%%:*}
    version=${spec##*:}
    patches=("$root/patches/$crate-$version"-*.patch)
    stamp=$(cat "${patches[@]}" | sha256sum | cut -d' ' -f1)
    dest=$vendor/$crate
    if [[ -f $dest/.vendor-stamp && $(cat "$dest/.vendor-stamp") == "$stamp" ]]; then
        continue
    fi

    echo "vendoring $crate $version with ${#patches[@]} patch(es)" >&2
    rm -rf "$dest"
    scratch=$(mktemp -d "$vendor/.download.XXXXXX")
    curl -fsSL "https://static.crates.io/crates/$crate/$crate-$version.crate" | tar -xz -C "$scratch"
    mv "$scratch/$crate-$version" "$dest"
    rmdir "$scratch"
    for p in "${patches[@]}"; do
        patch -s -p1 -d "$dest" <"$p"
    done
    echo "$stamp" >"$dest/.vendor-stamp"
done
