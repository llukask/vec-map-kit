#!/usr/bin/env bash
# Vendor the maplibre-style-spec expression test fixtures into
# `tests/fixtures/maplibre-style-spec/`. This is an infrequent
# operation — re-run it to refresh against a newer upstream tag.
#
# Pinning rationale: we vendor the fixtures (rather than pulling
# them as a git submodule) so the repo builds and tests cleanly
# offline, and so `cargo test` has no network dependency. The
# trade-off is that picking up upstream changes requires a
# deliberate re-run of this script and a commit.
#
# Usage:
#   script/vendor_maplibre_fixtures.sh              # uses the pinned ref
#   script/vendor_maplibre_fixtures.sh v24.9.0      # vendor a different tag
#   script/vendor_maplibre_fixtures.sh <sha>        # vendor a specific commit

set -euo pipefail

# Pinned ref — bump alongside the manifest when refreshing fixtures.
# Latest tagged release as of 2026-05-16.
PINNED_REF="v24.8.5"

REF="${1:-$PINNED_REF}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_ROOT="$REPO_ROOT/tests/fixtures/maplibre-style-spec"
WORK_DIR="$REPO_ROOT/tmp/maplibre-fixture-vendor"
TARBALL_URL="https://codeload.github.com/maplibre/maplibre-style-spec/tar.gz/refs/tags/$REF"

# If the ref isn't a tag (e.g. raw SHA), fall back to the commit URL.
if ! curl -fsSI "$TARBALL_URL" >/dev/null 2>&1; then
    TARBALL_URL="https://codeload.github.com/maplibre/maplibre-style-spec/tar.gz/$REF"
fi

echo "Vendoring maplibre-style-spec fixtures from $REF"
echo "  source: $TARBALL_URL"
echo "  dest:   $DEST_ROOT"

# Fresh working tree so we never mix stale extracts.
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# Download and extract only what we want. `--strip-components=1`
# drops the `maplibre-style-spec-<ref>/` prefix, leaving paths like
# `test/integration/expression/tests/...` directly under WORK_DIR.
#
# tar's `--wildcards` matches a directory pattern by entry name, not
# recursively, but it does pull in everything under a matched
# directory, so a single anchor at the directory itself plus a
# pattern that matches descendants is enough. We use a broad
# `*/test/integration/expression/*` to catch all descendants.
curl -fsSL "$TARBALL_URL" \
    | tar -xz -C "$WORK_DIR" \
        --strip-components=1 \
        --wildcards \
        'maplibre-style-spec-*/test/integration/expression/*' \
        'maplibre-style-spec-*/LICENSE.txt'

SRC_FIXTURES="$WORK_DIR/test/integration/expression/tests"
SRC_LICENSE="$WORK_DIR/LICENSE.txt"

if [ ! -d "$SRC_FIXTURES" ]; then
    echo "error: expected $SRC_FIXTURES after extraction" >&2
    exit 1
fi
if [ ! -f "$SRC_LICENSE" ]; then
    echo "error: expected $SRC_LICENSE after extraction" >&2
    exit 1
fi

# Replace the vendored tree wholesale so removed-upstream fixtures
# disappear from our copy too.
rm -rf "$DEST_ROOT/expression"
mkdir -p "$DEST_ROOT"
cp -R "$SRC_FIXTURES" "$DEST_ROOT/expression"
cp "$SRC_LICENSE" "$DEST_ROOT/LICENSE-MapLibre.txt"

# Resolve the ref to a concrete commit SHA for reproducibility. If
# the API call fails (no network at refresh time, rate-limit, etc.)
# we still record the ref as given.
RESOLVED_SHA="$(curl -fsSL "https://api.github.com/repos/maplibre/maplibre-style-spec/commits/$REF" \
    | sed -n 's/^  "sha": "\([0-9a-f]\{40\}\)",$/\1/p' \
    | head -n 1 || true)"

{
    echo "# Vendored from https://github.com/maplibre/maplibre-style-spec"
    echo "# Refreshed by script/vendor_maplibre_fixtures.sh"
    echo "ref=$REF"
    if [ -n "$RESOLVED_SHA" ]; then
        echo "sha=$RESOLVED_SHA"
    fi
} > "$DEST_ROOT/VERSION"

FIXTURE_COUNT="$(find "$DEST_ROOT/expression" -name 'test.json' | wc -l | tr -d ' ')"

echo "Vendored $FIXTURE_COUNT fixtures from $REF"
echo "  manifest: $DEST_ROOT/VERSION"
echo "  license:  $DEST_ROOT/LICENSE-MapLibre.txt"

rm -rf "$WORK_DIR"
