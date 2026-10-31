# Session notes

## Environment

- `mvt`, `style` and `tilejson` are copied from `../vec-map`,
  including its uncommitted changes (a `style/mod.rs` change that predates
  this crate, the ring-classification fix, `symbol-placement` as a
  `Property`). Those changes are still uncommitted in `../vec-map`, and the
  two copies will drift.
- Comments in the copied `style` code refer to vec-map's roadmap and
  "Phase N" milestones, none of which exist here. `cargo doc` also warns
  about 14 links from public docs to private or missing items there.
- `src/mvt/proto.rs` is checked-in `prost-build` output; regenerating it
  needs a throwaway build script (see its header).
- `perf record` fails inside the agent sandbox (no perf events), so gpui's
  internal frame cost is measured as main-thread CPU time instead.
- Headless sway caps frame callbacks at ~30 fps on its 60 Hz output, even
  for a trivial scene, so fps numbers from the bench are not meaningful;
  main-thread CPU per frame is.
- `WindowOptions` window bounds are ignored by the user's tiling
  compositor, so probe window sizes there were not what was requested.

## Rendering follow-ups

- The GPUI patches in `patches/` should go upstream (gpui-kit pins
  `gpui-pre` 0.3.9). The compact-vertex patch does not update the WebGL
  instance transport (`shaders_webgl.wgsl`), so it breaks GPUI's web
  target as is; that needs a `load_path_info` there before upstreaming.
- Apps embedding `MapView` only get the GPUI patches if they add the same
  `[patch.crates-io]` section; `[patch]` doesn't propagate to dependents.
- Triangles are culled per triangle against the tile clip rect, but every
  triangle of a visible tile is still projected each frame. A per-tile
  spatial chunking would skip off-screen chunks wholesale.
- Path vertices still carry a constant `st_position` (8 of their 24 bytes)
  that solid fills never use; a solid-path flag upstream would drop it.
- Tile arrivals each `cx.notify()` the view; a burst of arrivals causes
  several full repaints in a row (visible as CPU spikes while zooming
  across a level).
- No labels, icons, `fill-pattern`, `line-dasharray`, `line-gap-width`,
  `line-offset`, `fill-outline-color`, `fill-translate`, or
  `fill-extrusion` yet. Road casings in maptiler styles are missing
  because of `line-gap-width`.
- `style::Color::parse` documents that named CSS colours are unsupported;
  styles using `"white"` etc. would fall back to black.
- `text-font` in the core is a constant `Vec<String>`, not an expression;
  styles with zoom-dependent fonts will need a core change.

## Network sources

- A tile that fails to fetch (timeout, 5xx) is cached as absent until it
  is evicted; there is no retry with backoff yet.
- The style's own `sources` are ignored; every layer draws from the one
  `TileJsonSource` passed to `MapView`. A style URL alone is not enough to
  open a map yet.
- `script/bench-vienna.sh` spends ~100 MapTiler tile requests per run
  against `MAPTILER_KEY` (from `.env`). It fetches MapTiler's current
  online style, so results can shift when MapTiler edits that style.

## Labels follow-ups

- Icons (`icon-image`, sprites) are not drawn; symbol layers render text
  only. `fill-pattern` needs the same sprite atlas.
- Not applied yet: `text-letter-spacing`, `text-justify`,
  `text-variable-anchor`, `text-rotate`, `text-halo-blur`,
  `text-writing-mode`, `symbol-avoid-edges`, `symbol-z-order`, and
  data-driven halo colour/width.
- Line anchors are chosen at build time against an estimated label length
  (0.6 em per char); a label longer than the estimate is checked again when
  placed and dropped if it doesn't fit, so some short streets lose names.
- Glyphs are bitmaps per (font, size): sizes are rounded to 0.5 px, so
  zoom-interpolated `text-size` steps slightly rather than gliding.
- Collision boxes for line labels are one square per glyph; MapLibre uses
  circles along the line and is a little denser on curvy streets.
- Labels in the neighbouring tile's MVT buffer are dropped to avoid
  duplicates, but a label straddling a tile edge can still be clipped by
  the collision pass of the tile that owns it.
- `style`: `symbol-placement` keeps zoom functions; the other enum layout
  properties (`text-anchor`, `text-transform`,
  `line-cap`, ...) still drop zoom functions the same way it used to.

## Zoom-out flicker follow-ups

- Parent-level prefetch costs about 25% more tile requests (MapTiler
  quota). It only looks one level up; a fast zoom-out across several
  levels can still outrun it, and then the edges zooming out reveals stay
  background-coloured until tiles arrive.
- Labels are matched across tiles by layer, text and anchor within one
  font size. Line labels whose anchors differ between tile levels (symbol
  spacing is per tile level) don't match and cross-fade instead.
- Fading-out labels keep their old tile's mesh alive (via `Arc`) for the
  250 ms fade.
- `script/bench-vienna.sh` numbers vary by ~±30% with machine load; compare
  runs interleaved, on an idle machine, before reading small differences.
