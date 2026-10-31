# GPUI patches

Small changes to GPUI's path pipeline and glyph painting, applied to the exact crate
versions gpui-kit 0.8 pins (`gpui-pre` / `gpui-pre-wgpu` 0.3.9) by
`script/vendor-gpui.sh`. Both are meant to go upstream. Once a released
gpui-kit contains them, delete this directory, the script, and the
`[patch.crates-io]` section in `Cargo.toml`.

## Using vec-map-kit from an application

vec-map-kit does not build against unpatched GPUI. `[patch]` sections
only apply to the workspace that declares them, so an application needs
its own copy:

```toml
[patch.crates-io]
gpui-pre = { path = "path/to/vendor/gpui-pre" }
gpui-pre-wgpu = { path = "path/to/vendor/gpui-pre-wgpu" }
```

with the crates materialised by running this repository's
`script/vendor-gpui.sh` (it writes to `vendor/` next to it). Each patch
exports a `VEC_MAP_KIT_PATCH_*` marker constant that `src/lib.rs` refers
to, so a GPUI missing any patch fails with "cannot find value
`VEC_MAP_KIT_PATCH_...`" instead of building a map that has no labels or
runs three times slower.

## Why

A vector map draws every fill and stroke as GPUI path triangles, and
GPUI re-submits all of them each frame while the map pans. Over central
Vienna that is ~330k triangles (~1M vertices) per frame. Unpatched, GPUI
spent ~250 ns per triangle on the main thread: the vertices were copied
four times, and each one was widened to a ~104-byte record before upload,
so a frame pushed ~100 MB through `queue.write_buffer`.

## The patches

- `gpui-pre-wgpu-0.3.9-compact-path-vertices.patch`. Colour and clip
  bounds are per path, so they move into a per-path `PathRasterizationInfo`
  array bound next to the vertices; a vertex shrinks to 24 bytes (position,
  st, path index). Also sizes the vertex `Vec` up front.
  Known gap: the WebGL texture transport (`shaders_webgl.wgsl`) is not
  updated, so this breaks GPUI's web target as is.
- `gpui-pre-0.3.9-move-paths-into-scene.patch`. `Window::paint_path`
  scales vertices in place instead of collecting a new buffer, and
  `Scene::insert_primitive` moves a path into `paths` and logs only its id
  for `replay`, instead of storing a second clone in `paint_operations`.

- `gpui-pre-0.3.9-glyph-instances.patch`. Adds
  `Window::paint_glyph_instances`: rasterise a glyph once (grayscale, no
  subpixel offset) and place copies of it, each with its own origin,
  colour and rotation about its baseline origin. Map labels need this for
  text that follows a road, and for halos drawn as eight offset copies
  under each glyph. GPUI's sprite shader already applies a per-sprite
  transformation; `paint_glyph` just always passed the identity.

## Measured effect

`script/bench-vienna.sh`, main-thread CPU per frame, tiles loaded, 60
redraws per zoom, 1596x970 viewport:

| | mean | worst (z14.25) |
|---|---|---|
| unpatched | 42.2 ms | 93.8 ms |
| + compact path vertices | 11.7 ms | 21.0 ms |
| + no per-frame path clones | 9.9 ms | 17.0 ms |

A screenshot of the same view differs from the unpatched render in 56 of
1.55M pixels (ImageMagick `compare -metric AE`), with no visible change.
