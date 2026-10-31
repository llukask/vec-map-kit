// vec-map-kit: MapLibre-styled vector tile maps as a native GPUI view.
//
// The crate downloads the tiles a TileJSON document names (`tilejson`,
// `source`), decodes them (`mvt`), evaluates MapLibre styles against them
// (`style`), and owns everything between a decoded tile and GPUI's scene:
// the camera, per-tile mesh building, and the `MapView` entity that draws
// with GPUI paths so the map stays sharp at every fractional zoom.

pub mod camera;
pub mod label;
pub mod mesh;
pub mod mvt;
pub mod source;
pub mod style;
mod text;
pub mod tile;
pub mod tilejson;
mod view;

pub use camera::{Camera, WorldPoint};
pub use source::TileJsonSource;
pub use style::Style;
pub use tile::TileId;
pub use view::{FrameStats, MapView};

// ==============================================================================
// GPUI patch check
// ==============================================================================
//
// vec-map-kit only builds against GPUI with the patches in `patches/`: one
// adds the rotated-glyph call labels are drawn with, the other two make
// dense maps affordable. Each patch exports a marker constant, so a GPUI
// missing any of them stops the build on the line naming that patch, and
// the line itself says where the fix is.

const _: () = gpui_kit::VEC_MAP_KIT_PATCH_GLYPH_INSTANCES; // GPUI lacks vec-map-kit's patches, see its patches/README.md
const _: () = gpui_kit::VEC_MAP_KIT_PATCH_PATHS_INTO_SCENE; // GPUI lacks vec-map-kit's patches, see its patches/README.md
#[cfg(target_os = "linux")]
const _: () = gpui_wgpu::VEC_MAP_KIT_PATCH_COMPACT_PATH_VERTICES; // GPUI lacks vec-map-kit's patches, see its patches/README.md
