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
