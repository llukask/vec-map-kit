// Mapbox Vector Tile decoding. `proto` holds the raw `prost` bindings
// (the wire format); the rest hangs a small, ergonomic surface off them
// — `Tile` / `Layer` / `Feature` with `geo` geometries and resolved
// property maps — so callers don't have to learn the MVT command
// encoding or dictionary indirection.

mod error;
mod geometry;
mod model;
pub mod proto;

pub use error::{Error, Result};
pub use geometry::{Geometry, decode_geometry};
pub use model::{Feature, Layer, Tile, Value};
