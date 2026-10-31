// TileJSON 3.0 parser. See
// <https://github.com/mapbox/tilejson-spec/blob/master/3.0.0/README.md>.
//
// Real-world TileJSON documents (MapTiler, OpenMapTiles, Mapbox-
// compatible vendors) ship lots of optional and vendor-extension
// fields — `crs`, `crs_wkt`, `extent`, `mvt_buffer`, `generator`, …
// — that aren't in the spec but should not cause us to error. We
// model the fields we actively read and ignore the rest via
// `serde(flatten)` into a `_other` map.
//
// Only the subset of TileJSON the `render` module needs is exposed
// here: enough to drive an MVT `{z}/{x}/{y}` URL fetcher and to feed
// `parse_layer_specs` with a `vector_layers` blob shaped the way the
// PMTiles metadata reader expects.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::style::span::{self, ByteSpan, JsonPath};

/// One vector-layer descriptor as it appears in TileJSON's
/// `vector_layers` array. Field names match the spec; unknown keys
/// captured into `extra` so vendor extensions survive a parse →
/// re-emit round-trip without complaint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VectorLayer {
    pub id: String,
    #[serde(default)]
    pub fields: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minzoom: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maxzoom: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Parsed TileJSON document. Required fields are typed strictly;
/// optional / unrecognised keys are captured into `extra` so vendor
/// extensions like MapTiler's `crs` / `mvt_buffer` don't blow up the
/// deserializer.
#[derive(Debug, Clone, Deserialize)]
pub struct TileJson {
    /// TileJSON version string, e.g. `"2.0.0"` or `"3.0.0"`. We accept
    /// any value here; the spec versions only differ in optional
    /// fields we already model permissively.
    #[serde(default)]
    pub tilejson: Option<String>,

    /// URL templates with `{z}` / `{x}` / `{y}` placeholders.
    /// Required by the spec; we reject documents without it.
    pub tiles: Vec<String>,

    /// Minimum zoom level. Default `0` per spec.
    #[serde(default)]
    pub minzoom: u8,

    /// Maximum zoom level. Default `30` per spec.
    #[serde(default = "default_maxzoom")]
    pub maxzoom: u8,

    /// `[west, south, east, north]` in WGS-84 degrees.
    #[serde(default)]
    pub bounds: Option<[f64; 4]>,

    /// `[lon, lat]` or `[lon, lat, zoom]`. Spec allows either.
    #[serde(default)]
    pub center: Option<Vec<f64>>,

    #[serde(default)]
    pub attribution: Option<String>,

    /// `"pbf"` for MVT. We accept other values but the render layer
    /// will refuse to ingest them.
    #[serde(default)]
    pub format: Option<String>,

    /// `"gzip"` or `"none"`. Many TileJSON 3.0 docs advertise gzip
    /// even though the HTTP layer also transparently negotiates it —
    /// we sniff per-response and decode either way.
    #[serde(default)]
    pub compression: Option<String>,

    /// Per-layer metadata used by the inspector / layer panel.
    #[serde(default)]
    pub vector_layers: Option<Vec<VectorLayer>>,

    /// Any other key in the document. Captured so we don't reject
    /// vendor extensions.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

fn default_maxzoom() -> u8 {
    30
}

/// Errors raised while parsing TileJSON or substituting URL
/// templates.
#[derive(Debug, Error)]
pub enum TileJsonError {
    /// JSON syntax error or structural deserialize failure. Carries the
    /// 1-based `line` / `column` from `serde_json::Error` along with a
    /// derived byte span pointing at that position in the source.
    #[error("{message} (at line {line}, column {column})")]
    Json {
        line: usize,
        column: usize,
        message: String,
        span: ByteSpan,
    },

    /// `tiles` array is empty. Carries the byte span of the array so
    /// the snippet renderer can highlight the empty `[]`.
    #[error("TileJSON document has an empty `tiles` array")]
    NoTiles { span: ByteSpan },
}

impl TileJsonError {
    /// Byte span of the offending fragment in the source, if known.
    pub fn span(&self) -> Option<ByteSpan> {
        match self {
            TileJsonError::Json { span, .. } => Some(*span),
            TileJsonError::NoTiles { span } => Some(*span),
        }
    }

    fn from_serde_json(err: serde_json::Error, src: &str) -> Self {
        let line = err.line();
        let column = err.column();
        let offset = span::byte_offset_from_line_col(src, line, column);
        let end = offset.saturating_add(1).min(src.len());
        TileJsonError::Json {
            line,
            column,
            message: err.to_string(),
            span: ByteSpan::new(offset, end),
        }
    }
}

impl From<serde_json::Error> for TileJsonError {
    fn from(err: serde_json::Error) -> Self {
        // Source-less fallback for legacy call sites that don't have
        // the original document text. Prefer
        // [`TileJsonError::from_serde_json`] when the source is in hand.
        TileJsonError::Json {
            line: err.line(),
            column: err.column(),
            message: err.to_string(),
            span: ByteSpan::empty(),
        }
    }
}

impl TileJson {
    /// Parse from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, TileJsonError> {
        // Parse twice on the fast path: once as `Value` to build the
        // span map for any structural error, then again into the typed
        // `TileJson`. Documents are small (low KB) so the duplicated
        // work is invisible.
        let raw: serde_json::Value =
            serde_json::from_str(s).map_err(|e| TileJsonError::from_serde_json(e, s))?;
        let map = span::build_span_map(s, &raw);
        let tj: Self = serde_json::from_value(raw)
            .map_err(|e| TileJsonError::from_serde_json(e, s))?;
        if tj.tiles.is_empty() {
            let span = map
                .lookup(&JsonPath::root().child_key("tiles"))
                .unwrap_or(ByteSpan::empty());
            return Err(TileJsonError::NoTiles { span });
        }
        Ok(tj)
    }

    /// Resolve `tiles[0]` with the given coordinates substituted.
    /// Substitutes `{z}` / `{x}` / `{y}` only — other placeholders
    /// (`{ratio}`, `{quadkey}`, …) are deferred until we need them.
    pub fn tile_url(&self, z: u8, x: u32, y: u32) -> String {
        fill_template(&self.tiles[0], z, x, y)
    }
}

/// Substitute `{z}` / `{x}` / `{y}` in `template`. Other tokens are
/// left untouched.
pub fn fill_template(template: &str, z: u8, x: u32, y: u32) -> String {
    template
        .replace("{z}", &z.to_string())
        .replace("{x}", &x.to_string())
        .replace("{y}", &y.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // MapTiler's `tiles.json` response captured 2026-05-18. We keep
    // a trimmed copy in-line so this test doesn't require the
    // network; the full document includes vendor fields like
    // `crs_wkt` and `extent`, which we exercise here to confirm
    // they don't fail deserialization.
    const MAPTILER_SAMPLE: &str = r#"{
        "tilejson": "2.0.0",
        "format": "pbf",
        "version": "4.0.0",
        "type": "baselayer",
        "minzoom": 0,
        "maxzoom": 15,
        "compression": "gzip",
        "datestamp": "2026-05-12T00:00:00Z",
        "generator": "MapTiler Data",
        "name": "MapTiler Planet v4",
        "attribution": "© MapTiler",
        "bounds": [-180, -85.0511, 180, 85.0511],
        "center": [0, 0, 0],
        "crs": "EPSG:3857",
        "extent": [-1, -1, 1, 1],
        "tiles": ["https://api.maptiler.com/tiles/v4/{z}/{x}/{y}.pbf?key=ABC"],
        "vector_layers": [
            {"id": "water", "fields": {"class": "String"}, "minzoom": 0, "maxzoom": 15},
            {"id": "road",  "fields": {}, "minzoom": 6, "maxzoom": 15}
        ]
    }"#;

    #[test]
    fn parses_maptiler_v4_response() {
        let tj = TileJson::from_json_str(MAPTILER_SAMPLE).expect("parses");
        assert_eq!(tj.tilejson.as_deref(), Some("2.0.0"));
        assert_eq!(tj.minzoom, 0);
        assert_eq!(tj.maxzoom, 15);
        assert_eq!(tj.format.as_deref(), Some("pbf"));
        assert_eq!(tj.compression.as_deref(), Some("gzip"));
        assert_eq!(tj.tiles.len(), 1);
        assert!(tj.tiles[0].contains("{z}"));
        // Vendor extensions land in `extra` rather than blowing up
        // the deserializer.
        assert!(tj.extra.contains_key("crs"));
        assert!(tj.extra.contains_key("generator"));
        let layers = tj.vector_layers.as_ref().expect("vector_layers present");
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].id, "water");
        assert_eq!(layers[1].minzoom, Some(6));
    }

    #[test]
    fn rejects_empty_tiles() {
        let bad = r#"{"tilejson":"3.0.0","tiles":[]}"#;
        let err = TileJson::from_json_str(bad).unwrap_err();
        let TileJsonError::NoTiles { span } = err else {
            panic!("expected NoTiles, got {err:?}");
        };
        // The carried span must cover the (empty) `[]` so the snippet
        // renderer can highlight the offending value.
        assert_eq!(&bad[span.start..span.end], "[]");
    }

    #[test]
    fn fill_template_substitutes_xyz() {
        let url = fill_template(
            "https://api.example.com/v4/{z}/{x}/{y}.pbf?key=ABC",
            14,
            8567,
            5806,
        );
        assert_eq!(
            url,
            "https://api.example.com/v4/14/8567/5806.pbf?key=ABC"
        );
    }

    #[test]
    fn tile_url_picks_first_template() {
        let tj = TileJson::from_json_str(MAPTILER_SAMPLE).unwrap();
        let url = tj.tile_url(7, 65, 42);
        assert_eq!(
            url,
            "https://api.maptiler.com/tiles/v4/7/65/42.pbf?key=ABC"
        );
    }
}
