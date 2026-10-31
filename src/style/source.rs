// A style document's `sources` map identifies where tile data lives.
// The MapLibre spec defines several source kinds (`vector`, `raster`,
// `geojson`, ...); we model:
//
//   * `vector` — vector-tile source. Accepts the spec's `url` form
//     (pointing at a `pmtiles://` archive or a TileJSON document) and
//     the inline `tiles` array, plus the optional `minzoom` /
//     `maxzoom` / `bounds` / `center` / `attribution` fields. The
//     renderer consumes these via [`render::TileSource::from_style_source`].
//
//   * `raster` — recognised but rendered as a no-op. Common in real
//     stylesheets where a vendor uses it only to carry an
//     `attribution` string; refusing the variant would force callers
//     to hand-trim their style documents.
//
//   * `geojson` — inline GeoJSON object or a URL pointing at one. The
//     renderer synthesizes a virtual MVT layer (named after the source)
//     per requested tile by projecting the features into the tile's
//     MVT space; style layers reference the data via `source: <name>`
//     without a `source-layer`.
//
// Anything else (`raster-dem`, `image`, `video`, …) still fails to
// parse so we notice when a style document needs broader support.

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum Source {
    /// `type: "vector"` — vector-tile source.
    ///
    /// Either `url` (pointing at a TileJSON document or a
    /// `pmtiles://...` archive) or `tiles` (inline URL templates)
    /// drives the data fetch. The remaining fields are optional
    /// per-spec; they're surfaced verbatim so a renderer can choose
    /// whether to honour them.
    #[serde(rename = "vector")]
    Vector {
        /// TileJSON URL or `pmtiles://...` URL. Mutually compatible
        /// with `tiles` — when both are present, the renderer is
        /// free to skip the TileJSON fetch and use the inline
        /// templates.
        #[serde(default)]
        url: Option<String>,
        #[serde(default)]
        attribution: Option<String>,
        /// Inline `{z}/{x}/{y}` URL templates. Lets a style document
        /// carry the data location directly without a TileJSON
        /// round-trip.
        #[serde(default)]
        tiles: Option<Vec<String>>,
        #[serde(default)]
        minzoom: Option<u8>,
        #[serde(default)]
        maxzoom: Option<u8>,
        /// `[west, south, east, north]` in WGS-84 degrees.
        #[serde(default)]
        bounds: Option<[f64; 4]>,
        /// `[lon, lat]` or `[lon, lat, zoom]`.
        #[serde(default)]
        center: Option<Vec<f64>>,
    },

    /// `type: "raster"` — raster-tile source. Accepted so style
    /// documents that include an attribution-only raster source (or
    /// a vendor logo overlay) still parse, but **not rendered**.
    /// Honour the fields once we actually grow raster support.
    #[serde(rename = "raster")]
    Raster {
        #[serde(default)]
        url: Option<String>,
        #[serde(default)]
        attribution: Option<String>,
        #[serde(default)]
        tiles: Option<Vec<String>>,
        #[serde(default)]
        minzoom: Option<u8>,
        #[serde(default)]
        maxzoom: Option<u8>,
        #[serde(default)]
        tile_size: Option<u32>,
        #[serde(default)]
        bounds: Option<[f64; 4]>,
    },

    /// `type: "geojson"` — inline or URL-referenced GeoJSON data. The
    /// renderer projects the features into per-tile MVT space at
    /// fetch time and exposes a virtual layer whose name matches this
    /// source's key in `style.sources`. Style layers bound to a
    /// `geojson` source reference it via `source: <name>` and omit
    /// `source-layer` (the source IS the layer).
    #[serde(rename = "geojson")]
    GeoJson {
        /// Either an inline GeoJSON object (`Feature`,
        /// `FeatureCollection`, or a bare geometry) or a string URL
        /// pointing at one. Untyped here — the renderer parses it
        /// into a typed feature collection at bootstrap.
        data: serde_json::Value,
        #[serde(default)]
        attribution: Option<String>,
        /// Optional bounding box in WGS-84 degrees. Unused today; the
        /// renderer derives the bounds from the feature geometries.
        #[serde(default)]
        bounds: Option<[f64; 4]>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vector_source() {
        let src: Source = serde_json::from_str(
            r#"{
                "type": "vector",
                "url": "pmtiles://https://example.com/tiles.pmtiles",
                "attribution": "© OpenStreetMap"
            }"#,
        )
        .unwrap();
        assert_eq!(
            src,
            Source::Vector {
                url: Some("pmtiles://https://example.com/tiles.pmtiles".into()),
                attribution: Some("© OpenStreetMap".into()),
                tiles: None,
                minzoom: None,
                maxzoom: None,
                bounds: None,
                center: None,
            }
        );
    }

    #[test]
    fn vector_source_url_is_optional() {
        let src: Source = serde_json::from_str(r#"{"type": "vector"}"#).unwrap();
        assert_eq!(
            src,
            Source::Vector {
                url: None,
                attribution: None,
                tiles: None,
                minzoom: None,
                maxzoom: None,
                bounds: None,
                center: None,
            }
        );
    }

    #[test]
    fn parses_vector_source_with_inline_tiles() {
        // Shape used by stylesheets that point at an MVT endpoint
        // directly, skipping the TileJSON round-trip.
        let src: Source = serde_json::from_str(
            r#"{
                "type": "vector",
                "tiles": ["https://example.com/{z}/{x}/{y}.pbf"],
                "minzoom": 0,
                "maxzoom": 14,
                "bounds": [-180, -85, 180, 85]
            }"#,
        )
        .unwrap();
        match src {
            Source::Vector {
                tiles,
                minzoom,
                maxzoom,
                bounds,
                ..
            } => {
                let templates = tiles.expect("tiles populated");
                assert_eq!(templates.len(), 1);
                assert!(templates[0].contains("{z}"));
                assert_eq!(minzoom, Some(0));
                assert_eq!(maxzoom, Some(14));
                assert_eq!(bounds, Some([-180.0, -85.0, 180.0, 85.0]));
            }
            _ => panic!("expected vector"),
        }
    }

    #[test]
    fn parses_attribution_only_raster_source() {
        // Common shape in real stylesheets: an empty raster source
        // exists purely to expose an `attribution` string for the
        // vendor.
        let src: Source = serde_json::from_str(
            r#"{
                "type": "raster",
                "attribution": "© Vendor"
            }"#,
        )
        .unwrap();
        assert!(matches!(src, Source::Raster { .. }));
    }

    #[test]
    fn parses_inline_geojson_source() {
        let src: Source = serde_json::from_str(
            r#"{
                "type": "geojson",
                "data": {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
                    "properties": {}
                }
            }"#,
        )
        .unwrap();
        match src {
            Source::GeoJson { data, .. } => {
                assert_eq!(data["type"], "Feature");
            }
            _ => panic!("expected geojson"),
        }
    }

    #[test]
    fn parses_url_geojson_source() {
        // `data` may be a string URL rather than an inline object.
        let src: Source = serde_json::from_str(
            r#"{"type": "geojson", "data": "https://example.com/x.geojson"}"#,
        )
        .unwrap();
        assert!(matches!(src, Source::GeoJson { .. }));
    }

    #[test]
    fn rejects_unknown_source_type() {
        let err = serde_json::from_str::<Source>(r#"{"type": "image", "url": "..."}"#)
            .unwrap_err();
        assert!(err.to_string().contains("image"));
    }

    #[test]
    fn rejects_unknown_field_on_vector() {
        // Spec extensions outside the modeled set (e.g. `scheme`)
        // still error rather than be silently dropped.
        let err =
            serde_json::from_str::<Source>(r#"{"type": "vector", "scheme": "tms"}"#).unwrap_err();
        assert!(err.to_string().contains("scheme"));
    }
}
