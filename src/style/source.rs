// A style document's `sources` map names where each layer's data lives.
// The MapLibre spec defines several source kinds; this models `vector`,
// `raster` and `geojson` with their common properties, and parses every
// other kind (`raster-dem`, `image`, `video`) as `Source::Other`.
//
// `MapView` doesn't read these yet: every layer draws from the one
// `TileJsonSource` it was given. They are parsed so that styles load, and
// so a later multi-source renderer has them.

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
// Properties the spec defines but this enum doesn't model (`scheme`,
// `promoteId`, `volatile`, ...) are ignored rather than rejected, so any
// style MapLibre accepts parses.
#[serde(tag = "type")]
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
        #[serde(default, rename = "tileSize")]
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

    /// Any other source type (`raster-dem`, `image`, `video`). Its
    /// properties are dropped; layers drawing from it are not rendered.
    #[serde(other)]
    Other,
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
    fn raster_tile_size_is_camel_case() {
        // OpenFreeMap's styles carry this Natural Earth relief source.
        let src: Source = serde_json::from_str(
            r#"{
                "type": "raster",
                "maxzoom": 6,
                "tileSize": 256,
                "tiles": ["https://tiles.openfreemap.org/natural_earth/ne2sr/{z}/{x}/{y}.png"]
            }"#,
        )
        .unwrap();
        assert!(matches!(src, Source::Raster { tile_size: Some(256), maxzoom: Some(6), .. }));
    }

    #[test]
    fn spec_properties_not_modelled_are_ignored() {
        let src: Source = serde_json::from_str(
            r#"{
                "type": "vector",
                "url": "https://example.com/tiles.json",
                "scheme": "xyz",
                "promoteId": "osm_id",
                "volatile": false,
                "encoding": "mvt"
            }"#,
        )
        .unwrap();
        assert!(matches!(src, Source::Vector { url: Some(_), .. }));
    }

    #[test]
    fn other_source_types_parse_as_other() {
        let src: Source = serde_json::from_str(
            r#"{"type": "raster-dem", "url": "https://example.com/terrain.json", "encoding": "terrarium"}"#,
        )
        .unwrap();
        assert_eq!(src, Source::Other);
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
}
