// High-level, ergonomic view of a Mapbox Vector Tile.
//
// `proto::*` types mirror the wire schema byte-for-byte; the types in
// this module trade fidelity for usability: dictionary-encoded tags are
// resolved into a `HashMap`, the multi-variant `Value` message is folded
// into a Rust enum, and command-stream geometry is materialized as `geo`
// objects via `super::geometry::decode_geometry`.

use std::collections::HashMap;

use super::error::{Error, Result};
use super::geometry::{Geometry, decode_geometry};
use super::proto;

// ==============================================================================
// Public types
// ==============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub name: String,
    pub version: u32,
    /// Tile-coordinate extent (e.g. 4096). Defaults per the spec when the
    /// proto omits the field.
    pub extent: u32,
    pub features: Vec<Feature>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Feature {
    pub id: Option<u64>,
    pub properties: HashMap<String, Value>,
    pub geometry: Geometry,
}

/// Decoded analogue of `proto::tile::Value`, plus a few richer variants
/// the MVT wire format doesn't support.
///
/// The first seven variants correspond 1:1 to MVT tag value kinds, and
/// the proto decoder produces exactly one variant per value (or fails
/// with [`Error::AmbiguousValue`]).
///
/// `Null`, `Array`, and `Object` are not representable in MVT and so
/// never appear in proto-decoded tiles. They exist because [`Feature`]
/// also serves as the abstract input to style-expression evaluation,
/// where the MapLibre spec requires richer values (e.g. for
/// `["literal", [1,2,3]]` or for fixture-fed feature properties).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Float(f32),
    Double(f64),
    Int(i64),
    Uint(u64),
    SInt(i64),
    Bool(bool),
    /// Absence of a value. Used by callers that build [`Feature`]
    /// instances directly — MVT itself represents missing data as the
    /// absence of a key in the properties map.
    Null,
    /// Heterogeneous list. Items follow the same enrichment rules as
    /// the enclosing value.
    Array(Vec<Value>),
    /// String-keyed map. Iteration order is unspecified.
    Object(HashMap<String, Value>),
}

// ==============================================================================
// Conversions from the proto layer
// ==============================================================================

impl TryFrom<&proto::Tile> for Tile {
    type Error = Error;

    fn try_from(tile: &proto::Tile) -> Result<Self> {
        let layers = tile
            .layers
            .iter()
            .map(Layer::try_from)
            .collect::<Result<_>>()?;
        Ok(Tile { layers })
    }
}

impl TryFrom<&proto::tile::Layer> for Layer {
    type Error = Error;

    // We resolve the layer-wide `values` dictionary up front so each
    // feature only does cheap index lookups. This also fails the whole
    // layer if any single `Value` is malformed, which matches the spec's
    // requirement that every value have exactly one populated variant.
    fn try_from(layer: &proto::tile::Layer) -> Result<Self> {
        let values: Vec<Value> = layer
            .values
            .iter()
            .map(Value::try_from)
            .collect::<Result<_>>()?;

        let features = layer
            .features
            .iter()
            .map(|f| Feature::from_proto(f, &layer.keys, &values))
            .collect::<Result<_>>()?;

        Ok(Layer {
            name: layer.name.clone(),
            version: layer.version,
            // Proto marks `extent` optional but the MVT spec says it is
            // required and prescribes 4096 as the convention; the
            // generated default lines up.
            extent: layer.extent.unwrap_or(4096),
            features,
        })
    }
}

impl Feature {
    fn from_proto(f: &proto::tile::Feature, keys: &[String], values: &[Value]) -> Result<Self> {
        let properties = decode_tags(&f.tags, keys, values)?;

        // An out-of-range `type` is treated as UNKNOWN, mirroring the
        // spec's forward-compatibility allowance for unrecognized geometry
        // types (§ 4.3.4.1).
        let geom_type = f
            .r#type
            .and_then(|t| proto::tile::GeomType::try_from(t).ok())
            .unwrap_or(proto::tile::GeomType::Unknown);
        let geometry = decode_geometry(geom_type, &f.geometry)?;

        Ok(Feature {
            id: f.id,
            properties,
            geometry,
        })
    }
}

impl TryFrom<&proto::tile::Value> for Value {
    type Error = Error;

    // A `proto::tile::Value` is essentially a manual union: seven
    // `optional` fields where exactly one MUST be populated. We walk
    // every field, count hits, and only accept the message when the
    // count is exactly one — both zero and two-or-more are pathological.
    fn try_from(v: &proto::tile::Value) -> Result<Self> {
        let mut chosen: Option<Value> = None;
        let mut populated = 0usize;
        let mut set = |val: Value| {
            populated += 1;
            chosen = Some(val);
        };

        if let Some(s) = &v.string_value {
            set(Value::String(s.clone()));
        }
        if let Some(f) = v.float_value {
            set(Value::Float(f));
        }
        if let Some(d) = v.double_value {
            set(Value::Double(d));
        }
        if let Some(i) = v.int_value {
            set(Value::Int(i));
        }
        if let Some(u) = v.uint_value {
            set(Value::Uint(u));
        }
        if let Some(s) = v.sint_value {
            set(Value::SInt(s));
        }
        if let Some(b) = v.bool_value {
            set(Value::Bool(b));
        }

        match populated {
            1 => Ok(chosen.expect("populated == 1 implies chosen is Some")),
            n => Err(Error::AmbiguousValue { populated: n }),
        }
    }
}

// ==============================================================================
// Tag → property-map decoding
// ==============================================================================

// Tags are flattened `[key_index, value_index, …]` pairs that reference
// the layer-wide `keys`/`values` dictionaries (spec § 4.4). The spec
// requires (a) an even count, (b) every index within range, and (c)
// unique keys within a single feature. We enforce (a) and (b) explicitly;
// (c) falls out of inserting into a HashMap, which means a duplicate key
// would just overwrite the earlier value rather than fail loudly — that's
// a deliberate trade-off so partially malformed real-world tiles still
// decode usefully.
fn decode_tags(raw: &[u32], keys: &[String], values: &[Value]) -> Result<HashMap<String, Value>> {
    if !raw.len().is_multiple_of(2) {
        return Err(Error::OddTags { count: raw.len() });
    }
    let mut out = HashMap::with_capacity(raw.len() / 2);
    for pair in raw.as_chunks::<2>().0 {
        let (k_idx, v_idx) = (pair[0], pair[1]);
        let key = keys
            .get(k_idx as usize)
            .ok_or(Error::TagOutOfRange {
                kind: "key",
                idx: k_idx,
                len: keys.len(),
            })?
            .clone();
        let val = values
            .get(v_idx as usize)
            .ok_or(Error::TagOutOfRange {
                kind: "value",
                idx: v_idx,
                len: values.len(),
            })?
            .clone();
        out.insert(key, val);
    }
    Ok(out)
}

// ==============================================================================
// Tests
// ==============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn proto_value_string(s: &str) -> proto::tile::Value {
        proto::tile::Value {
            string_value: Some(s.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn value_accepts_single_populated_variant() {
        let pv = proto_value_string("hi");
        assert_eq!(Value::try_from(&pv).unwrap(), Value::String("hi".into()));
    }

    #[test]
    fn value_rejects_no_variant() {
        let pv = proto::tile::Value::default();
        assert_eq!(
            Value::try_from(&pv),
            Err(Error::AmbiguousValue { populated: 0 })
        );
    }

    #[test]
    fn value_rejects_multiple_variants() {
        let pv = proto::tile::Value {
            string_value: Some("a".into()),
            int_value: Some(1),
            ..Default::default()
        };
        assert_eq!(
            Value::try_from(&pv),
            Err(Error::AmbiguousValue { populated: 2 })
        );
    }

    #[test]
    fn tags_resolve_against_dictionaries() {
        let keys = vec!["name".to_string(), "rank".to_string()];
        let values = vec![Value::String("Monaco".into()), Value::Int(7)];
        let raw = vec![0, 0, 1, 1];
        let out = decode_tags(&raw, &keys, &values).unwrap();
        assert_eq!(out.get("name"), Some(&Value::String("Monaco".into())));
        assert_eq!(out.get("rank"), Some(&Value::Int(7)));
    }

    #[test]
    fn tags_reject_odd_count() {
        assert_eq!(
            decode_tags(&[0, 0, 1], &["k".into()], &[Value::Int(0)]),
            Err(Error::OddTags { count: 3 })
        );
    }

    #[test]
    fn tags_reject_out_of_range_key() {
        let err = decode_tags(&[5, 0], &["k".into()], &[Value::Int(0)]).unwrap_err();
        assert_eq!(
            err,
            Error::TagOutOfRange {
                kind: "key",
                idx: 5,
                len: 1
            }
        );
    }

    #[test]
    fn tags_reject_out_of_range_value() {
        let err = decode_tags(&[0, 9], &["k".into()], &[Value::Int(0)]).unwrap_err();
        assert_eq!(
            err,
            Error::TagOutOfRange {
                kind: "value",
                idx: 9,
                len: 1
            }
        );
    }

    #[test]
    fn layer_with_no_features_round_trips() {
        let pl = proto::tile::Layer {
            version: 2,
            name: "empty".into(),
            extent: Some(4096),
            keys: vec![],
            values: vec![],
            features: vec![],
        };
        let layer = Layer::try_from(&pl).unwrap();
        assert_eq!(layer.name, "empty");
        assert_eq!(layer.version, 2);
        assert_eq!(layer.extent, 4096);
        assert!(layer.features.is_empty());
    }
}
