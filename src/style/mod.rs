// MapLibre style document parser, narrowly scoped to what
// `minimal_style.json` exercises. See the project plan for the full
// scope rationale; the short version is that this module models
// exactly the subset of the spec we need today and rejects anything
// else, with the AST shaped so future spec coverage is purely
// additive.
//
// Public entry points:
//
// * [`Style::from_json_str`] / [`Style::from_json_slice`] parse a
//   serialized style document into a typed [`Style`] tree, surfacing
//   any unsupported field or operator as a [`StyleError`].
// * [`expression::Expr::eval`] / [`expression::Filter::matches`] and
//   [`layer::Property::resolve`] evaluate the parsed AST against a
//   given feature and zoom level.

pub mod color;
pub mod expression;
pub mod layer;
pub mod legacy;
pub mod source;
pub mod span;

use std::collections::HashMap;

use serde::Deserialize;

pub use color::{Color, ColorError};
pub use expression::{EvalContext, EvalError, Expr, ExprError, Filter, Value};
pub use span::{ByteSpan, JsonPath, PathSegment, SpanMap};
pub use layer::{
    BackgroundLayer, BackgroundPaint, FillLayer, FillPaint, IconAnchor, Layer, LayerKind, LineCap,
    LineJoin, LineLayer, LineLayout, LinePaint, Property, SymbolLayer, SymbolLayout, SymbolPaint,
    SymbolPlacement, TextAnchor, TextJustify, TextTransform, TextWritingMode, TranslateAnchor,
    Visibility,
};
pub use source::Source;

/// A parsed MapLibre style document. The fields mirror the top-level
/// keys of the JSON file. Spec keys we recognize but don't render
/// yet (`metadata`, `light`, `terrain`, `bearing`, `pitch`,
/// `center`, `zoom`) are accepted and round-tripped as raw JSON so a
/// user can load a real-world style document without first hand-
/// trimming it.
///
/// `deny_unknown_fields` is **not** used: real-world MapLibre style
/// documents carry vendor-specific top-level fields (`id`, `owner`,
/// `created`, `modified`, …) that aren't in the spec. Capturing them
/// all explicitly is high-maintenance churn for no rendering benefit,
/// so the parser is lenient at this level. Misspelled spec fields
/// will still surface when the renderer tries to consume them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Style {
    /// MapLibre style spec version. We only accept `8`; any other
    /// number is rejected post-deserialize.
    pub version: u8,
    /// Display name for the style.
    pub name: Option<String>,
    /// Template URL for glyph PBFs. We don't fetch them here, just
    /// store the string for downstream rendering code.
    pub glyphs: Option<String>,
    /// Sprite URL prefix. The runtime fetches `{sprite}.json` and
    /// `{sprite}.png` from this base. Phase 5: the spec also allows
    /// an array of `{ id, url }` entries for multi-sprite sources,
    /// but every real-world style we've seen uses the single-string
    /// form. The custom deserializer below accepts the array shape,
    /// emits a [`StyleWarning::MultiSpriteUnsupported`], and keeps
    /// the first entry's `url` — matching what a degraded MapLibre
    /// client would visibly do.
    #[serde(default, deserialize_with = "deserialize_sprite")]
    pub sprite: Option<String>,
    /// Free-form metadata. Stored opaquely.
    pub metadata: Option<serde_json::Value>,
    /// Default-view hints. Renderers may consume these to seed the
    /// camera; we don't here.
    #[serde(default)]
    pub center: Option<serde_json::Value>,
    #[serde(default)]
    pub zoom: Option<f64>,
    #[serde(default)]
    pub bearing: Option<f64>,
    #[serde(default)]
    pub pitch: Option<f64>,
    /// Light and terrain blocks are accepted but ignored.
    #[serde(default)]
    pub light: Option<serde_json::Value>,
    #[serde(default)]
    pub terrain: Option<serde_json::Value>,
    /// Named sources keyed by the id used in each layer's `source`
    /// field.
    pub sources: HashMap<String, Source>,
    /// Ordered list of layers. Render order is source-order (first
    /// layer drawn first / on the bottom).
    pub layers: Vec<Layer>,
}

impl Style {
    /// Parse from a JSON string. Returns [`StyleError::UnsupportedVersion`]
    /// if the document's `version` is not `8`.
    ///
    /// Use [`Style::parse_with_warnings`] to also receive the list of
    /// per-document warnings about ignored / unsupported features.
    pub fn from_json_str(s: &str) -> Result<Self, StyleError> {
        let (style, _) = Self::parse_with_warnings(s)?;
        Ok(style)
    }

    /// Parse from a JSON byte slice. Same validation as
    /// [`Style::from_json_str`].
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, StyleError> {
        let s = std::str::from_utf8(bytes).map_err(StyleError::Utf8)?;
        Self::from_json_str(s)
    }

    /// Parse and also return a list of [`StyleWarning`]s describing
    /// fields / features the parser recognised but does not honour
    /// at render time (vendor-extension top-level keys, fill-layer
    /// `layout` blocks, legacy `{stops: …}` paint functions, raster
    /// sources, …).
    ///
    /// Callers that want to surface these to the user (CLI viewers,
    /// editors) should iterate the returned `Vec`; callers that
    /// don't care can keep using [`Style::from_json_str`].
    pub fn parse_with_warnings(s: &str) -> Result<(Self, Vec<StyleWarning>), StyleError> {
        // Walk the raw JSON first so the warning detection sees the
        // verbatim document (after serde-typed parsing we've already
        // discarded the unrecognised bits). The typed parse then
        // happens through serde as normal.
        let mut raw: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| StyleError::from_serde_json(e, s))?;
        let mut warnings = Vec::new();
        collect_warnings(&raw, &mut warnings);

        // Drop `fill-extrusion` layers before the typed parse: the
        // renderer has no 3D geometry pipeline, and rejecting the whole
        // document over a layer we couldn't draw anyway makes real-world
        // styles unusable. `collect_warnings` has already surfaced the
        // `UnsupportedLayerType` notice; stripping now keeps any
        // unrecognised paint/layout keys on the layer out of the typed
        // deserialize path and out of the expression pre-pass.
        strip_unsupported_layers(&mut raw);

        // Run the path-aware expression pre-pass before serde takes
        // over. If any expression in the document is malformed, we
        // surface the located error with the SpanMap built from `s`
        // so the renderer can resolve `path` → byte span.
        if let Err(expr_err) = validate_expressions_in_raw(&raw) {
            let map = span::build_span_map(s, &raw);
            return Err(StyleError::expression(expr_err, &map, s));
        }

        let style: Style =
            serde_json::from_value(raw.clone()).map_err(|e| {
                // Serde failures after the expression pre-pass are
                // structural (unsupported layer type, bad color
                // literal, unknown enum variant, …). They land here
                // with no JsonPath, but `serde_json::Error::line/column`
                // does give us a byte position into the source.
                StyleError::from_serde_json(e, s)
            })?;
        let style = Self::validate(style, &raw, s)?;
        Ok((style, warnings))
    }

    fn validate(style: Style, raw: &serde_json::Value, src: &str) -> Result<Self, StyleError> {
        if style.version != 8 {
            // Point the snippet at the `"version": N` value in the
            // source so the user can see what they had set.
            let map = span::build_span_map(src, raw);
            let span = map
                .lookup(&span::JsonPath::root().child_key("version"))
                .unwrap_or(ByteSpan::empty());
            return Err(StyleError::UnsupportedVersion {
                value: style.version,
                span,
            });
        }
        Ok(style)
    }
}

// ==============================================================================
// Style warnings
// ==============================================================================

/// A non-fatal observation about a parsed style document — a field
/// or feature we recognised but won't honour at render time. Emit
/// these (via `eprintln!` or a logger) so users can tell when a
/// stylesheet expects something the renderer hasn't implemented yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StyleWarning {
    /// Top-level key that isn't part of the modeled spec subset.
    /// Common in real-world stylesheets (`id`, `owner`, `created`,
    /// `modified`, vendor blobs).
    UnknownTopLevelField(String),
    /// Source whose `type` isn't `"vector"` — the renderer can't
    /// draw it (raster, raster-dem, geojson, …).
    UnsupportedSourceType {
        source_id: String,
        source_type: String,
    },
    /// Paint / layout keys we don't model for this layer kind.
    /// Includes spec-standard keys we just haven't wired up
    /// (`visibility`, `fill-sort-key`) as well as vendor extensions.
    IgnoredLayerKeys {
        layer_id: String,
        layer_type: String,
        /// `"paint"` or `"layout"` — which block the keys came from.
        block: &'static str,
        keys: Vec<String>,
    },
    /// Pre-spec-v8 `{ "stops": [...] }` style function. We rewrite
    /// these into modern expressions at parse time; emitting the
    /// warning lets stylesheet maintainers know they should consider
    /// migrating to `["interpolate", …]` / `["step", …]`.
    LegacyFunctionConverted {
        layer_id: String,
        property: String,
    },
    /// A symbol layer's `text-field` is a string literal that
    /// contains a legacy `{…}` token but isn't a clean single-token
    /// reference (e.g. `"the name is {name}"`). We support only the
    /// single-token shape (`"{name}"` → `["get", "name"]`); anything
    /// more elaborate stays as literal text. Authors should use the
    /// modern `["concat", …]` expression for mixed strings.
    MultiTokenTextField {
        layer_id: String,
        value: String,
    },
    /// `style.sprite` was authored as an array of `{ id, url }`
    /// entries (the spec-legal multi-sprite form). We honour only the
    /// first entry's `url`; downstream sprite lookups won't know
    /// about the trailing entries' namespaced names. Authors who need
    /// every sprite source should consolidate them into a single
    /// atlas. See the Phase 5 SESSION.md follow-up for the multi-
    /// sprite plumbing roadmap.
    MultiSpriteUnsupported,
    /// Layer of a spec-defined kind that this renderer does not
    /// implement (currently only `fill-extrusion`). The parser drops
    /// the layer before serde sees it so the rest of the document can
    /// still load; downstream code never observes one of these.
    UnsupportedLayerType {
        layer_id: String,
        layer_type: String,
    },
}

impl std::fmt::Display for StyleWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StyleWarning::UnknownTopLevelField(name) => write!(
                f,
                "ignored unknown top-level field {name:?}"
            ),
            StyleWarning::UnsupportedSourceType {
                source_id,
                source_type,
            } => write!(
                f,
                "source {source_id:?}: type {source_type:?} is parsed but not rendered"
            ),
            StyleWarning::IgnoredLayerKeys {
                layer_id,
                layer_type,
                block,
                keys,
            } => write!(
                f,
                "layer {layer_id:?} ({layer_type}): {block} keys not honored: {keys:?}"
            ),
            StyleWarning::LegacyFunctionConverted { layer_id, property } => write!(
                f,
                "layer {layer_id:?}: paint property {property:?} uses pre-v8 `stops` form — auto-converted to a modern expression"
            ),
            StyleWarning::MultiTokenTextField { layer_id, value } => write!(
                f,
                "layer {layer_id:?}: text-field {value:?} mixes literal text with legacy {{token}} substitution; only single-token strings like \"{{name}}\" are honored — use [\"concat\", …] for mixed strings"
            ),
            StyleWarning::MultiSpriteUnsupported => write!(
                f,
                "style.sprite is an array of multi-sprite entries; only the first entry's `url` is used (multi-sprite namespacing is not implemented yet)"
            ),
            StyleWarning::UnsupportedLayerType {
                layer_id,
                layer_type,
            } => write!(
                f,
                "layer {layer_id:?}: type {layer_type:?} is not supported by this renderer; layer was dropped"
            ),
        }
    }
}

// Custom deserializer for `style.sprite`. The MapLibre spec lets this
// field be either a plain string (the canonical, near-universal form)
// or an array of `{ id, url }` entries (multi-sprite, rare). We accept
// both shapes here so real-world documents load; the multi-sprite case
// keeps only the first entry's `url` and surfaces a
// [`StyleWarning::MultiSpriteUnsupported`] during the raw-JSON warning
// pass so authors know we collapsed the array.
fn deserialize_sprite<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<serde_json::Value>::deserialize(d)?;
    let Some(raw) = raw else { return Ok(None) };
    match raw {
        serde_json::Value::String(s) => Ok(Some(s)),
        serde_json::Value::Array(entries) => {
            // Pick the first `{url: "..."}` we see. The matching
            // warning is emitted by `collect_warnings` so the typed
            // path can't accidentally swallow it.
            for entry in entries {
                if let Some(url) = entry.get("url").and_then(|v| v.as_str()) {
                    return Ok(Some(url.to_string()));
                }
            }
            // Empty array or entries without `url`: treat as "no
            // sprite" rather than erroring — the array form is rare
            // enough that being strict here would refuse valid (if
            // useless) documents.
            Ok(None)
        }
        serde_json::Value::Null => Ok(None),
        other => Err(serde::de::Error::custom(format!(
            "style.sprite must be a string or array, got {other}"
        ))),
    }
}

// ==============================================================================
// Path-aware expression pre-pass
// ==============================================================================
//
// `serde_json::from_value::<Style>` runs expression parsing inside its
// `deserialize_with` helpers (see `style::layer::prop_de::*`). Any
// `ExprError` raised there gets string-wrapped by `serde::de::Error::custom`,
// which loses the variant identity and any JsonPath. We can't avoid that
// without rewriting the deserializer.
//
// Instead, before handing the raw JSON to serde, we run the same
// expression checks ourselves — walking every layer's filter, paint,
// and layout JSON and trying `Expr::from_json_at` with the appropriate
// `JsonPath`. If anything is malformed, the path-aware error fires here
// and surfaces to the caller via the new `StyleError` variants. If
// every expression is well-formed, control passes to serde for the
// structural deserialize, which will then succeed on the expression
// side and only ever raise structural / literal errors.
//
// The cost is one extra walk of the same expressions on the happy
// path — they get parsed once here (and discarded) and once by serde.
// Style documents are small (hundreds of KB at most) so the duplicate
// work is negligible next to the rendering pipeline that follows.

/// Property keys whose serde deserializer routes through
/// `property_from_json` — i.e. the keys that can carry either a
/// literal of their concrete type or an expression. We pre-validate
/// only these so list-shaped properties (`text-font`, `line-dasharray`)
/// and numeric-tuple properties (`text-offset`, `fill-translate`) —
/// whose value happens to be a JSON array — aren't misread as
/// expression calls.
///
/// Kept in lockstep with the `#[serde(deserialize_with = "prop_de::opt_*")]`
/// attributes in [`super::layer`]. New property-shaped keys land here
/// when their field gains an `opt_color` / `opt_f32` / `opt_string` /
/// `opt_text_field` deserializer.
const PROPERTY_STYLE_KEYS: &[&str] = &[
    // background
    "background-color",
    "background-opacity",
    "background-pattern",
    // fill
    "fill-color",
    "fill-opacity",
    "fill-pattern",
    // line paint
    "line-color",
    "line-gap-width",
    "line-miter-limit",
    "line-offset",
    "line-opacity",
    "line-pattern",
    "line-round-limit",
    "line-width",
    // symbol paint
    "icon-color",
    "icon-halo-blur",
    "icon-halo-color",
    "icon-halo-width",
    "icon-opacity",
    "text-color",
    "text-halo-blur",
    "text-halo-color",
    "text-halo-width",
    "text-opacity",
    // symbol layout
    "icon-rotate",
    "icon-size",
    "symbol-sort-key",
    "symbol-spacing",
    "text-field",
    "text-letter-spacing",
    "text-line-height",
    "text-max-angle",
    "text-max-width",
    "text-radial-offset",
    "text-size",
];

fn validate_expressions_in_raw(raw: &serde_json::Value) -> Result<(), expression::ExprError> {
    let Some(layers) = raw.get("layers").and_then(|v| v.as_array()) else {
        return Ok(());
    };
    let base = span::JsonPath::root().child_key("layers");
    for (i, layer) in layers.iter().enumerate() {
        let layer_path = base.child_index(i);
        let Some(obj) = layer.as_object() else { continue };

        if let Some(filter) = obj.get("filter") {
            let converted = if legacy::is_legacy_filter(filter) {
                legacy::convert_legacy_filter(filter)
            } else {
                filter.clone()
            };
            expression::Expr::from_filter_json_at(
                &converted,
                layer_path.child_key("filter"),
            )?;
        }

        let layer_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        for block in ["paint", "layout"] {
            let Some(props) = obj.get(block).and_then(|v| v.as_object()) else {
                continue;
            };
            for (key, value) in props {
                // Skip properties whose serde helper isn't the
                // `property_from_json` family — their JSON-array shape
                // is a literal list, not an expression call.
                if !PROPERTY_STYLE_KEYS.contains(&key.as_str()) {
                    continue;
                }
                let prop_path = layer_path.child_key(block).child_key(key.clone());
                if layer::is_expression_shape(value) {
                    expression::Expr::from_json_at(value, prop_path)?;
                } else if legacy::is_legacy_function(value) {
                    // The converter parses the rewritten modern form
                    // via `Expr::from_json`. Any error fires inside
                    // that call and surfaces with no path attached;
                    // wrap it here with the original property's path
                    // so the source snippet still points at the
                    // legacy-shape value the user wrote.
                    let spec = legacy_spec_for(layer_type, key);
                    legacy::convert_legacy_function(value, spec.as_ref())
                        .map(drop)
                        .map_err(|e| e.with_path(prop_path))?;
                }
            }
        }
    }
    Ok(())
}

fn legacy_spec_for(layer_type: &str, prop: &str) -> Option<serde_json::Value> {
    // Spec hints used to gate the legacy-function rewriter. For
    // `text-field` we flip on `tokens: true` so `{prop}` placeholders
    // in stop outputs convert to `["to-string", ["get", "prop"]]`.
    // For other properties the type hint alone is enough; the values
    // here mirror the synthetic specs `prop_de` constructs.
    match (layer_type, prop) {
        ("symbol", "text-field") => Some(serde_json::json!({
            "type": "string", "tokens": true
        })),
        _ => None,
    }
}

// ==============================================================================
// Warning collection — walks raw JSON to find ignored fields
// ==============================================================================

/// Top-level keys the parser models. Anything else lands in a
/// `UnknownTopLevelField` warning. Kept in sync with [`Style`]'s
/// `serde` fields.
const STYLE_TOP_KEYS: &[&str] = &[
    "version",
    "name",
    "glyphs",
    "sprite",
    "metadata",
    "center",
    "zoom",
    "bearing",
    "pitch",
    "light",
    "terrain",
    "sources",
    "layers",
];

/// Layer-level keys honoured for every kind. The paint / layout
/// blocks themselves get walked separately against per-kind tables.
const LAYER_COMMON_KEYS: &[&str] = &[
    "id",
    "type",
    "source",
    "source-layer",
    "minzoom",
    "maxzoom",
    "filter",
    "paint",
    "layout",
];

fn honored_paint_keys(layer_type: &str) -> &'static [&'static str] {
    match layer_type {
        "background" => &[
            "background-color",
            "background-opacity",
            // Phase 8: `background-pattern` is rendered via the
            // pattern-fill pipeline drawing a full-surface quad after
            // the clear. See `MapView::paint` for the dispatch.
            "background-pattern",
        ],
        "fill" => &[
            "fill-color",
            "fill-opacity",
            // Phase 8: `fill-pattern` routes to the pattern-fill
            // pipeline; `fill-translate` / `-anchor` apply per-layer
            // per-frame offsets. `fill-antialias` is parse-only —
            // see the TODO breadcrumb on `FillPaint`.
            "fill-pattern",
            "fill-translate",
            "fill-translate-anchor",
            "fill-antialias",
        ],
        "line" => &[
            "line-color",
            "line-width",
            "line-opacity",
            "line-miter-limit",
            "line-round-limit",
            "line-dasharray",
            "line-gap-width",
            "line-offset",
            // Phase 8: `line-pattern` is parse-only (see TODO on
            // `LinePaint`). `line-translate` / `-anchor` apply
            // per-layer per-frame offsets via the same uniform
            // additions the fill side uses.
            "line-pattern",
            "line-translate",
            "line-translate-anchor",
        ],
        "symbol" => &[
            "text-color",
            "text-opacity",
            "icon-opacity",
            // Phase 4 paint: halo properties — `text-halo-blur` is
            // parse-only because we don't run an SDF pipeline, but
            // it's still accepted so styles load cleanly.
            "text-halo-color",
            "text-halo-width",
            "text-halo-blur",
            // Phase 5 paint: icon tint + halo (halo parse-only —
            // see the TODO breadcrumb on `SymbolPaint`).
            "icon-color",
            "icon-halo-color",
            "icon-halo-width",
            "icon-halo-blur",
        ],
        _ => &[],
    }
}

fn honored_layout_keys(layer_type: &str) -> &'static [&'static str] {
    // `visibility` is universal — every kind accepts it and the
    // renderer honours it. Per-kind additions stack on top.
    match layer_type {
        "line" => &["visibility", "line-cap", "line-join"],
        "symbol" => &[
            "visibility",
            "text-field",
            "text-font",
            "text-size",
            // Phase 4 layout. `text-padding` is parse-only (collision
            // is Phase 7); `text-radial-offset` is parse-only because
            // it conflicts with `text-offset` per spec and our renderer
            // honours offset. The rest are wired through to the shaper.
            "text-anchor",
            "text-offset",
            "text-radial-offset",
            "text-max-width",
            "text-letter-spacing",
            "text-line-height",
            "text-transform",
            "text-justify",
            "text-padding",
            // Phase 5 layout (icon-*). All consumed except `icon-padding`,
            // which is parse-only until Phase 7's collision pass.
            "icon-image",
            "icon-size",
            "icon-rotate",
            "icon-anchor",
            "icon-offset",
            "icon-padding",
            // Phase 6 placement. `symbol-placement` flips to the line-
            // following path; `symbol-spacing`, `text-max-angle`, and
            // `text-keep-upright` tune that pipeline; `text-writing-mode`
            // is parse-only (vertical CJK is Phase 13+).
            "symbol-placement",
            "symbol-spacing",
            "text-max-angle",
            "text-keep-upright",
            "text-writing-mode",
            // Phase 7 collision. Consumed by the per-frame collision
            // pass in `render::mesh::collision`; `symbol-avoid-edges`
            // is the coarse "drop labels near the map edge" heuristic
            // documented on the `SymbolLayout` field.
            "symbol-sort-key",
            "symbol-avoid-edges",
            "text-allow-overlap",
            "text-ignore-placement",
            "text-optional",
            "icon-allow-overlap",
            "icon-ignore-placement",
            "icon-optional",
        ],
        // background / fill / unknown kinds: visibility only.
        _ => &["visibility"],
    }
}

fn collect_warnings(v: &serde_json::Value, out: &mut Vec<StyleWarning>) {
    let Some(obj) = v.as_object() else { return };

    for key in obj.keys() {
        if !STYLE_TOP_KEYS.iter().any(|k| *k == key) {
            out.push(StyleWarning::UnknownTopLevelField(key.clone()));
        }
    }

    if let Some(sources) = obj.get("sources").and_then(|v| v.as_object()) {
        for (id, src) in sources {
            let Some(kind) = src.get("type").and_then(|v| v.as_str()) else {
                continue;
            };
            if !matches!(kind, "vector" | "geojson") {
                out.push(StyleWarning::UnsupportedSourceType {
                    source_id: id.clone(),
                    source_type: kind.to_string(),
                });
            }
        }
    }

    // Surface the multi-sprite array shape *during the raw walk*: by
    // the time the typed `Style` lands, the deserializer has already
    // collapsed the array to its first `url`. Catching the shape here
    // is what keeps the warning in sync with the lossy fallback above.
    if matches!(obj.get("sprite"), Some(serde_json::Value::Array(_))) {
        out.push(StyleWarning::MultiSpriteUnsupported);
    }

    if let Some(layers) = obj.get("layers").and_then(|v| v.as_array()) {
        for layer in layers {
            collect_layer_warnings(layer, out);
        }
    }
}

/// Layer kinds the parser recognises as legal MapLibre spec values but
/// silently drops because the renderer can't draw them. Keeping the list
/// here (rather than scattered match arms) makes it easy to extend if
/// the renderer ever gains 3D / raster pipelines.
const UNSUPPORTED_LAYER_TYPES: &[&str] = &["fill-extrusion"];

fn is_unsupported_layer_type(t: &str) -> bool {
    UNSUPPORTED_LAYER_TYPES.contains(&t)
}

/// Strip layers whose `type` is in [`UNSUPPORTED_LAYER_TYPES`] from the
/// raw style JSON. Called between `collect_warnings` (which records the
/// dropped layers as `StyleWarning::UnsupportedLayerType`) and the typed
/// serde parse, so downstream code sees a document the renderer can
/// actually handle.
fn strip_unsupported_layers(raw: &mut serde_json::Value) {
    let Some(layers) = raw
        .as_object_mut()
        .and_then(|o| o.get_mut("layers"))
        .and_then(|v| v.as_array_mut())
    else {
        return;
    };
    layers.retain(|layer| {
        let kind = layer
            .as_object()
            .and_then(|o| o.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        !is_unsupported_layer_type(kind)
    });
}

fn collect_layer_warnings(layer: &serde_json::Value, out: &mut Vec<StyleWarning>) {
    let Some(obj) = layer.as_object() else { return };
    let layer_id = obj
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("<unnamed>")
        .to_string();
    let layer_type = obj
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // For layer kinds we intentionally drop (e.g. `fill-extrusion`),
    // surface a single warning naming the layer and stop — walking the
    // paint/layout blocks would emit "ignored key" warnings for every
    // property on a layer we already chose to discard wholesale.
    if is_unsupported_layer_type(&layer_type) {
        out.push(StyleWarning::UnsupportedLayerType {
            layer_id,
            layer_type,
        });
        return;
    }

    // Unrecognised top-level layer keys (rare; spec keeps this set
    // tight). Avoid double-counting `paint` / `layout` keys —
    // they're walked below.
    let unknown_common: Vec<String> = obj
        .keys()
        .filter(|k| !LAYER_COMMON_KEYS.contains(&k.as_str()))
        .cloned()
        .collect();
    if !unknown_common.is_empty() {
        out.push(StyleWarning::IgnoredLayerKeys {
            layer_id: layer_id.clone(),
            layer_type: layer_type.clone(),
            block: "root",
            keys: unknown_common,
        });
    }

    if let Some(paint) = obj.get("paint").and_then(|v| v.as_object()) {
        let honored = honored_paint_keys(&layer_type);
        let ignored: Vec<String> = paint
            .keys()
            .filter(|k| !honored.contains(&k.as_str()))
            .cloned()
            .collect();
        if !ignored.is_empty() {
            out.push(StyleWarning::IgnoredLayerKeys {
                layer_id: layer_id.clone(),
                layer_type: layer_type.clone(),
                block: "paint",
                keys: ignored,
            });
        }
        // Surface legacy `{stops: …}` paint values so authors know we
        // rewrote them. Only flag honored properties — ignored ones
        // already drew their own warning above.
        for (prop, value) in paint {
            if !honored.contains(&prop.as_str()) {
                continue;
            }
            if legacy::is_legacy_function(value) {
                out.push(StyleWarning::LegacyFunctionConverted {
                    layer_id: layer_id.clone(),
                    property: prop.clone(),
                });
            }
        }
    }

    if let Some(layout) = obj.get("layout").and_then(|v| v.as_object()) {
        let honored = honored_layout_keys(&layer_type);
        let ignored: Vec<String> = layout
            .keys()
            .filter(|k| !honored.contains(&k.as_str()))
            .cloned()
            .collect();
        if !ignored.is_empty() {
            out.push(StyleWarning::IgnoredLayerKeys {
                layer_id: layer_id.clone(),
                layer_type: layer_type.clone(),
                block: "layout",
                keys: ignored,
            });
        }

        // Legacy `{token}` substitution: surface mixed-text strings
        // that we won't honor. Only symbol layers carry `text-field`,
        // so the check is restricted to them — checking other kinds
        // would be wasted work and could match unrelated string
        // values in the future.
        if layer_type == "symbol"
            && let Some(tf) = layout.get("text-field").and_then(|v| v.as_str())
            && matches!(layer::classify_token_string(tf), layer::TokenForm::Mixed)
        {
            out.push(StyleWarning::MultiTokenTextField {
                layer_id,
                value: tf.to_string(),
            });
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StyleError {
    /// Top-level JSON syntax error or post-pre-pass structural error
    /// from serde (unknown layer type, bad color literal, …). Carries
    /// the 1-based `line`/`column` `serde_json::Error` reported plus a
    /// derived byte span pointing at that position in the source.
    #[error("{message} (at line {line}, column {column})")]
    Json {
        line: usize,
        column: usize,
        message: String,
        span: ByteSpan,
    },
    /// Expression parse failure with the JSON path of the offending
    /// expression and the resolved byte span. The inner `ExprError` is
    /// the original variant (already peeled out of `Located` so callers
    /// can pattern-match without re-unwrapping).
    #[error("expression error at {path}: {source}")]
    Expression {
        path: JsonPath,
        span: ByteSpan,
        #[source]
        source: ExprError,
    },
    #[error("style document is not valid UTF-8: {0}")]
    Utf8(std::str::Utf8Error),
    #[error("unsupported style version: {value} (only 8 is supported)")]
    UnsupportedVersion { value: u8, span: ByteSpan },
}

impl StyleError {
    /// Byte span of the offending fragment in the source, if known.
    /// Examples wrap this into a `miette::SourceSpan` for snippet
    /// rendering; library callers can ignore it.
    pub fn span(&self) -> Option<ByteSpan> {
        match self {
            StyleError::Json { span, .. } => Some(*span),
            StyleError::Expression { span, .. } => Some(*span),
            StyleError::UnsupportedVersion { span, .. } => Some(*span),
            StyleError::Utf8(_) => None,
        }
    }

    /// JSON path of the failing fragment, if known. Currently only
    /// `Expression` errors carry a path.
    pub fn path(&self) -> Option<&JsonPath> {
        match self {
            StyleError::Expression { path, .. } => Some(path),
            _ => None,
        }
    }

    pub(crate) fn from_serde_json(err: serde_json::Error, src: &str) -> Self {
        let line = err.line();
        let column = err.column();
        let offset = span::byte_offset_from_line_col(src, line, column);
        // serde_json typically reports the position of the *next*
        // character after the failure point; a one-byte span there is
        // a sensible single-caret highlight.
        let end = offset.saturating_add(1).min(src.len());
        StyleError::Json {
            line,
            column,
            message: err.to_string(),
            span: ByteSpan::new(offset, end),
        }
    }

    pub(crate) fn expression(err: ExprError, map: &SpanMap, _src: &str) -> Self {
        // Peel any `Located` wrapper so the `source` field exposes the
        // original variant directly. If the path is missing — possible
        // for paths produced by callers other than [`Expr::from_json_at`]
        // — fall back to the document root.
        let (path, inner) = match err {
            ExprError::Located { path, source } => (path, *source),
            other => (JsonPath::root(), other),
        };
        let span = map.lookup_or_ancestor(&path).unwrap_or(ByteSpan::empty());
        StyleError::Expression {
            path,
            span,
            source: inner,
        }
    }
}

impl From<serde_json::Error> for StyleError {
    fn from(err: serde_json::Error) -> Self {
        // Preserved for the legacy call sites that build a StyleError
        // without the original source text. The span field defaults to
        // empty — callers that *have* the source should prefer
        // [`StyleError::from_serde_json`].
        StyleError::Json {
            line: err.line(),
            column: err.column(),
            message: err.to_string(),
            span: ByteSpan::empty(),
        }
    }
}

// ==============================================================================
// Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::mvt::Feature;
    use crate::mvt::Geometry;
    use crate::mvt::Value as MvtValue;

    // The fixture lives at the repo root. `include_str!` resolves
    // relative to this source file, so we walk up two levels.
    const MINIMAL_STYLE: &str = include_str!("../../tests/fixtures/minimal_style.json");

    fn highway_feature() -> Feature {
        let mut props = HashMap::new();
        props.insert("kind".into(), MvtValue::String("highway".into()));
        props.insert("name".into(), MvtValue::String("Main St".into()));
        Feature {
            id: None,
            properties: props,
            geometry: Geometry::Unknown,
        }
    }

    #[test]
    fn parses_minimal_style_fixture() {
        let style = Style::from_json_str(MINIMAL_STYLE).expect("parses");
        assert_eq!(style.version, 8);
        assert_eq!(style.name.as_deref(), Some("My Style"));
        assert_eq!(
            style.glyphs.as_deref(),
            Some("https://example.com/fonts/{fontstack}/{range}.pbf")
        );

        // Sources: one vector source under the key `protomaps`.
        assert_eq!(style.sources.len(), 1);
        let src = style.sources.get("protomaps").expect("source");
        assert!(matches!(
            src,
            Source::Vector { url: Some(u), attribution: Some(_), .. }
                if u == "pmtiles://https://example.com/tiles.pmtiles"
        ));

        // Layers: ordered, five of them.
        assert_eq!(style.layers.len(), 5);
        assert_eq!(style.layers[0].id, "background");
        assert_eq!(style.layers[1].id, "water");
        assert_eq!(style.layers[2].id, "highways");
        assert_eq!(style.layers[3].id, "major-roads");
        assert_eq!(style.layers[4].id, "place-labels");

        // Background.
        let LayerKind::Background(bg) = &style.layers[0].kind else {
            panic!("layer 0 should be background");
        };
        assert_eq!(
            bg.paint.background_color,
            Some(Property::Constant(Color {
                r: 0xf0,
                g: 0xee,
                b: 0xe6,
                a: 0xff
            }))
        );

        // Highways layer carries the interpolate expression and filter.
        let highways = &style.layers[2];
        assert_eq!(highways.minzoom, None);
        assert!(highways.filter.is_some());
        let LayerKind::Line(line) = &highways.kind else {
            panic!("highways should be a line layer");
        };
        assert!(matches!(
            line.paint.line_width,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));

        // Symbol layer reads `name` via an expression.
        let LayerKind::Symbol(sym) = &style.layers[4].kind else {
            panic!("layer 4 should be symbol");
        };
        assert!(matches!(
            sym.layout.text_field,
            Some(Property::Expression(Expr::Get(_)))
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let bad = r#"{
            "version": 7,
            "sources": {},
            "layers": []
        }"#;
        let err = Style::from_json_str(bad).unwrap_err();
        let span = err.span().expect("version error has a span");
        assert!(matches!(err, StyleError::UnsupportedVersion { value: 7, .. }));
        // The carried span must cover the literal `7` so the renderer
        // can highlight exactly what the user had set.
        assert_eq!(&bad[span.start..span.end], "7");
    }

    #[test]
    fn expression_error_carries_path_and_span() {
        // `["interpolate", ["wat"], …]` fails inside the line-width
        // expression. The pre-pass attaches `.layers[0].paint.line-width`
        // and resolves the byte span to the array literal.
        let bad = r##"{
            "version": 8,
            "sources": {"x": {"type": "vector", "tiles": ["https://x/{z}/{x}/{y}"]}},
            "layers": [{
                "id": "roads",
                "type": "line",
                "source": "x",
                "source-layer": "roads",
                "paint": {
                    "line-color": "#abc",
                    "line-width": ["interpolate", ["wat"], ["zoom"], 0, 1]
                }
            }]
        }"##;
        let err = Style::from_json_str(bad).unwrap_err();
        match &err {
            StyleError::Expression { path, source, .. } => {
                assert_eq!(
                    path.to_string(),
                    ".layers[0].paint.line-width",
                );
                assert!(matches!(source, ExprError::UnsupportedInterpolation(_)));
            }
            other => panic!("expected Expression, got {other:?}"),
        }
        let span = err.span().expect("expression error has a span");
        let text = &bad[span.start..span.end];
        assert!(text.starts_with('['));
        assert!(text.trim_end().ends_with(']'));
        assert!(text.contains("\"interpolate\""));
    }

    #[test]
    fn syntax_error_carries_line_column_and_span() {
        // `?` is a stray invalid token. `serde_json::Error` reports
        // line/column; we translate that to a byte span pointing at
        // the bad character.
        let bad = "{\n  \"version\": ?\n}";
        let err = Style::from_json_str(bad).unwrap_err();
        let StyleError::Json {
            line, column, span, ..
        } = err
        else {
            panic!("expected Json, got {err:?}");
        };
        assert_eq!(line, 2);
        // The byte at the reported column is the `?`.
        let offset = span::byte_offset_from_line_col(bad, line, column);
        assert_eq!(bad.as_bytes()[offset], b'?');
        assert!(span.start <= offset && offset < span.end);
    }

    #[test]
    fn accepts_spec_top_level_fields() {
        // `sprite` / `metadata` / `light` / `terrain` / `bearing` /
        // `pitch` / `center` / `zoom` are spec-defined; we accept and
        // store them opaquely so real-world styles load.
        let ok = r#"{
            "version": 8,
            "sources": {},
            "layers": [],
            "sprite": "https://example.com/sprite",
            "metadata": {"x": 1},
            "bearing": 30
        }"#;
        let style = Style::from_json_str(ok).unwrap();
        assert!(style.sprite.is_some());
        assert_eq!(style.bearing, Some(30.0));
    }

    #[test]
    fn warnings_flag_unknown_top_level_fields() {
        let raw = r#"{
            "version": 8,
            "sources": {},
            "layers": [],
            "id": "vendor-id",
            "owner": "vendor"
        }"#;
        let (_, warnings) = Style::parse_with_warnings(raw).unwrap();
        let names: Vec<&str> = warnings
            .iter()
            .filter_map(|w| match w {
                StyleWarning::UnknownTopLevelField(n) => Some(n.as_str()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"id"));
        assert!(names.contains(&"owner"));
    }

    #[test]
    fn warnings_flag_raster_sources() {
        let raw = r#"{
            "version": 8,
            "sources": {
                "vendor_raster": {"type": "raster", "tiles": ["https://x/{z}/{x}/{y}.png"]}
            },
            "layers": []
        }"#;
        let (_, warnings) = Style::parse_with_warnings(raw).unwrap();
        assert!(warnings.iter().any(|w| matches!(
            w,
            StyleWarning::UnsupportedSourceType { source_id, source_type }
                if source_id == "vendor_raster" && source_type == "raster"
        )));
    }

    #[test]
    fn parses_layout_visibility_on_every_kind() {
        // Spec puts `visibility` inside `layout` for every layer
        // kind. The parser lifts it to the top-level `Layer.visibility`
        // field so render-side checks don't have to reach into a
        // kind-specific layout struct.
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [
                {"id": "bg", "type": "background",
                 "layout": {"visibility": "none"},
                 "paint": {"background-color": "#000"}},
                {"id": "fl", "type": "fill", "source": "x", "source-layer": "fl",
                 "layout": {"visibility": "visible"},
                 "paint": {"fill-color": "#abc"}},
                {"id": "ln", "type": "line", "source": "x", "source-layer": "ln",
                 "layout": {"visibility": "none", "line-cap": "round"},
                 "paint": {"line-color": "#abc"}},
                {"id": "sy", "type": "symbol", "source": "x", "source-layer": "sy",
                 "layout": {"visibility": "none", "text-field": "{name}"}}
            ]
        }"##;
        let style = Style::from_json_str(raw).unwrap();
        assert_eq!(style.layers[0].visibility, Visibility::None);
        assert_eq!(style.layers[1].visibility, Visibility::Visible);
        assert_eq!(style.layers[2].visibility, Visibility::None);
        assert_eq!(style.layers[3].visibility, Visibility::None);
        // The `line-cap` is honored too — visibility lives alongside
        // kind-specific layout fields.
        if let LayerKind::Line(line) = &style.layers[2].kind {
            assert_eq!(line.layout.line_cap, Some(LineCap::Round));
        } else {
            panic!("layer 2 should be line");
        }
    }

    #[test]
    fn visibility_defaults_visible_when_absent() {
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [{
                "id": "bg",
                "type": "background",
                "paint": {"background-color": "#000"}
            }]
        }"##;
        let style = Style::from_json_str(raw).unwrap();
        assert_eq!(style.layers[0].visibility, Visibility::Visible);
    }

    #[test]
    fn rejects_unknown_visibility_value() {
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [{
                "id": "bg",
                "type": "background",
                "layout": {"visibility": "hidden"},
                "paint": {"background-color": "#000"}
            }]
        }"##;
        let err = Style::from_json_str(raw).unwrap_err();
        assert!(err.to_string().contains("visibility"));
    }

    #[test]
    fn visibility_does_not_warn_as_ignored_key() {
        // Regression: before honoring visibility we emitted an
        // `IgnoredLayerKeys` warning for every layer that set it.
        // Now that the parser consumes it, no warning should fire.
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [{
                "id": "bg",
                "type": "background",
                "layout": {"visibility": "visible"},
                "paint": {"background-color": "#000"}
            }]
        }"##;
        let (_, warnings) = Style::parse_with_warnings(raw).unwrap();
        assert!(
            !warnings.iter().any(|w| matches!(
                w,
                StyleWarning::IgnoredLayerKeys { keys, .. } if keys.iter().any(|k| k == "visibility")
            )),
            "expected no visibility warning, got: {warnings:?}"
        );
    }

    #[test]
    fn warnings_flag_unmodeled_layout_keys() {
        // `visibility` is now honored, so it shouldn't fire a
        // warning. Other layout keys we don't model (`fill-sort-key`
        // here) still do.
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [{
                "id": "wood",
                "type": "fill",
                "source": "x",
                "source-layer": "wood",
                "layout": {"visibility": "visible", "fill-sort-key": 5},
                "paint": {"fill-color": "#abc"}
            }]
        }"##;
        let (_, warnings) = Style::parse_with_warnings(raw).unwrap();
        let hit = warnings.iter().find_map(|w| match w {
            StyleWarning::IgnoredLayerKeys {
                layer_id,
                block,
                keys,
                ..
            } if layer_id == "wood" && *block == "layout" => Some(keys.clone()),
            _ => None,
        });
        assert_eq!(hit, Some(vec!["fill-sort-key".into()]));
    }

    #[test]
    fn warnings_flag_legacy_function_form() {
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [{
                "id": "wood",
                "type": "fill",
                "source": "x",
                "source-layer": "wood",
                "paint": {
                    "fill-color": "#abc",
                    "fill-opacity": {"stops": [[0, 0.5], [12, 1.0]]}
                }
            }]
        }"##;
        let (_, warnings) = Style::parse_with_warnings(raw).unwrap();
        assert!(warnings.iter().any(|w| matches!(
            w,
            StyleWarning::LegacyFunctionConverted { layer_id, property }
                if layer_id == "wood" && property == "fill-opacity"
        )));
    }

    #[test]
    fn single_token_text_field_loads_without_warning() {
        // `"text-field": "{name}"` is the legacy single-token shape
        // we accept silently; the field is normalized into a `Get`
        // expression by the deserializer, and no warning should fire
        // since the substitution is fully honored.
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [{
                "id": "places",
                "type": "symbol",
                "source": "x",
                "source-layer": "places",
                "layout": {"text-field": "{name}"}
            }]
        }"##;
        let (style, warnings) = Style::parse_with_warnings(raw).unwrap();
        assert!(
            !warnings
                .iter()
                .any(|w| matches!(w, StyleWarning::MultiTokenTextField { .. })),
            "expected no MultiTokenTextField warning, got: {warnings:?}",
        );
        let LayerKind::Symbol(sym) = &style.layers[0].kind else {
            panic!("expected symbol");
        };
        assert!(matches!(
            sym.layout.text_field,
            Some(Property::Expression(Expr::Get(ref n))) if n == "name",
        ));
    }

    #[test]
    fn warnings_flag_multi_token_text_field() {
        // A mixed string like `"the name is {name}"` falls outside
        // the single-token shape we support — the literal is kept
        // verbatim and a warning identifies the offending layer and
        // value so the user can migrate to `["concat", …]`.
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [{
                "id": "labels",
                "type": "symbol",
                "source": "x",
                "source-layer": "places",
                "layout": {"text-field": "the name is {name}"}
            }]
        }"##;
        let (_, warnings) = Style::parse_with_warnings(raw).unwrap();
        let hit = warnings.iter().find_map(|w| match w {
            StyleWarning::MultiTokenTextField { layer_id, value } => {
                Some((layer_id.clone(), value.clone()))
            }
            _ => None,
        });
        assert_eq!(
            hit,
            Some(("labels".into(), "the name is {name}".into())),
        );
    }

    #[test]
    fn drops_fill_extrusion_layers_with_warning() {
        // `fill-extrusion` is a spec-valid layer kind we can't render
        // (no 3D pipeline). The parser should drop the layer instead of
        // failing the whole document, surface a single
        // `UnsupportedLayerType` warning naming it, and leave the
        // surrounding layers intact. We also pack the dropped layer
        // with unmodeled paint keys to confirm we don't double-warn
        // about properties on a layer we already discarded.
        let raw = r##"{
            "version": 8,
            "sources": {"x": {"type": "vector", "tiles": ["http://e/{z}/{x}/{y}"]}},
            "layers": [
                {"id": "bg", "type": "background",
                 "paint": {"background-color": "#000"}},
                {"id": "buildings3d", "type": "fill-extrusion",
                 "source": "x", "source-layer": "buildings",
                 "paint": {"fill-extrusion-color": "#abc",
                           "fill-extrusion-height": 12}},
                {"id": "roads", "type": "line",
                 "source": "x", "source-layer": "roads",
                 "paint": {"line-color": "#abc"}}
            ]
        }"##;
        let (style, warnings) = Style::parse_with_warnings(raw).unwrap();

        // The 3D layer is gone; its neighbours survive in original order.
        assert_eq!(style.layers.len(), 2);
        assert_eq!(style.layers[0].id, "bg");
        assert_eq!(style.layers[1].id, "roads");

        // Exactly one warning about the dropped layer, and no
        // `IgnoredLayerKeys` noise about its paint properties.
        let unsupported: Vec<_> = warnings
            .iter()
            .filter_map(|w| match w {
                StyleWarning::UnsupportedLayerType {
                    layer_id,
                    layer_type,
                } => Some((layer_id.clone(), layer_type.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            unsupported,
            vec![("buildings3d".into(), "fill-extrusion".into())],
        );
        assert!(
            !warnings.iter().any(|w| matches!(
                w,
                StyleWarning::IgnoredLayerKeys { layer_id, .. }
                    if layer_id == "buildings3d"
            )),
            "expected no IgnoredLayerKeys for the dropped layer, got: {warnings:?}",
        );
    }

    #[test]
    fn warnings_flag_unmodeled_paint_keys() {
        // `fill-extrusion-color` is the canonical "we never honour
        // this" stand-in — `fill-extrusion-*` is a 3D layer kind
        // and explicitly non-goal under this roadmap. (Before Phase 8
        // this test used `fill-antialias`, which has since been
        // accepted at parse time as a parse-only property; we picked
        // a still-truly-ignored field to keep the assertion live.)
        let raw = r##"{
            "version": 8,
            "sources": {},
            "layers": [{
                "id": "wood",
                "type": "fill",
                "source": "x",
                "source-layer": "wood",
                "paint": {"fill-color": "#abc", "fill-extrusion-color": "#abc"}
            }]
        }"##;
        let (_, warnings) = Style::parse_with_warnings(raw).unwrap();
        let keys = warnings.iter().find_map(|w| match w {
            StyleWarning::IgnoredLayerKeys {
                layer_id,
                block,
                keys,
                ..
            } if layer_id == "wood" && *block == "paint" => Some(keys.clone()),
            _ => None,
        });
        assert_eq!(keys, Some(vec!["fill-extrusion-color".into()]));
    }

    #[test]
    fn sprite_string_form_round_trips() {
        // The canonical shape — a single URL prefix. This is what
        // every production style we've seen authors uses.
        let raw = r#"{
            "version": 8,
            "sources": {},
            "layers": [],
            "sprite": "https://example.com/sprite"
        }"#;
        let (style, warnings) = Style::parse_with_warnings(raw).unwrap();
        assert_eq!(style.sprite.as_deref(), Some("https://example.com/sprite"));
        assert!(
            !warnings
                .iter()
                .any(|w| matches!(w, StyleWarning::MultiSpriteUnsupported)),
            "single-string sprite must not warn, got: {warnings:?}"
        );
    }

    #[test]
    fn sprite_array_form_warns_and_picks_first_url() {
        // Multi-sprite array: spec-legal but rare. We pick the first
        // entry's `url` and surface a `MultiSpriteUnsupported` so
        // authors know trailing entries were dropped.
        let raw = r#"{
            "version": 8,
            "sources": {},
            "layers": [],
            "sprite": [
                {"id": "default", "url": "https://example.com/sprite"},
                {"id": "extra",   "url": "https://example.com/sprite2"}
            ]
        }"#;
        let (style, warnings) = Style::parse_with_warnings(raw).unwrap();
        assert_eq!(style.sprite.as_deref(), Some("https://example.com/sprite"));
        assert!(warnings
            .iter()
            .any(|w| matches!(w, StyleWarning::MultiSpriteUnsupported)));
    }

    #[test]
    fn accepts_unknown_top_level_field() {
        // Real-world MapLibre styles carry vendor extensions (`id`,
        // `owner`, `created`, …) we don't model. The parser is
        // lenient at this level so a typo like `spritte` slips
        // through; the renderer surfaces issues with documented spec
        // fields when it actually consumes them. The trade-off is
        // worth it — refusing valid real-world documents costs more
        // than catching typos here.
        let ok = r#"{
            "version": 8,
            "sources": {},
            "layers": [],
            "spritte": "https://example.com/sprite"
        }"#;
        let style = Style::from_json_str(ok).unwrap();
        assert!(style.sprite.is_none());
    }

    #[test]
    fn fixture_filter_matches_highway() {
        // End-to-end eval smoke test: parse the fixture, look up the
        // roads filter, run it against a highway-tagged feature.
        let style = Style::from_json_str(MINIMAL_STYLE).unwrap();
        let roads = &style.layers[2];
        let filter = roads.filter.as_ref().expect("filter present");
        let ctx = EvalContext {
            zoom: 13.0,
            canonical: None,
            ..Default::default()
        };
        assert!(filter.matches(&ctx, &highway_feature()).unwrap());

        // A non-highway feature is filtered out.
        let mut river = highway_feature();
        river
            .properties
            .insert("kind".into(), MvtValue::String("river".into()));
        assert!(!filter.matches(&ctx, &river).unwrap());
    }

    #[test]
    fn fixture_line_width_interpolates_with_zoom() {
        let style = Style::from_json_str(MINIMAL_STYLE).unwrap();
        let LayerKind::Line(line) = &style.layers[2].kind else {
            panic!("roads is a line layer");
        };
        let width = line.paint.line_width.as_ref().expect("line-width set");

        let resolve = |z: f64| {
            width
                .resolve(
                    &EvalContext {
                        zoom: z,
                        canonical: None,
                        ..Default::default()
                    },
                    &highway_feature(),
                )
                .unwrap()
        };

        // The interpolate stops are `(3 → 0.5, 10 → 1, 16 → 6)`.
        // Inner stops and their midpoints land exactly; below the
        // first stop and above the last clamp to the endpoint values.
        assert!((resolve(3.0) - 0.5).abs() < 1e-5);
        assert!((resolve(10.0) - 1.0).abs() < 1e-5);
        assert!((resolve(16.0) - 6.0).abs() < 1e-5);
        // Midpoint of the 10→16 segment lands at 3.5.
        assert!((resolve(13.0) - 3.5).abs() < 1e-5);
        // Below the first stop clamps to 0.5; above the last to 6.0.
        assert!((resolve(1.0) - 0.5).abs() < 1e-5);
        assert!((resolve(20.0) - 6.0).abs() < 1e-5);
    }
}
