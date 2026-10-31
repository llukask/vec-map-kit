// A style layer describes one renderable pass: what to draw from which
// source-layer, optionally filtered, and what paint / layout
// properties to use. The MapLibre spec has a small zoo of layer kinds;
// we only model the four `minimal_style.json` uses, each with the
// subset of paint/layout properties that file actually sets.
//
// Common fields (`id`, `source`, `source-layer`, `minzoom`, `maxzoom`,
// `filter`) live on the outer [`Layer`] struct so they don't need to
// be duplicated across each kind. Kind-specific paint/layout are
// pulled in via [`LayerKind`].

use serde::Deserialize;
use serde::de::{self, Deserializer};
use serde_json::Value as Json;

use super::color::Color;
use super::expression::{EvalContext, EvalError, Expr, ExprError, Filter, Value};
use crate::mvt::Feature;

// ==============================================================================
// Property<T> — either a constant `T` or an expression that evaluates to one
// ==============================================================================

/// A style property that is either a plain literal or an expression.
/// Paint and layout properties use this so they can be data-driven
/// or zoom-driven without forcing every property to be an `Expr`.
#[derive(Debug, Clone, PartialEq)]
pub enum Property<T> {
    Constant(T),
    Expression(Expr),
}

impl<T> Property<T> {
    /// Resolve the property to a concrete `T` for a given context and
    /// feature. For constants this just clones the inner value; for
    /// expressions it evaluates and converts via `T`'s
    /// [`FromStyleValue`] impl.
    pub fn resolve(&self, ctx: &EvalContext, feature: &Feature) -> Result<T, PropertyError>
    where
        T: FromStyleValue + Clone,
    {
        match self {
            Property::Constant(v) => Ok(v.clone()),
            Property::Expression(e) => {
                let v = e.eval(ctx, feature)?;
                T::from_style_value(v)
            }
        }
    }
}

/// Conversion from the dynamic style [`Value`] into the typed property
/// `T`. Each property type implements this with its own coercion
/// rules.
pub trait FromStyleValue: Sized {
    fn from_style_value(v: Value) -> Result<Self, PropertyError>;
}

impl FromStyleValue for Color {
    fn from_style_value(v: Value) -> Result<Self, PropertyError> {
        match v {
            // CSS color strings (hex / `rgb(...)` / `hsl(...)`) stay
            // unresolved until first read.
            Value::String(s) => Color::parse(&s)
                .map_err(|e| PropertyError::TypeMismatch(format!("invalid color: {e}"))),
            // Phase 6+ expressions (`rgba(...)`, `to-color`, color
            // interpolation, etc.) produce a `Value::Color` directly.
            // Channels are straight `[r, g, b, a]` in 0..=1, matching
            // upstream's normalized output shape — convert to the u8
            // representation the renderer expects.
            Value::Color([r, g, b, a]) => {
                let to_u8 = |c: f64| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
                Ok(Color {
                    r: to_u8(r),
                    g: to_u8(g),
                    b: to_u8(b),
                    a: to_u8(a),
                })
            }
            other => Err(PropertyError::TypeMismatch(format!(
                "expected a color, got {other:?}"
            ))),
        }
    }
}

impl FromStyleValue for f32 {
    fn from_style_value(v: Value) -> Result<Self, PropertyError> {
        match v {
            Value::Number(n) => Ok(n as f32),
            other => Err(PropertyError::TypeMismatch(format!(
                "expected a number, got {other:?}"
            ))),
        }
    }
}

impl FromStyleValue for SymbolPlacement {
    fn from_style_value(v: Value) -> Result<Self, PropertyError> {
        match v {
            Value::String(s) => match s.as_str() {
                "point" => Ok(Self::Point),
                "line" => Ok(Self::Line),
                "line-center" => Ok(Self::LineCenter),
                other => Err(PropertyError::TypeMismatch(format!(
                    "unknown symbol-placement {other:?}"
                ))),
            },
            other => Err(PropertyError::TypeMismatch(format!(
                "expected a symbol-placement string, got {other:?}"
            ))),
        }
    }
}

impl FromStyleValue for String {
    fn from_style_value(v: Value) -> Result<Self, PropertyError> {
        // The MapLibre spec defines a `to-string` coercion for the
        // common scalar types. We mirror that — including booleans —
        // because labels often want to display data-driven values.
        match v {
            Value::String(s) => Ok(s),
            Value::Number(n) => Ok(n.to_string()),
            Value::Bool(b) => Ok(b.to_string()),
            Value::Null => Ok(String::new()),
            // Arrays and objects don't appear in property-shaped style
            // values today; expressions that produce them get rejected
            // at resolve time rather than rendered as JSON, since
            // labels with structural payloads are almost certainly a
            // bug upstream. Same goes for `Formatted` / `Collator` —
            // they're meaningful only to specific operators and never
            // make it to a raw text property.
            other @ (Value::Array(_)
            | Value::Object(_)
            | Value::Formatted(_)
            | Value::Collator(_)
            | Value::Color(_)
            // Resolved images carry a sprite name we *could* surface
            // here, but the property-resolution layer hasn't grown an
            // image-aware path yet — a raw text property that
            // resolves to an image is almost certainly a bug at the
            // style author level, so we report the mismatch the same
            // way as `Formatted` / `Collator`.
            | Value::Image(_)) => Err(PropertyError::TypeMismatch(format!(
                "expected a stringable scalar, got {other:?}"
            ))),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PropertyError {
    #[error("property type mismatch: {0}")]
    TypeMismatch(String),
    #[error(transparent)]
    Eval(#[from] EvalError),
}

// ==============================================================================
// `Property<T>` Deserialize:  literal of T  ||  expression
// ==============================================================================
//
// Deserialization happens in two stages: first land on a
// `serde_json::Value`, then decide based on shape. Arrays whose first
// element is a string are expressions; everything else we try to parse
// as a literal `T` via serde. This mirrors how MapLibre style
// implementations dispatch — a string-headed array is the unambiguous
// signal of an expression call.
//
// We can't blanket-impl `Deserialize` for `Property<T>` because the
// "everything else" branch needs a per-`T` literal parser. Instead we
// have a small free function `deserialize_property` per `T` we care
// about. It would be possible to wrap each `T` in a custom visitor,
// but going via `serde_json::Value` keeps the code paths obviously
// equivalent across types.

fn property_from_json<T>(v: Json) -> Result<Property<T>, PropertyDeserializeError>
where
    T: serde::de::DeserializeOwned,
{
    // Three shapes can land here:
    //
    //   1. A JSON array headed by a string — a modern expression call
    //      (`["interpolate", ["linear"], …]`). Pass to `Expr::from_json`.
    //   2. A JSON object with `stops` / `property` / `type` — a
    //      pre-v8 *function*, the legacy data-driven form many
    //      real-world stylesheets (MapTiler, OpenMapTiles) still
    //      ship. Route through `convert_legacy_function` so it
    //      becomes the equivalent modern expression.
    //   3. Everything else (string, number, bool, plain array of
    //      numbers) — a literal of the property's concrete type.
    //      Hand to serde and surface a precise per-literal error.
    if is_expression_shape(&v) {
        let expr = Expr::from_json(&v).map_err(PropertyDeserializeError::Expr)?;
        Ok(Property::Expression(expr))
    } else if super::legacy::is_legacy_function(&v) {
        let expr = super::legacy::convert_legacy_function(&v, None)
            .map_err(PropertyDeserializeError::Expr)?;
        Ok(Property::Expression(expr))
    } else {
        let lit: T = serde_json::from_value(v).map_err(PropertyDeserializeError::Literal)?;
        Ok(Property::Constant(lit))
    }
}

pub(super) fn is_expression_shape(v: &Json) -> bool {
    matches!(
        v,
        Json::Array(items)
            if items.first().and_then(Json::as_str).is_some()
    )
}

#[derive(Debug, thiserror::Error)]
enum PropertyDeserializeError {
    #[error("expression parse error: {0}")]
    Expr(ExprError),
    #[error("literal parse error: {0}")]
    Literal(serde_json::Error),
}

// ==============================================================================
// Legacy `{token}` substitution in `text-field`
// ==============================================================================
//
// The original Mapbox GL Style Spec defined a "tokens" feature on
// `text-field` (and `icon-image`): a literal string like `"{name}"`
// was substituted with the feature's `name` property at render time.
// MapLibre inherited the behavior but dropped the prose; current
// runtimes still honor it for back-compat.
//
// We support only the *single-token* form (`"{name}"`), normalized at
// parse time into the equivalent modern expression `["get", "name"]`.
// Mixed strings (`"the name is {name}"`) or multi-token strings stay
// as literal text and trigger a [`StyleWarning::MultiTokenTextField`]
// from `mod.rs` — authors who want concatenation should reach for
// `["concat", …]` instead. We deliberately don't try to model nested
// or escaped braces; the legacy spec didn't either.

#[derive(Debug, PartialEq, Eq)]
pub(super) enum TokenForm<'a> {
    /// No balanced `{…}` token in the string.
    None,
    /// The whole string is exactly one `{property}` token; carries
    /// the inner property name (without the braces).
    Single(&'a str),
    /// At least one `{…}` token is present, but the string isn't a
    /// clean single-token reference.
    Mixed,
}

/// Scan `s` for balanced `{property}` runs and classify the result.
/// Empty `{}`, runs that never close, and any nested `{` inside an
/// open token disqualify that occurrence — we only count well-formed
/// `{name}` shapes, matching what real-world stylesheets emit.
pub(super) fn classify_token_string(s: &str) -> TokenForm<'_> {
    let bytes = s.as_bytes();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // Look for the first `{` or `}` after this one. If it's a
            // `}` with at least one byte of name between them, we
            // have a well-formed token; otherwise this `{` is just
            // literal text and we advance past it.
            if let Some(rel) = s[i + 1..].find(['{', '}']) {
                let end = i + 1 + rel;
                if bytes[end] == b'}' && end > i + 1 {
                    spans.push((i, end + 1));
                    i = end + 1;
                    continue;
                }
            }
        }
        i += 1;
    }

    match spans.as_slice() {
        [] => TokenForm::None,
        [(start, end)] if *start == 0 && *end == s.len() => {
            TokenForm::Single(&s[start + 1..end - 1])
        }
        _ => TokenForm::Mixed,
    }
}

/// Rewrite a single-token literal `text-field` value into the
/// equivalent `["get", name]` expression. Plain strings and mixed
/// strings pass through untouched — the mixed case is surfaced as a
/// style warning during raw-JSON traversal, not here.
fn normalize_text_token(prop: Property<String>) -> Property<String> {
    if let Property::Constant(s) = &prop
        && let TokenForm::Single(name) = classify_token_string(s)
    {
        return Property::Expression(Expr::Get(name.to_string()));
    }
    prop
}

// Per-`T` named wrappers around `property_from_json`. `#[serde(deserialize_with
// = ...)]` needs a path that already binds the type parameter, so we
// supply one wrapper per concrete property type rather than a generic
// `deserialize_property<T>`. All paint/layout fields are `Option`-wrapped
// (the spec lets every property default), so only the `opt_*` forms are
// needed here.

mod prop_de {
    use super::{Color, Property};
    use serde::{Deserialize, Deserializer};

    pub fn opt_color<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Property<Color>>, D::Error> {
        Option::<serde_json::Value>::deserialize(d)?
            .map(|v| super::property_from_json::<Color>(v).map_err(serde::de::Error::custom))
            .transpose()
    }

    pub fn opt_f32<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Property<f32>>, D::Error> {
        Option::<serde_json::Value>::deserialize(d)?
            .map(|v| super::property_from_json::<f32>(v).map_err(serde::de::Error::custom))
            .transpose()
    }

    pub fn opt_string<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Property<String>>, D::Error> {
        Option::<serde_json::Value>::deserialize(d)?
            .map(|v| super::property_from_json::<String>(v).map_err(serde::de::Error::custom))
            .transpose()
    }

    /// `text-field` deserializer. Handles three shapes:
    ///
    /// 1. Modern expression (`["concat", …]`, `["get", "name"]`, …):
    ///    parsed by the generic property pipeline.
    /// 2. Legacy zoom-stops function (`{ "stops": [[z, "{TOKEN}"], …] }`,
    ///    as used by the MapLibre demo's `countries-label` layer):
    ///    routed through [`super::super::legacy::convert_legacy_function`]
    ///    with a synthetic property spec that flips on the converter's
    ///    `tokens` flag, so each `{prop}` stop output becomes
    ///    `["to-string", ["get", "prop"]]`. Without the spec the
    ///    literals fall through unchanged and render as `{TOKEN}` —
    ///    visible especially under `text-transform: uppercase`, which
    ///    turns `{name}` into the literal `{NAME}`.
    /// 3. Plain string literal: passed through `normalize_text_token`
    ///    so the single-token shorthand `"{name}"` becomes
    ///    `["get", "name"]`. Mixed-token strings stay as literals
    ///    (flagged by `collect_layer_warnings` during the raw-JSON pass).
    pub fn opt_text_field<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Property<String>>, D::Error> {
        let Some(raw) = Option::<serde_json::Value>::deserialize(d)? else {
            return Ok(None);
        };
        let prop = if super::is_expression_shape(&raw) {
            super::property_from_json::<String>(raw).map_err(serde::de::Error::custom)?
        } else if super::super::legacy::is_legacy_function(&raw) {
            // `tokens: true` is the bit that drives stop-output token
            // expansion inside the legacy converter; `type: "string"`
            // passes the `is_string_kind` gate next to it.
            let spec = serde_json::json!({ "type": "string", "tokens": true });
            let expr = super::super::legacy::convert_legacy_function(&raw, Some(&spec))
                .map_err(serde::de::Error::custom)?;
            Property::Expression(expr)
        } else {
            let lit: String = serde_json::from_value(raw).map_err(serde::de::Error::custom)?;
            super::normalize_text_token(Property::Constant(lit))
        };
        Ok(Some(prop))
    }

    /// `text-font` accepts either a literal `["Family", …]` list or an
    /// expression that evaluates to one at render time. We only model
    /// the literal form today, so the expression form is accepted and
    /// silently dropped — the renderer falls back to its default font
    /// stack rather than refusing the whole layer.
    pub fn opt_string_list<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Vec<String>>, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(d)?;
        let Some(raw) = raw else { return Ok(None) };
        Ok(parse_string_list(&raw))
    }

    fn parse_string_list(v: &serde_json::Value) -> Option<Vec<String>> {
        let arr = v.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            out.push(item.as_str()?.to_string());
        }
        Some(out)
    }

    /// `line-dasharray` is a list of numbers in line-width units, but
    /// the spec also lets it be an expression that resolves to one. We
    /// only model the literal form today — the renderer's dash uniform
    /// is uploaded once per frame so a per-feature data-driven pattern
    /// has no place to land. The expression form is accepted and
    /// silently dropped (the layer renders as solid), mirroring how we
    /// handle `text-font` and `line-cap` / `line-join`.
    pub fn opt_f32_list<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<f32>>, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(d)?;
        let Some(raw) = raw else { return Ok(None) };
        if super::is_expression_shape(&raw) {
            return Ok(None);
        }
        Ok(parse_f32_list(&raw))
    }

    fn parse_f32_list(v: &serde_json::Value) -> Option<Vec<f32>> {
        let arr = v.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            out.push(item.as_f64()? as f32);
        }
        Some(out)
    }

    // The enum-valued layout properties (`line-cap`, `line-join`) accept
    // either a bare string literal or an expression that resolves to
    // one. The spec allows data-driven values, but real-world styles
    // overwhelmingly write them as literals — the same precedent we
    // followed for `text-font`. We accept the expression shape so the
    // document still loads, but silently drop it; the renderer falls
    // back to the spec default. A future phase can add expression
    // evaluation here without changing the field's public type.
    pub fn opt_line_cap<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<super::LineCap>, D::Error> {
        opt_enum_literal::<D, super::LineCap>(d)
    }

    pub fn opt_line_join<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<super::LineJoin>, D::Error> {
        opt_enum_literal::<D, super::LineJoin>(d)
    }

    // Phase 4 enum-valued symbol layout properties: anchor / transform /
    // justify. Same accept-the-literal, silently-drop-the-expression
    // policy `text-font` / `line-cap` use. The renderer reads the
    // resolved enum at layout time rather than per-feature, so a
    // data-driven shape has nowhere to land.

    pub fn opt_text_anchor<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<super::TextAnchor>, D::Error> {
        opt_enum_literal::<D, super::TextAnchor>(d)
    }

    pub fn opt_text_transform<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<super::TextTransform>, D::Error> {
        opt_enum_literal::<D, super::TextTransform>(d)
    }

    pub fn opt_text_justify<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<super::TextJustify>, D::Error> {
        opt_enum_literal::<D, super::TextJustify>(d)
    }

    /// `icon-anchor` follows the same accept-the-literal /
    /// silently-drop-the-expression pattern as `text-anchor`. The
    /// renderer reads the resolved enum once per layer at draw time;
    /// a data-driven shape has nowhere to land without per-feature
    /// re-bind machinery (Phase 7 collision territory).
    pub fn opt_icon_anchor<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<super::IconAnchor>, D::Error> {
        opt_enum_literal::<D, super::IconAnchor>(d)
    }

    /// `icon-image` deserializer. Spec quirk: unlike `text-field`,
    /// the legacy `{token}` shorthand **does not** apply to
    /// `icon-image` — `"airport_11"` is the literal sprite name,
    /// not `{airport_11}` magic. We therefore route through the
    /// generic string property dispatch without the
    /// `normalize_text_token` rewrite that `text-field` uses, so
    /// `"{name}"` lands as the literal string `"{name}"` (which
    /// will miss in the atlas and silently skip — matching upstream
    /// MapLibre).
    pub fn opt_icon_image<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Property<String>>, D::Error> {
        opt_string(d)
    }

    /// `icon-offset` is `[dx, dy]` **in pixels** (the spec is
    /// explicit: pixels for icons, em for text). Mirrors
    /// [`opt_em_offset`] otherwise — same accept-the-literal /
    /// drop-the-expression policy.
    pub fn opt_icon_offset<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<[f32; 2]>, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(d)?;
        let Some(raw) = raw else { return Ok(None) };
        if super::is_expression_shape(&raw) {
            return Ok(None);
        }
        let arr = raw.as_array().ok_or_else(|| {
            serde::de::Error::custom("icon-offset must be a 2-element array")
        })?;
        if arr.len() != 2 {
            return Err(serde::de::Error::custom(
                "icon-offset must be a 2-element array",
            ));
        }
        let dx = arr[0]
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("icon-offset[0] must be a number"))?
            as f32;
        let dy = arr[1]
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("icon-offset[1] must be a number"))?
            as f32;
        Ok(Some([dx, dy]))
    }

    /// `icon-padding` mirrors `text-padding` in shape (scalar or
    /// 4-vec, both kept verbatim as a `Vec<f32>`). Parsed-only
    /// because collision detection is Phase 7; the field sits on
    /// `SymbolLayout` so styles that set it load without warnings.
    pub fn opt_icon_padding<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Vec<f32>>, D::Error> {
        // The shape is identical to text-padding, so reuse it
        // directly — keeps the spec semantics in one place. (If the
        // spec ever diverges the two we can split them; today
        // they're the same.)
        opt_text_padding(d)
    }

    fn opt_enum_literal<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
    where
        D: Deserializer<'de>,
        T: serde::de::DeserializeOwned,
    {
        let raw = Option::<serde_json::Value>::deserialize(d)?;
        let Some(raw) = raw else { return Ok(None) };
        // Expression-shaped or legacy-function-shaped value: accept
        // and drop, mirroring the `text-font` / `text-padding` policy.
        // MapTiler stylesheets ship zoom-based legacy functions for
        // enum-valued knobs like `symbol-placement` ({base, stops}),
        // which we can't honor without per-zoom rebind machinery.
        if super::is_expression_shape(&raw) || super::super::legacy::is_legacy_function(&raw) {
            return Ok(None);
        }
        serde_json::from_value(raw)
            .map(Some)
            .map_err(serde::de::Error::custom)
    }

    /// `text-offset` is `[dx, dy]` in `em` units. The spec also allows
    /// an expression that resolves to a two-element array; we drop that
    /// shape for the same reason as `text-font` / `line-dasharray` —
    /// the renderer reads the offset once per layer at shape time and
    /// has nowhere to plumb a per-feature data-driven value.
    pub fn opt_em_offset<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[f32; 2]>, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(d)?;
        let Some(raw) = raw else { return Ok(None) };
        if super::is_expression_shape(&raw) {
            return Ok(None);
        }
        let arr = raw.as_array().ok_or_else(|| {
            serde::de::Error::custom("text-offset must be a 2-element array")
        })?;
        if arr.len() != 2 {
            return Err(serde::de::Error::custom(
                "text-offset must be a 2-element array",
            ));
        }
        let dx = arr[0]
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("text-offset[0] must be a number"))?
            as f32;
        let dy = arr[1]
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("text-offset[1] must be a number"))?
            as f32;
        Ok(Some([dx, dy]))
    }

    /// `text-padding` is *parse-only* until Phase 7 wires collision
    /// detection. The spec allows a scalar number (uniform padding) or
    /// a 4-element array `[top, right, bottom, left]`; we accept either
    /// as a `Vec<f32>` so the document loads cleanly, but the renderer
    /// never reads it. Any other shape — modern expression
    /// (`[string, …]`) or the legacy function form (`{"stops": …}`,
    /// emitted by MapTiler's stylesheets) — is silently dropped,
    /// matching the `text-font` policy: since collision is the only
    /// consumer and it's not built yet, rejecting these shapes would
    /// just refuse real-world documents for no end-user gain.
    pub fn opt_text_padding<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Vec<f32>>, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(d)?;
        let Some(raw) = raw else { return Ok(None) };
        if super::is_expression_shape(&raw) || super::super::legacy::is_legacy_function(&raw) {
            return Ok(None);
        }
        if let Some(n) = raw.as_f64() {
            return Ok(Some(vec![n as f32]));
        }
        if let Some(arr) = raw.as_array() {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                let n = item.as_f64().ok_or_else(|| {
                    serde::de::Error::custom("text-padding array must contain numbers")
                })?;
                out.push(n as f32);
            }
            return Ok(Some(out));
        }
        Err(serde::de::Error::custom(
            "text-padding must be a number or an array of numbers",
        ))
    }

    // ----- Phase 6: line-following label deserializers --------------------
    //
    // The four new layout properties (`symbol-placement`,
    // `symbol-spacing`, `text-max-angle`, `text-keep-upright`) plus the
    // parse-only `text-writing-mode` follow the standard accept-literal
    // / drop-expression policy established by `text-anchor` and friends.
    // `symbol-placement` and `text-writing-mode` are enums dispatched
    // through `opt_enum_literal`; the scalar / boolean knobs go through
    // dedicated wrappers below.

    pub fn opt_symbol_placement<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Property<super::SymbolPlacement>>, D::Error> {
        Option::<serde_json::Value>::deserialize(d)?
            .map(|v| {
                super::property_from_json::<super::SymbolPlacement>(v)
                    .map_err(serde::de::Error::custom)
            })
            .transpose()
    }

    pub fn opt_text_writing_mode<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<super::TextWritingMode>, D::Error> {
        opt_enum_literal::<D, super::TextWritingMode>(d)
    }

    /// `fill-translate-anchor` / `line-translate-anchor` (Phase 8) —
    /// enum literal `Map` / `Viewport`. Same accept-the-literal /
    /// drop-the-expression policy `text-anchor` uses. The renderer
    /// reads the resolved anchor once per layer per frame, so a
    /// data-driven shape has nowhere to land.
    pub fn opt_translate_anchor<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<super::TranslateAnchor>, D::Error> {
        opt_enum_literal::<D, super::TranslateAnchor>(d)
    }

    /// `fill-translate` / `line-translate` (Phase 8) — `[dx, dy]` in
    /// **pixels**. Mirrors `opt_icon_offset` in shape: expression
    /// form silently dropped (the renderer reads the resolved
    /// translate once per layer per frame from a `Property<[f32; 2]>`
    /// constant, so a per-feature data-driven shape has nowhere to
    /// plumb). Negative values are legal — the spec uses them for
    /// drop-shadow effects offset up/left of a fill.
    pub fn opt_px_offset<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[f32; 2]>, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(d)?;
        let Some(raw) = raw else { return Ok(None) };
        if super::is_expression_shape(&raw) {
            return Ok(None);
        }
        let arr = raw
            .as_array()
            .ok_or_else(|| serde::de::Error::custom("translate must be a 2-element array"))?;
        if arr.len() != 2 {
            return Err(serde::de::Error::custom(
                "translate must be a 2-element array",
            ));
        }
        let dx = arr[0]
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("translate[0] must be a number"))?
            as f32;
        let dy = arr[1]
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("translate[1] must be a number"))?
            as f32;
        Ok(Some([dx, dy]))
    }

    /// `text-keep-upright` (and any future boolean layout knobs).
    /// Accepts a bare boolean literal; drops the expression form per
    /// the policy used by `text-anchor` / `line-cap` — the renderer
    /// reads the resolved value once per layer at shape time and has
    /// nowhere to land a per-feature data-driven boolean without a
    /// per-glyph re-bind.
    pub fn opt_bool<'de, D: Deserializer<'de>>(d: D) -> Result<Option<bool>, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(d)?;
        let Some(raw) = raw else { return Ok(None) };
        if super::is_expression_shape(&raw) {
            return Ok(None);
        }
        raw.as_bool()
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected a boolean literal"))
    }
}

// ==============================================================================
// Paint / layout structs per layer kind
// ==============================================================================
//
// Field naming: serde renames each property to its kebab-case spec name.
// All fields are optional — the spec defines defaults we don't need to
// know about for parsing, only for evaluation. Unknown fields are an
// error to surface typos and unsupported properties.

// Paint and layout deserializers below are lenient (no
// `deny_unknown_fields`): the spec defines many properties per layer
// kind that we don't render yet (e.g. `fill-opacity`, `line-dasharray`,
// `text-halo-color`), and a real-world style document typically sets
// several of them. Silently ignoring them keeps `--style PATH` useful
// even when the style references properties we haven't wired through.
// Misspellings will be visually obvious (the property won't apply)
// rather than blocking the whole document from loading.

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct BackgroundPaint {
    #[serde(
        rename = "background-color",
        default,
        deserialize_with = "prop_de::opt_color"
    )]
    pub background_color: Option<Property<Color>>,
    /// Spec default is `1.0` (fully opaque). `None` here means
    /// "absent in the document"; the renderer treats that as the
    /// default. Data-driven so styles can fade the background via
    /// zoom interpolation.
    #[serde(
        rename = "background-opacity",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub background_opacity: Option<Property<f32>>,
    /// `background-pattern` (Phase 8) — sprite name to tile across
    /// the full surface as the page-level background. Spec lets the
    /// pattern fall back to `background-color` if the sprite name
    /// misses; the renderer follows that rule. Data-driven shapes
    /// (zoom-interpolated pattern swaps) are supported via the
    /// standard `Property<String>` machinery.
    #[serde(
        rename = "background-pattern",
        default,
        deserialize_with = "prop_de::opt_string"
    )]
    pub background_pattern: Option<Property<String>>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct FillPaint {
    #[serde(
        rename = "fill-color",
        default,
        deserialize_with = "prop_de::opt_color"
    )]
    pub fill_color: Option<Property<Color>>,
    /// Per-feature opacity multiplier; spec default is `1.0`. Treated
    /// as `None`-means-default to match the sibling colour field; the
    /// renderer skips features whose resolved opacity is `<= 0.0`.
    #[serde(
        rename = "fill-opacity",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub fill_opacity: Option<Property<f32>>,
    /// `fill-pattern` (Phase 8) — sprite name to tile across the
    /// fill geometry. When set, the layer routes through the pattern
    /// fill pipeline (`render::pipeline::pattern_fill`) which samples
    /// the sprite atlas in *world space* so the pattern stays
    /// continuous across tile boundaries; see the
    /// `PATTERN_SHADER_SRC` doc-comment for the UV math. Missing
    /// sprite names fall back to the solid `fill-color`. Data-driven
    /// shapes (`["match", ["get", "kind"], …]`) resolve per feature.
    #[serde(
        rename = "fill-pattern",
        default,
        deserialize_with = "prop_de::opt_string"
    )]
    pub fill_pattern: Option<Property<String>>,
    /// `fill-translate` (Phase 8) — `[dx, dy]` paint-space offset in
    /// **pixels**. Combined with [`Self::fill_translate_anchor`] to
    /// pick between world-space (`Map`) and screen-space (`Viewport`)
    /// shifts. Negative values are legal — drop-shadows under
    /// landuse polygons are the canonical use. Expression form
    /// silently dropped per the `text-offset` precedent.
    #[serde(
        rename = "fill-translate",
        default,
        deserialize_with = "prop_de::opt_px_offset"
    )]
    pub fill_translate: Option<[f32; 2]>,
    /// `fill-translate-anchor` (Phase 8) — `Map` (default) shifts the
    /// fill in world space (moves with the map) while `Viewport`
    /// shifts in screen space (stays fixed relative to the viewer).
    /// Expression form dropped per the enum-literal policy.
    #[serde(
        rename = "fill-translate-anchor",
        default,
        deserialize_with = "prop_de::opt_translate_anchor"
    )]
    pub fill_translate_anchor: Option<TranslateAnchor>,
    /// `fill-antialias` (Phase 8, **parse-only**). The spec default
    /// is `true`; our fills are flat triangle meshes drawn without
    /// MSAA, so the `false` setting is effectively the only mode we
    /// can match today. Authors who set this load cleanly but the
    /// renderer ignores the value.
    ///
    /// TODO(phase-13 polish): wiring real fill anti-aliasing means
    /// enabling MSAA on the surface or adding shader-side edge AA —
    /// both add render-state cost across every fill draw. Defer
    /// until a polish phase justifies the lift.
    #[serde(rename = "fill-antialias", default)]
    pub fill_antialias: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct LinePaint {
    #[serde(
        rename = "line-color",
        default,
        deserialize_with = "prop_de::opt_color"
    )]
    pub line_color: Option<Property<Color>>,
    #[serde(rename = "line-width", default, deserialize_with = "prop_de::opt_f32")]
    pub line_width: Option<Property<f32>>,
    /// Per-feature opacity multiplier; spec default is `1.0`.
    /// Data-driven, so `interpolate(zoom, …)` can fade lines in or
    /// out across zoom thresholds. The renderer skips strokes whose
    /// resolved value is `<= 0.0` so we don't tessellate invisible
    /// geometry.
    #[serde(
        rename = "line-opacity",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub line_opacity: Option<Property<f32>>,
    /// Maximum allowed miter ratio (length / half-width) before a
    /// miter join collapses to a bevel. Spec default is 2.0; sharper
    /// corners than this produce visually noisy spikes, so the
    /// fallback to bevel is what production maps expect to see.
    #[serde(
        rename = "line-miter-limit",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub line_miter_limit: Option<Property<f32>>,
    /// Threshold (in radians) below which round joins are coarsened
    /// to fewer segments. The spec defines this as the ratio of the
    /// approximated circular arc length to the true arc length —
    /// values closer to 1.0 mean fewer segments and a slightly
    /// faceted look. Spec default is 1.05.
    #[serde(
        rename = "line-round-limit",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub line_round_limit: Option<Property<f32>>,
    /// Dash pattern as alternating dash/gap lengths *in line-width
    /// units*: `[3, 2]` means "3-width dash, 2-width gap, repeat".
    /// Spec-legal as an expression too, but the renderer uploads a
    /// single dash uniform per frame, so a per-feature data-driven
    /// pattern has nowhere to land — the expression form is accepted
    /// at parse time and silently dropped, mirroring `text-font`'s
    /// policy. Absent means "solid line".
    #[serde(
        rename = "line-dasharray",
        default,
        deserialize_with = "prop_de::opt_f32_list"
    )]
    pub line_dasharray: Option<Vec<f32>>,
    /// Gap (in pixels) between two parallel strokes. When nonzero,
    /// each stroke is drawn at `line_width` thickness, separated by
    /// this gap — the classic road-casing trick implemented in one
    /// layer rather than two. Spec default is 0 (single stroke).
    #[serde(
        rename = "line-gap-width",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub line_gap_width: Option<Property<f32>>,
    /// Perpendicular offset (in pixels) applied to the line before
    /// the half-width expansion. Positive offsets shift the line in
    /// the direction 90° counter-clockwise from its travel direction
    /// (matching upstream MapLibre's sign convention). Useful for
    /// inset borders and shifted tunnels. Spec default is 0.
    #[serde(rename = "line-offset", default, deserialize_with = "prop_de::opt_f32")]
    pub line_offset: Option<Property<f32>>,
    /// `line-pattern` (Phase 8, **parse-only**). Implementing a
    /// pattern-textured line needs a separate line pipeline that
    /// computes per-segment UVs along the arc-length direction
    /// (rotated to follow each segment's tangent) — comparable
    /// effort to Phase 6's line-following labels. Authored values
    /// are accepted so styles parse cleanly; the layer draws as a
    /// solid `line-color` line.
    ///
    /// TODO(phase-8-followup): line-pattern pipeline. Needs
    /// arc-length-driven UVs in line space, the same axis the dash
    /// shader already walks, mapped into the sprite atlas. See
    /// SESSION.md for the deferral note.
    #[serde(
        rename = "line-pattern",
        default,
        deserialize_with = "prop_de::opt_string"
    )]
    pub line_pattern: Option<Property<String>>,
    /// `line-translate` (Phase 8) — `[dx, dy]` paint-space offset in
    /// **pixels**, applied to every line vertex. Same shape as
    /// `fill-translate`; see [`FillPaint::fill_translate`] for the
    /// anchor semantics. Expression form silently dropped.
    #[serde(
        rename = "line-translate",
        default,
        deserialize_with = "prop_de::opt_px_offset"
    )]
    pub line_translate: Option<[f32; 2]>,
    /// `line-translate-anchor` (Phase 8) — `Map` / `Viewport` toggle.
    /// See [`FillPaint::fill_translate_anchor`] for the semantics.
    #[serde(
        rename = "line-translate-anchor",
        default,
        deserialize_with = "prop_de::opt_translate_anchor"
    )]
    pub line_translate_anchor: Option<TranslateAnchor>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct LineLayout {
    /// Geometry of line endpoints. Default is `Butt` per the spec.
    /// Data-driven (expression-form) values are silently dropped —
    /// see [`prop_de::opt_line_cap`].
    #[serde(
        rename = "line-cap",
        default,
        deserialize_with = "prop_de::opt_line_cap"
    )]
    pub line_cap: Option<LineCap>,
    /// Geometry at internal vertices. Default is `Miter` per the spec.
    /// Data-driven values are silently dropped — see
    /// [`prop_de::opt_line_join`].
    #[serde(
        rename = "line-join",
        default,
        deserialize_with = "prop_de::opt_line_join"
    )]
    pub line_join: Option<LineJoin>,
}

/// `line-cap` spec enum. We don't currently distinguish between
/// `start-cap` and `end-cap` because the spec only exposes `line-cap`
/// (a single value applying to both ends).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LineCap {
    #[default]
    Butt,
    Round,
    Square,
}

/// `line-join` spec enum. `MiterClip` is a lyon-internal variant we
/// don't surface — the spec only defines `miter` (with `line-miter-limit`
/// controlling the bevel fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LineJoin {
    Bevel,
    Round,
    #[default]
    Miter,
}

/// `fill-translate-anchor` / `line-translate-anchor` (and the
/// `*-translate-anchor` family generally). Picks whether a paint-
/// space translate is applied in **world space** (`Map`, the spec
/// default — the offset moves with the map as the user pans / zooms)
/// or in **screen space** (`Viewport` — the offset stays fixed
/// relative to the viewer, useful for drop-shadows that should feel
/// attached to the screen). The renderer reads the resolved anchor
/// once per layer per frame and picks which uniform slot the
/// translate lands in; see `render::pipeline::pattern_fill` /
/// `render::pipeline::styled`.
///
/// A single shared enum because fill / line / circle / icon / text
/// all use the same shape. Deserialized as a literal — expression
/// form silently dropped; see [`prop_de::opt_translate_anchor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TranslateAnchor {
    #[default]
    Map,
    Viewport,
}

/// `text-anchor` spec enum. Names the part of the label box pinned to
/// the feature point; `Center` is the spec default. The renderer
/// translates the anchor into a screen-space offset before pushing the
/// glyphon `TextArea`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TextAnchor {
    #[default]
    Center,
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// `text-transform` spec enum. Applied to the resolved label string
/// before shaping. `None` is the spec default (pass-through).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextTransform {
    #[default]
    None,
    Uppercase,
    Lowercase,
}

/// `text-justify` spec enum. Controls multi-line alignment. `Auto`
/// follows the spec's "left for LTR runs" — in our orthographic 2D
/// renderer we don't model bidi text, so we treat `Auto` as `Center`
/// in the shaper to match what most styles intuitively expect from an
/// unspecified value (production Protomaps labels read centred when
/// left-justify forces ragged columns).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextJustify {
    #[default]
    Auto,
    Left,
    Center,
    Right,
}

/// `icon-anchor` spec enum. Names the part of the icon quad pinned to
/// the feature point; `Center` is the spec default. The renderer reads
/// this once per layer at draw time (same as `TextAnchor`) to translate
/// the rect's corner into a screen-space shift before the offset is
/// applied. Mirrors the `kebab-case` representation `TextAnchor` uses
/// so authors can copy/paste between the two properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IconAnchor {
    #[default]
    Center,
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// `symbol-placement` spec enum (Phase 6). Decides where on a feature's
/// geometry the label's anchor(s) live:
///
///   - `Point` — one label per geometry point. The historical default;
///     the renderer keeps using glyphon's `TextArea` path for these.
///   - `Line` — labels are stamped along the line at intervals of
///     `symbol-spacing` pixels and curve along the local tangent.
///   - `LineCenter` — single label at the line's arc-length midpoint,
///     curving along the local tangent the same way.
///
/// Parsed as a [`Property`], so zoom functions and expressions are kept;
/// see [`prop_de::opt_symbol_placement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SymbolPlacement {
    #[default]
    Point,
    Line,
    LineCenter,
}

/// `text-writing-mode` spec enum (Phase 6, *parse-only*). Vertical CJK /
/// Japanese station labels need this to be `Vertical`; until then every
/// label renders horizontally. The field exists on `SymbolLayout` so
/// styles that set it parse without an `IgnoredLayerKeys` warning.
///
/// TODO(phase-13+): vertical CJK shaping requires reorienting each
/// glyph 90°, plus picking the right per-glyph vertical metric from the
/// font. Out of scope for Phase 6 — the orthographic camera doesn't
/// need it to draw road labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TextWritingMode {
    #[default]
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct SymbolPaint {
    #[serde(
        rename = "text-color",
        default,
        deserialize_with = "prop_de::opt_color"
    )]
    pub text_color: Option<Property<Color>>,
    /// Glyph-fill opacity multiplier; spec default is `1.0`.
    /// Resolved per feature so labels can fade in with zoom.
    #[serde(
        rename = "text-opacity",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub text_opacity: Option<Property<f32>>,
    /// Icon opacity multiplier; spec default is `1.0`. Parsed only —
    /// icons aren't rendered yet (Phase 5), so the renderer never
    /// reads this field. Kept on the paint struct so styles that
    /// declare it load cleanly and so the wiring is in place once
    /// the sprite pipeline lands.
    #[serde(
        rename = "icon-opacity",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub icon_opacity: Option<Property<f32>>,
    /// Glyph halo colour. The renderer emits a faux halo (8 offset
    /// copies in this colour, fill on top) when `text-halo-width > 0`;
    /// `text-opacity` multiplies into both passes. Spec default is
    /// transparent black, which is treated the same as "no halo" by
    /// the renderer because faux-halo passes with α=0 just waste atlas
    /// work.
    #[serde(
        rename = "text-halo-color",
        default,
        deserialize_with = "prop_de::opt_color"
    )]
    pub text_halo_color: Option<Property<Color>>,
    /// Halo width in pixels. `MapView` draws it as eight offset copies
    /// of the glyphs under the fill (see `text::paint`).
    /// Spec default is 0 (no halo).
    #[serde(
        rename = "text-halo-width",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub text_halo_width: Option<Property<f32>>,
    /// Halo blur in pixels. *Parsed-only*: we don't run an SDF glyph
    /// pipeline so we can't approximate the Gaussian falloff the spec
    /// describes; emitting more offset copies at fractional radii would
    /// be a poor stand-in for very little visual gain. Authored values
    /// load cleanly so the document still renders.
    #[serde(
        rename = "text-halo-blur",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub text_halo_blur: Option<Property<f32>>,
    /// `icon-color` — tint applied to **SDF** sprites only; non-SDF
    /// (alpha-keyed) sprites ignore this field per the spec. Used by
    /// styles to recolour shield / arrow / chevron glyphs without
    /// shipping a sprite variant per colour. The mesh worker resolves
    /// this per feature and pre-multiplies it into the icon's tint
    /// vertex (alpha gated by `icon-opacity`).
    #[serde(
        rename = "icon-color",
        default,
        deserialize_with = "prop_de::opt_color"
    )]
    pub icon_color: Option<Property<Color>>,
    /// `icon-halo-color` — *parsed-only* in Phase 5. The real
    /// implementation needs the SDF threshold trick the spec
    /// describes (outline = SDF distance > `halo-width` rendered in
    /// halo colour). Without an SDF glyph pipeline we'd be faking it
    /// with the same 8-direction offset hack `text-halo-*` uses, but
    /// for sprite quads that'd inflate every icon into 9 textured
    /// draws. The field lands here so styles parse cleanly; the
    /// renderer ignores it. TODO(phase-5-followup): SDF-channel halo.
    #[serde(
        rename = "icon-halo-color",
        default,
        deserialize_with = "prop_de::opt_color"
    )]
    pub icon_halo_color: Option<Property<Color>>,
    /// `icon-halo-width` — parsed-only twin of `icon-halo-color`. See
    /// the TODO above; no halo geometry today.
    #[serde(
        rename = "icon-halo-width",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub icon_halo_width: Option<Property<f32>>,
    /// `icon-halo-blur` — parsed-only twin of the text-side blur
    /// equivalent. Same TODO breadcrumb as `icon-halo-color`.
    #[serde(
        rename = "icon-halo-blur",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub icon_halo_blur: Option<Property<f32>>,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct SymbolLayout {
    /// `text-field` accepts the modern expression form, a plain
    /// string label, *or* a legacy `{property}` token (the latter
    /// inherited from the original Mapbox spec, still emitted by some
    /// stylesheets). Single-token strings like `"{name}"` are
    /// normalized to `["get", "name"]` at parse time — see
    /// [`classify_token_string`] and [`prop_de::opt_text_field`].
    #[serde(
        rename = "text-field",
        default,
        deserialize_with = "prop_de::opt_text_field"
    )]
    pub text_field: Option<Property<String>>,
    /// `text-font` accepts either a literal `["Family", …]` list or
    /// an expression that picks one at render time. We only model the
    /// literal form today; the deserializer silently drops the
    /// expression form (renderer falls back to its default stack)
    /// rather than refusing the whole layer.
    #[serde(
        rename = "text-font",
        default,
        deserialize_with = "prop_de::opt_string_list"
    )]
    pub text_font: Option<Vec<String>>,
    #[serde(rename = "text-size", default, deserialize_with = "prop_de::opt_f32")]
    pub text_size: Option<Property<f32>>,
    /// `text-anchor` — where the label box pins to the feature point.
    /// Default is `Center`. Data-driven values are silently dropped
    /// (see [`prop_de::opt_text_anchor`]) because the renderer
    /// resolves the anchor once per layer at shaping time.
    #[serde(
        rename = "text-anchor",
        default,
        deserialize_with = "prop_de::opt_text_anchor"
    )]
    pub text_anchor: Option<TextAnchor>,
    /// `text-offset` — `[dx, dy]` in `em` units, applied on top of
    /// the anchor shift. Default is `[0, 0]`. Spec also allows
    /// `text-radial-offset` (a scalar that conflicts with this); we
    /// don't model radial offset, see field [`Self::text_radial_offset`].
    #[serde(
        rename = "text-offset",
        default,
        deserialize_with = "prop_de::opt_em_offset"
    )]
    pub text_offset: Option<[f32; 2]>,
    /// `text-radial-offset` — accepted on the parser as a scalar `em`
    /// distance, but the renderer doesn't apply it. The spec defines
    /// it as conflicting with `text-offset` and most production styles
    /// only set one or the other; if both are set we honour
    /// `text-offset` because that's what the existing rendering path
    /// already does. *Parsed-only* — flagged as such in the roadmap.
    #[serde(
        rename = "text-radial-offset",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub text_radial_offset: Option<Property<f32>>,
    /// `text-max-width` — wrap target in `em`. The shaper converts to
    /// pixels via `em * size_px` and feeds it to glyphon's word-wrap.
    /// Default is `10` per the spec.
    #[serde(
        rename = "text-max-width",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub text_max_width: Option<Property<f32>>,
    /// `text-letter-spacing` — extra space between glyphs, in `em`.
    /// Converted to pixels (`em * size_px`) at shape time and applied
    /// via `cosmic_text::Attrs::letter_spacing`. Default is 0.
    #[serde(
        rename = "text-letter-spacing",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub text_letter_spacing: Option<Property<f32>>,
    /// `text-line-height` — line-height multiplier. The shaper feeds
    /// `size_px * multiplier` as the glyphon line-height in
    /// `Metrics::new`. Spec default is 1.2.
    #[serde(
        rename = "text-line-height",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub text_line_height: Option<Property<f32>>,
    /// `text-transform` — applied to the resolved label string before
    /// pushing to the renderer. Default is [`TextTransform::None`].
    #[serde(
        rename = "text-transform",
        default,
        deserialize_with = "prop_de::opt_text_transform"
    )]
    pub text_transform: Option<TextTransform>,
    /// `text-justify` — multi-line alignment. Default is
    /// [`TextJustify::Auto`], which the renderer collapses to
    /// `Center` (see the enum docs for why).
    #[serde(
        rename = "text-justify",
        default,
        deserialize_with = "prop_de::opt_text_justify"
    )]
    pub text_justify: Option<TextJustify>,
    /// `text-padding` — accepted at parse time but *not* read by the
    /// renderer until Phase 7 (collision) lands. The spec allows a
    /// scalar (uniform padding) or a 4-vec (`[top, right, bottom,
    /// left]`); both decode into this `Vec<f32>` unchanged so the
    /// future collision pass can inspect the authored shape directly.
    #[serde(
        rename = "text-padding",
        default,
        deserialize_with = "prop_de::opt_text_padding"
    )]
    pub text_padding: Option<Vec<f32>>,

    // -- icon properties (Phase 5) ------------------------------------------
    //
    // The icon half of `SymbolLayer`. Each field mirrors the analogous
    // text field in shape but reads from the spec's icon variant. None
    // of these accept the legacy `{token}` shorthand — only `text-field`
    // does. Data-driven shapes follow the same accept-and-drop policy
    // (`icon-anchor` enum / `icon-offset` array) or full expression
    // dispatch (the scalar / colour properties).

    /// `icon-image` — sprite name to look up in the loaded atlas.
    /// Resolves per feature (commonly `["get", "kind"]` against a
    /// `kind` property carrying the sprite id). The string is the
    /// final atlas key; absence here means "no icon for this layer"
    /// — the worker skips the icon emit path entirely.
    #[serde(
        rename = "icon-image",
        default,
        deserialize_with = "prop_de::opt_icon_image"
    )]
    pub icon_image: Option<Property<String>>,
    /// `icon-size` — multiplier on the sprite's intrinsic size (in
    /// pixels). Spec default is `1.0`; values below 1 shrink, above
    /// 1 enlarge. Data-driven so `interpolate(zoom, …)` can fade
    /// icons in at deeper zooms.
    #[serde(rename = "icon-size", default, deserialize_with = "prop_de::opt_f32")]
    pub icon_size: Option<Property<f32>>,
    /// `icon-rotate` — clockwise rotation in degrees applied around
    /// the icon's anchor point. Spec default is `0`. Useful for
    /// directional one-way arrows that take their angle from a
    /// feature property.
    #[serde(rename = "icon-rotate", default, deserialize_with = "prop_de::opt_f32")]
    pub icon_rotate: Option<Property<f32>>,
    /// `icon-anchor` — which corner of the icon quad pins to the
    /// feature point. Default `Center`. Data-driven shapes are
    /// silently dropped (see [`prop_de::opt_icon_anchor`]).
    #[serde(
        rename = "icon-anchor",
        default,
        deserialize_with = "prop_de::opt_icon_anchor"
    )]
    pub icon_anchor: Option<IconAnchor>,
    /// `icon-offset` — `[dx, dy]` **in pixels** applied on top of
    /// the anchor shift. Spec contrast with `text-offset` (em);
    /// authors hitting this for the first time often mis-author it
    /// in em — the parser doesn't second-guess them.
    #[serde(
        rename = "icon-offset",
        default,
        deserialize_with = "prop_de::opt_icon_offset"
    )]
    pub icon_offset: Option<[f32; 2]>,
    /// `icon-padding` — *parsed-only* until Phase 7 (collision)
    /// honours it. Shape matches `text-padding`: scalar or 4-vec,
    /// stored verbatim as `Vec<f32>`.
    #[serde(
        rename = "icon-padding",
        default,
        deserialize_with = "prop_de::opt_icon_padding"
    )]
    pub icon_padding: Option<Vec<f32>>,

    // -- placement properties (Phase 6) -------------------------------------
    //
    // `symbol-placement` toggles between the historical point path
    // (rendered through glyphon's `TextArea`) and the new line-following
    // pipeline that walks the polyline and stamps rotated glyph quads
    // (see `render::mesh::glyph_atlas` + `render::pipeline::glyph`).
    // The three companion knobs only have meaning under `line` /
    // `line-center`; they decode in either case so a styling author
    // can toggle the placement without re-authoring the other fields.

    /// `symbol-placement` — drives the per-feature placement strategy.
    /// Default is [`SymbolPlacement::Point`]. Layout properties are
    /// evaluated at the tile's zoom, so zoom functions (MapTiler's
    /// street names switch from point to line labels at z11) resolve
    /// per tile level.
    #[serde(
        rename = "symbol-placement",
        default,
        deserialize_with = "prop_de::opt_symbol_placement"
    )]
    pub symbol_placement: Option<Property<SymbolPlacement>>,
    /// `symbol-spacing` — distance (in screen pixels) between
    /// successive label anchors under `symbol-placement: line`. Spec
    /// default is `250`. Sub-symbol-spacing line segments fall back
    /// to skipping the label entirely (matches MapLibre upstream).
    #[serde(
        rename = "symbol-spacing",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub symbol_spacing: Option<Property<f32>>,
    /// `text-max-angle` — maximum allowed angle (in degrees) between
    /// successive glyph pairs along a line-placed label. Anything
    /// above this threshold drops the entire label rather than
    /// distorting the curve. Spec default is `45` degrees.
    #[serde(
        rename = "text-max-angle",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub text_max_angle: Option<Property<f32>>,
    /// `text-keep-upright` — if the line's average tangent would
    /// render the label upside-down (text running right-to-left in
    /// the screen frame), reverse the glyph order and flip each
    /// glyph by π so it reads left-to-right. Spec default is `true`.
    /// Expression form dropped per the accept-literal policy.
    #[serde(
        rename = "text-keep-upright",
        default,
        deserialize_with = "prop_de::opt_bool"
    )]
    pub text_keep_upright: Option<bool>,
    /// `text-writing-mode` — *parse-only* (vertical CJK is Phase
    /// 13+). Decodes the literal enum so styles that set it load
    /// without an `IgnoredLayerKeys` warning; the renderer ignores
    /// the value.
    #[serde(
        rename = "text-writing-mode",
        default,
        deserialize_with = "prop_de::opt_text_writing_mode"
    )]
    pub text_writing_mode: Option<TextWritingMode>,

    // -- collision properties (Phase 7) ----------------------------------------
    //
    // These knobs drive the per-frame symbol collision pass that runs
    // in `render::mesh::collision`. The pass tests every label / icon
    // candidate against an accepted-AABB list in priority order;
    // see the module-level docs there for the design rationale.
    //
    // All eight are *layer-level* properties — the collision pass
    // reads them once per layer at draw time, not per feature. Data-
    // driven shapes (where the spec allows them) decode through the
    // `Property<T>` machinery so styles that author them with
    // `["case", …]` / `["match", …]` still load. The `bool` knobs
    // accept the literal form only, following the precedent set by
    // `text-anchor` / `text-keep-upright` (data-driven booleans have
    // nowhere to land without per-feature collision state).

    /// `symbol-sort-key` — placement priority. Higher values are
    /// placed first; ties fall back to layer render order. Default
    /// `0`. Data-driven so `["get", "priority"]` lets the author put
    /// road shields above POIs based on classification.
    #[serde(
        rename = "symbol-sort-key",
        default,
        deserialize_with = "prop_de::opt_f32"
    )]
    pub symbol_sort_key: Option<Property<f32>>,
    /// `symbol-avoid-edges` — drop the candidate when its AABB centre
    /// falls inside `~text-size` of the visible map rect edge. This
    /// is a coarse approximation of upstream's "don't double-place
    /// labels straddling adjacent tile seams"; same intent, cheaper
    /// to compute, and good enough for v1. Default `false`.
    /// Expression form silently dropped per the boolean-literal
    /// policy.
    #[serde(
        rename = "symbol-avoid-edges",
        default,
        deserialize_with = "prop_de::opt_bool"
    )]
    pub symbol_avoid_edges: Option<bool>,
    /// `text-allow-overlap` — when true, the text candidate skips
    /// the overlap test entirely (and is therefore always placed).
    /// Default `false`.
    #[serde(
        rename = "text-allow-overlap",
        default,
        deserialize_with = "prop_de::opt_bool"
    )]
    pub text_allow_overlap: Option<bool>,
    /// `text-ignore-placement` — when true, the placed text does
    /// **not** push its AABB onto the accepted list, so other
    /// candidates can land on top of it. The "watermark / shield
    /// decoration" semantic. Default `false`.
    #[serde(
        rename = "text-ignore-placement",
        default,
        deserialize_with = "prop_de::opt_bool"
    )]
    pub text_ignore_placement: Option<bool>,
    /// `text-optional` — by default a text/icon pair must both fit;
    /// marking the text optional lets it drop alone while the icon
    /// persists. Default `false`.
    #[serde(
        rename = "text-optional",
        default,
        deserialize_with = "prop_de::opt_bool"
    )]
    pub text_optional: Option<bool>,
    /// `icon-allow-overlap` — twin of `text-allow-overlap` for the
    /// icon half. Default `false`.
    #[serde(
        rename = "icon-allow-overlap",
        default,
        deserialize_with = "prop_de::opt_bool"
    )]
    pub icon_allow_overlap: Option<bool>,
    /// `icon-ignore-placement` — twin of `text-ignore-placement` for
    /// the icon half. Default `false`.
    #[serde(
        rename = "icon-ignore-placement",
        default,
        deserialize_with = "prop_de::opt_bool"
    )]
    pub icon_ignore_placement: Option<bool>,
    /// `icon-optional` — twin of `text-optional` for the icon half.
    /// Default `false`.
    #[serde(
        rename = "icon-optional",
        default,
        deserialize_with = "prop_de::opt_bool"
    )]
    pub icon_optional: Option<bool>,
}

// ==============================================================================
// Per-kind layer bodies and the LayerKind enum
// ==============================================================================

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BackgroundLayer {
    pub paint: BackgroundPaint,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FillLayer {
    pub paint: FillPaint,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LineLayer {
    pub paint: LinePaint,
    pub layout: LineLayout,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SymbolLayer {
    pub paint: SymbolPaint,
    pub layout: SymbolLayout,
}

// Symbol layers carry far more properties than the other kinds. A style
// has at most a few hundred layers, so the padding costs little and saves
// a `Box` on every access.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Background(BackgroundLayer),
    Fill(FillLayer),
    Line(LineLayer),
    Symbol(SymbolLayer),
}

// ==============================================================================
// Layer — common fields plus a kind
// ==============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub id: String,
    pub source: Option<String>,
    pub source_layer: Option<String>,
    pub minzoom: Option<f64>,
    pub maxzoom: Option<f64>,
    pub filter: Option<Filter>,
    /// Layout-level `visibility` flag, lifted to the layer root so
    /// the renderer doesn't have to reach into each kind-specific
    /// layout struct. Defaults to [`Visibility::Visible`].
    pub visibility: Visibility,
    pub kind: LayerKind,
}

/// `layout.visibility` per the MapLibre spec. Applies to every
/// layer kind; the renderer treats [`Visibility::None`] as a
/// "skip this layer's draws" flag rather than tearing the layer
/// out of the document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    #[default]
    Visible,
    None,
}

// ==============================================================================
// Layer Deserialize
// ==============================================================================
//
// We can't use serde's `#[serde(tag = "type", ...)]` machinery directly
// because the outer struct also has common fields (`id`, `source`,
// `minzoom`, `filter`, ...) and the per-kind paint/layout live inside
// the same JSON object. The standard pattern is: deserialize to a
// flat helper, dispatch on the `type` discriminant, then extract paint
// / layout from the remaining JSON.
//
// Implementation strategy: deserialize into an intermediate
// `LayerRaw` struct that holds the common fields and the raw JSON for
// `paint`, `layout`, and `filter`. Then dispatch on `type` to decode
// the matching paint / layout via serde and convert `filter` via
// `Expr::from_json`.

// `LayerRaw` is also lenient — see the paint/layout comment above for
// the rationale. Real-world style documents commonly set `metadata`,
// `interactive`, or vendor extensions at the layer level; rejecting
// them would refuse the entire document.
#[derive(Deserialize)]
struct LayerRaw {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(rename = "source-layer", default)]
    source_layer: Option<String>,
    #[serde(default)]
    minzoom: Option<f64>,
    #[serde(default)]
    maxzoom: Option<f64>,
    #[serde(default)]
    filter: Option<Json>,
    #[serde(default)]
    paint: Option<Json>,
    #[serde(default)]
    layout: Option<Json>,
}

impl<'de> Deserialize<'de> for Layer {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = LayerRaw::deserialize(d)?;

        // `layout.visibility` is universal — spec says every layer
        // kind accepts it. Lift it out of the raw layout JSON before
        // dispatching to the per-kind layout decoder so the value
        // applies uniformly even for kinds (fill, background) that
        // don't otherwise model a layout struct. Unknown values are
        // a hard error: silently treating `"hidden"` as `Visible`
        // would mask typos in real-world stylesheets.
        let visibility = match raw
            .layout
            .as_ref()
            .and_then(|j| j.as_object())
            .and_then(|o| o.get("visibility"))
        {
            Some(v) => match v.as_str() {
                Some("visible") => Visibility::Visible,
                Some("none") => Visibility::None,
                _ => {
                    return Err(de::Error::custom(format!(
                        "layout.visibility must be \"visible\" or \"none\", got {v}"
                    )));
                }
            },
            None => Visibility::Visible,
        };

        // The `background` layer is special: it doesn't reference a
        // source-layer or a source. We still allow `source` /
        // `source-layer` to be absent on any kind, but for background
        // we additionally reject them so a typo doesn't silently
        // disappear into a layer kind that wouldn't have used them.
        let kind = match raw.kind.as_str() {
            "background" => {
                ensure_background_has_no_source(&raw.source, &raw.source_layer)?;
                let paint = decode_optional::<BackgroundPaint, D::Error>(raw.paint)?;
                LayerKind::Background(BackgroundLayer { paint })
            }
            "fill" => {
                let paint = decode_optional::<FillPaint, D::Error>(raw.paint)?;
                expect_no_layout(&raw.layout, "fill")?;
                LayerKind::Fill(FillLayer { paint })
            }
            "line" => {
                let paint = decode_optional::<LinePaint, D::Error>(raw.paint)?;
                let layout = decode_optional::<LineLayout, D::Error>(raw.layout)?;
                LayerKind::Line(LineLayer { paint, layout })
            }
            "symbol" => {
                let paint = decode_optional::<SymbolPaint, D::Error>(raw.paint)?;
                let layout = decode_optional::<SymbolLayout, D::Error>(raw.layout)?;
                LayerKind::Symbol(SymbolLayer { paint, layout })
            }
            other => {
                return Err(de::Error::custom(format!(
                    "unsupported layer type {other:?}"
                )));
            }
        };

        // Filters in real-world style documents are often written in
        // the pre-spec-v8 dialect (bare property names, variadic `in`,
        // `none`, the `$type` / `$id` pseudo-properties). Route
        // through the legacy converter when we recognize that shape;
        // modern expression filters pass through unchanged. See
        // [`super::legacy::convert_legacy_filter`] for the rewrite
        // rules.
        let filter = raw
            .filter
            .map(|j| {
                let converted = if super::legacy::is_legacy_filter(&j) {
                    super::legacy::convert_legacy_filter(&j)
                } else {
                    j
                };
                Expr::from_filter_json(&converted).map(Filter)
            })
            .transpose()
            .map_err(de::Error::custom)?;

        Ok(Layer {
            id: raw.id,
            source: raw.source,
            source_layer: raw.source_layer,
            minzoom: raw.minzoom,
            maxzoom: raw.maxzoom,
            filter,
            visibility,
            kind,
        })
    }
}

// `decode_optional` is generic over the caller's error type so we can
// map the (always concrete) `serde_json::Error` from `from_value` into
// whatever `Deserializer::Error` we're propagating up. This indirection
// is what makes `?` type-check inside `Layer::deserialize`.
fn decode_optional<T, E>(j: Option<Json>) -> Result<T, E>
where
    T: serde::de::DeserializeOwned + Default,
    E: de::Error,
{
    match j {
        Some(v) => serde_json::from_value(v).map_err(E::custom),
        None => Ok(T::default()),
    }
}

fn ensure_background_has_no_source<E: de::Error>(
    source: &Option<String>,
    source_layer: &Option<String>,
) -> Result<(), E> {
    // The spec says background layers MUST NOT have a source. We
    // surface this rather than silently dropping the field because
    // it's almost always a copy-paste mistake.
    if source.is_some() {
        return Err(E::custom("background layer must not have a `source`"));
    }
    if source_layer.is_some() {
        return Err(E::custom("background layer must not have a `source-layer`"));
    }
    Ok(())
}

fn expect_no_layout<E: de::Error>(layout: &Option<Json>, kind: &'static str) -> Result<(), E> {
    // The MapLibre spec puts a `visibility` field in every layer's
    // `layout` block, and many stylesheets also carry `fill-sort-key`
    // / `line-sort-key` on fill/line layers. We don't honour any of
    // them yet — the renderer treats every layer as visible and uses
    // declaration order for stacking — but rejecting the whole layer
    // because of one ignored key would lock us out of real-world
    // documents (e.g. `maptiler_style.json`). Accept and ignore.
    let Some(v) = layout else { return Ok(()) };
    if v.as_object().is_some() {
        return Ok(());
    }
    Err(E::custom(format!(
        "layer kind {kind:?} `layout` must be an object, got {v}"
    )))
}

// ==============================================================================
// Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;
    use crate::mvt::Geometry;
    use crate::mvt::Value as MvtValue;

    fn parse(j: serde_json::Value) -> Result<Layer, serde_json::Error> {
        serde_json::from_value(j)
    }

    fn empty_feature() -> Feature {
        Feature {
            id: None,
            properties: HashMap::new(),
            geometry: Geometry::Unknown,
        }
    }

    fn feature_with(props: &[(&str, MvtValue)]) -> Feature {
        let properties = props
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect::<HashMap<_, _>>();
        Feature {
            id: None,
            properties,
            geometry: Geometry::Unknown,
        }
    }

    // ---- background ----

    #[test]
    fn parses_background_layer() {
        let layer = parse(json!({
            "id": "background",
            "type": "background",
            "paint": { "background-color": "#f0eee6" }
        }))
        .unwrap();
        assert_eq!(layer.id, "background");
        let LayerKind::Background(bg) = &layer.kind else {
            panic!("expected background");
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
    }

    #[test]
    fn background_with_source_errors() {
        let err = parse(json!({
            "id": "background",
            "type": "background",
            "source": "x"
        }))
        .unwrap_err();
        assert!(err.to_string().contains("must not have a `source`"));
    }

    // ---- fill ----

    #[test]
    fn parses_fill_layer() {
        let layer = parse(json!({
            "id": "water",
            "type": "fill",
            "source": "protomaps",
            "source-layer": "water",
            "paint": { "fill-color": "#a0c8f0" }
        }))
        .unwrap();
        assert_eq!(layer.id, "water");
        assert_eq!(layer.source.as_deref(), Some("protomaps"));
        assert_eq!(layer.source_layer.as_deref(), Some("water"));
        let LayerKind::Fill(fill) = &layer.kind else {
            panic!("expected fill");
        };
        assert_eq!(
            fill.paint.fill_color,
            Some(Property::Constant(Color {
                r: 0xa0,
                g: 0xc8,
                b: 0xf0,
                a: 0xff
            }))
        );
    }

    // ---- line with filter and interpolate ----

    #[test]
    fn parses_line_layer_with_filter_and_interpolate() {
        let layer = parse(json!({
            "id": "roads",
            "type": "line",
            "source": "protomaps",
            "source-layer": "roads",
            "minzoom": 10,
            "filter": ["==", ["get", "kind"], "highway"],
            "paint": {
                "line-color": "#fff",
                "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1, 16, 6]
            }
        }))
        .unwrap();
        assert_eq!(layer.id, "roads");
        assert_eq!(layer.minzoom, Some(10.0));
        assert!(layer.filter.is_some());

        let LayerKind::Line(line) = &layer.kind else {
            panic!("expected line");
        };
        assert!(matches!(
            line.paint.line_color,
            Some(Property::Constant(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255
            }))
        ));
        assert!(matches!(
            line.paint.line_width,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));

        // Resolve the expression at zoom 13 and check the lerp.
        let width = line
            .paint
            .line_width
            .as_ref()
            .unwrap()
            .resolve(
                &EvalContext {
                    zoom: 13.0,
                    canonical: None,
                    ..Default::default()
                },
                &empty_feature(),
            )
            .unwrap();
        assert!((width - 3.5).abs() < 1e-5);

        // And the filter matches a highway feature.
        let feat = feature_with(&[("kind", MvtValue::String("highway".into()))]);
        assert!(
            layer
                .filter
                .as_ref()
                .unwrap()
                .matches(
                    &EvalContext {
                        zoom: 13.0,
                        canonical: None,
                        ..Default::default()
                    },
                    &feat
                )
                .unwrap()
        );
    }

    // ---- legacy `{token}` substitution on text-field ----

    #[test]
    fn classify_token_string_categories() {
        // Plain text — no tokens.
        assert_eq!(classify_token_string("Plain label"), TokenForm::None);
        // Single token covering the whole string — the canonical
        // legacy shape we normalize into `["get", …]`.
        assert_eq!(classify_token_string("{name}"), TokenForm::Single("name"));
        // Colon-bearing property names (common in OpenMapTiles:
        // `name:en`, `name:de`) are valid token bodies.
        assert_eq!(
            classify_token_string("{name:en}"),
            TokenForm::Single("name:en"),
        );
        // Mixed text + token — kept as a literal, warned on.
        assert_eq!(
            classify_token_string("the name is {name}"),
            TokenForm::Mixed,
        );
        // Two tokens concatenated, no surrounding text — still mixed.
        assert_eq!(classify_token_string("{a}{b}"), TokenForm::Mixed);
        // Empty `{}` doesn't count as a token at all.
        assert_eq!(classify_token_string("{}"), TokenForm::None);
        // Unterminated `{` is treated as literal text.
        assert_eq!(classify_token_string("oops {name"), TokenForm::None);
        // Stray `}` without an opener is literal text too.
        assert_eq!(classify_token_string("name}"), TokenForm::None);
    }

    #[test]
    fn single_token_text_field_normalizes_to_get_expression() {
        // The MapTiler "Dataviz" style ships several symbol layers
        // with `"text-field": "{name}"`. We treat the literal as
        // shorthand for `["get", "name"]` so the rest of the
        // expression evaluator sees a single shape.
        let layer = parse(json!({
            "id": "place-labels",
            "type": "symbol",
            "source": "protomaps",
            "source-layer": "places",
            "layout": { "text-field": "{name}" }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        let Some(Property::Expression(Expr::Get(name))) = &sym.layout.text_field else {
            panic!("expected text-field to normalize to Get, got {:?}", sym.layout.text_field);
        };
        assert_eq!(name, "name");
    }

    #[test]
    fn multi_token_text_field_stays_literal() {
        // Mixed text + token isn't a shape we honor — the field
        // round-trips as a literal constant. (The matching
        // `MultiTokenTextField` warning is asserted in the `mod.rs`
        // tests, where the warning collector is in scope.)
        let layer = parse(json!({
            "id": "place-labels",
            "type": "symbol",
            "source": "protomaps",
            "source-layer": "places",
            "layout": { "text-field": "the name is {name}" }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert_eq!(
            sym.layout.text_field,
            Some(Property::Constant("the name is {name}".to_string())),
        );
    }

    #[test]
    fn legacy_zoom_stops_text_field_expands_tokens() {
        // The MapLibre demo's `countries-label` layer authors
        // `text-field` as a zoom-stops legacy function whose stop
        // outputs are `{TOKEN}` literals (`{ABBREV}` at low zoom,
        // `{NAME}` at higher zoom). The legacy converter expands these
        // to `["to-string", ["get", "TOKEN"]]` once we route through
        // it with `tokens: true`. Without that, the literal `{NAME}`
        // would render as text (the bug we just fixed).
        let layer = parse(json!({
            "id": "countries-label",
            "type": "symbol",
            "source": "maplibre",
            "source-layer": "countries",
            "layout": {
                "text-field": {
                    "stops": [[2, "{ABBREV}"], [4, "{NAME}"]]
                }
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        let Some(Property::Expression(expr)) = &sym.layout.text_field else {
            panic!(
                "expected text-field to land as an expression, got {:?}",
                sym.layout.text_field,
            );
        };
        // At zoom 2 we sit on the first stop and read `ABBREV`; at
        // zoom 4 the curve advances to `NAME`. We feed both properties
        // so the test pins which token wins per zoom.
        let feat = feature_with(&[
            ("ABBREV", MvtValue::String("FR".into())),
            ("NAME", MvtValue::String("France".into())),
        ]);
        let low = expr
            .eval(
                &EvalContext {
                    zoom: 2.0,
                    canonical: None,
                    ..Default::default()
                },
                &feat,
            )
            .unwrap();
        let high = expr
            .eval(
                &EvalContext {
                    zoom: 4.0,
                    canonical: None,
                    ..Default::default()
                },
                &feat,
            )
            .unwrap();
        assert_eq!(low, Value::String("FR".into()));
        assert_eq!(high, Value::String("France".into()));
    }

    #[test]
    fn legacy_single_stop_zoom_function_collapses_to_constant() {
        // Single-stop legacy functions describe a constant — there's
        // no second breakpoint to advance to. Modern `step` rejects
        // the `(input, default)` 2-arg shape, so the converter has to
        // emit the wrapped output directly. The tilekiln-shortbread
        // `colorful.json` style hits this with
        // `"text-size": { "stops": [[14, 12]] }`.
        let layer = parse(json!({
            "id": "place-labels",
            "type": "symbol",
            "source": "x",
            "source-layer": "y",
            "layout": {
                "text-field": "x",
                "text-font": ["Noto Sans Regular"],
                "text-size": { "stops": [[14, 12]] }
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        // The single-stop output is constant 12 at every zoom; we
        // sample two zooms to pin that down.
        let Some(Property::Expression(expr)) = &sym.layout.text_size else {
            panic!("expected text-size expression, got {:?}", sym.layout.text_size);
        };
        for z in [0.0, 14.0, 22.0] {
            let v = expr
                .eval(
                    &EvalContext {
                        zoom: z,
                        canonical: None,
                        ..Default::default()
                    },
                    &feature_with(&[]),
                )
                .unwrap();
            assert_eq!(v, Value::Number(12.0), "zoom {z}");
        }
    }

    // ---- symbol with text-field expression ----

    #[test]
    fn parses_symbol_layer() {
        let layer = parse(json!({
            "id": "place-labels",
            "type": "symbol",
            "source": "protomaps",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-font": ["Noto Sans Regular"],
                "text-size": 12
            },
            "paint": { "text-color": "#333" }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        // text-field is an expression
        assert!(matches!(
            sym.layout.text_field,
            Some(Property::Expression(Expr::Get(_)))
        ));
        // text-font is a literal list
        assert_eq!(
            sym.layout.text_font.as_deref(),
            Some(&["Noto Sans Regular".to_string()][..])
        );
        // text-size is a constant number
        assert_eq!(sym.layout.text_size, Some(Property::Constant(12.0)));
        // text-color is a constant color
        assert_eq!(
            sym.paint.text_color,
            Some(Property::Constant(Color {
                r: 0x33,
                g: 0x33,
                b: 0x33,
                a: 0xff
            }))
        );
    }

    // ---- error surfaces ----

    #[test]
    fn rejects_unknown_layer_type() {
        let err = parse(json!({ "id": "x", "type": "raster" })).unwrap_err();
        assert!(err.to_string().contains("unsupported layer type"));
    }

    #[test]
    fn ignores_unknown_paint_property() {
        // Real-world styles set many paint properties we don't model
        // yet (e.g. `fill-outline-color`, `fill-pattern`). Loading
        // should succeed and just drop them silently — the alternative
        // (refusing the whole document) makes `--style PATH` useless
        // on any production style.
        let layer = parse(json!({
            "id": "x", "type": "fill",
            "paint": { "fill-color": "#fff", "fill-outline-color": "#000" }
        }))
        .unwrap();
        let LayerKind::Fill(fill) = &layer.kind else {
            panic!("expected fill layer");
        };
        assert!(fill.paint.fill_color.is_some());
    }

    // ==============================================================================
    // Opacity (Phase 2): one knob per layer kind, both literal and expression
    // ==============================================================================
    //
    // We test each opacity field on every kind that exposes it. The
    // wiring is identical (it's the same `prop_de::opt_f32` adapter
    // each `Property<f32>` already uses for `line-width` / `text-size`)
    // so a single literal + single expression spot-check per field is
    // enough to lock in the field name and default-when-absent shape.

    #[test]
    fn parses_fill_opacity_literal_and_expression() {
        let lit = parse(json!({
            "id": "x", "type": "fill",
            "paint": { "fill-color": "#fff", "fill-opacity": 0.5 }
        }))
        .unwrap();
        let LayerKind::Fill(fill) = &lit.kind else {
            panic!("expected fill");
        };
        assert_eq!(fill.paint.fill_opacity, Some(Property::Constant(0.5)));

        let expr = parse(json!({
            "id": "x", "type": "fill",
            "paint": {
                "fill-color": "#fff",
                "fill-opacity": ["interpolate", ["linear"], ["zoom"], 5, 0, 7, 1]
            }
        }))
        .unwrap();
        let LayerKind::Fill(fill) = &expr.kind else {
            panic!("expected fill");
        };
        assert!(matches!(
            fill.paint.fill_opacity,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));

        // Absent: the renderer's default-of-1.0 lives on its side, so
        // we just confirm the field stays `None` when omitted.
        let absent = parse(json!({
            "id": "x", "type": "fill", "paint": { "fill-color": "#fff" }
        }))
        .unwrap();
        let LayerKind::Fill(fill) = &absent.kind else {
            panic!("expected fill");
        };
        assert_eq!(fill.paint.fill_opacity, None);
    }

    #[test]
    fn parses_line_opacity_literal_and_expression() {
        let lit = parse(json!({
            "id": "x", "type": "line",
            "source-layer": "roads",
            "paint": { "line-color": "#fff", "line-opacity": 0.25 }
        }))
        .unwrap();
        let LayerKind::Line(line) = &lit.kind else {
            panic!("expected line");
        };
        assert_eq!(line.paint.line_opacity, Some(Property::Constant(0.25)));

        let expr = parse(json!({
            "id": "x", "type": "line",
            "source-layer": "roads",
            "paint": {
                "line-color": "#fff",
                "line-opacity": ["interpolate", ["linear"], ["zoom"], 8, 0, 12, 1]
            }
        }))
        .unwrap();
        let LayerKind::Line(line) = &expr.kind else {
            panic!("expected line");
        };
        assert!(matches!(
            line.paint.line_opacity,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
    }

    #[test]
    fn parses_text_and_icon_opacity_on_symbol() {
        // `icon-opacity` is parse-only (icons land in Phase 5) but the
        // parser must still accept it; verifying it surfaces on the
        // paint struct guarantees the wiring is ready.
        let layer = parse(json!({
            "id": "x", "type": "symbol",
            "source-layer": "places",
            "layout": { "text-field": ["get", "name"] },
            "paint": {
                "text-color": "#333",
                "text-opacity": 0.8,
                "icon-opacity": ["interpolate", ["linear"], ["zoom"], 10, 0, 14, 1]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.paint.text_opacity, Some(Property::Constant(0.8)));
        assert!(matches!(
            sym.paint.icon_opacity,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
    }

    #[test]
    fn parses_background_opacity() {
        let layer = parse(json!({
            "id": "bg", "type": "background",
            "paint": { "background-color": "#222", "background-opacity": 0.5 }
        }))
        .unwrap();
        let LayerKind::Background(bg) = &layer.kind else {
            panic!("expected background");
        };
        assert_eq!(bg.paint.background_opacity, Some(Property::Constant(0.5)));
    }

    #[test]
    fn ignores_unknown_top_level_field_on_layer() {
        // Same rationale: `metadata`, `interactive`, vendor extensions
        // routinely appear at the layer level. Loading should succeed.
        let layer = parse(json!({
            "id": "x", "type": "fill", "metadata": {"vendor": "y"}
        }))
        .unwrap();
        assert_eq!(layer.id, "x");
    }

    // ---- Property<T> direct ----

    #[test]
    fn property_constant_color() {
        let p = property_from_json::<Color>(json!("#fff")).unwrap();
        assert_eq!(
            p,
            Property::Constant(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255
            })
        );
    }

    #[test]
    fn property_expression_dispatch() {
        let p: Property<f32> =
            property_from_json::<f32>(json!(["interpolate", ["linear"], ["zoom"], 0, 0, 1, 1]))
                .unwrap();
        assert!(matches!(p, Property::Expression(Expr::Interpolate { .. })));
    }

    // ---- line layout: caps, joins, limits ----

    #[test]
    fn parses_line_layout_caps_and_joins() {
        let layer = parse(json!({
            "id": "roads",
            "type": "line",
            "source": "p",
            "source-layer": "roads",
            "layout": { "line-cap": "round", "line-join": "bevel" },
            "paint": {
                "line-color": "#fff",
                "line-miter-limit": 4,
                "line-round-limit": 1.1
            }
        }))
        .unwrap();
        let LayerKind::Line(line) = &layer.kind else {
            panic!("expected line");
        };
        assert_eq!(line.layout.line_cap, Some(LineCap::Round));
        assert_eq!(line.layout.line_join, Some(LineJoin::Bevel));
        assert_eq!(line.paint.line_miter_limit, Some(Property::Constant(4.0)));
        assert_eq!(line.paint.line_round_limit, Some(Property::Constant(1.1)));
    }

    #[test]
    fn line_cap_join_expression_form_is_dropped() {
        // Data-driven `line-cap` / `line-join` are spec-legal but rare;
        // we silently drop the expression form so the layer still loads
        // and falls back to the spec default. This mirrors how we
        // handle `text-font`.
        let layer = parse(json!({
            "id": "roads",
            "type": "line",
            "source": "p",
            "source-layer": "roads",
            "layout": {
                "line-cap": ["step", ["zoom"], "butt", 14, "round"],
                "line-join": ["literal", "miter"]
            }
        }))
        .unwrap();
        let LayerKind::Line(line) = &layer.kind else {
            panic!("expected line");
        };
        assert_eq!(line.layout.line_cap, None);
        assert_eq!(line.layout.line_join, None);
    }

    // ==============================================================================
    // Phase 3 paint: dasharray, gap-width, offset
    // ==============================================================================
    //
    // Three independent properties on `LinePaint`. `line-dasharray`
    // mirrors `text-font` (literal-only — expressions accepted then
    // dropped). `line-gap-width` and `line-offset` are scalar
    // `Property<f32>` and accept the usual literal-or-expression
    // dispatch via `prop_de::opt_f32`.

    #[test]
    fn parses_line_dasharray_literal() {
        let layer = parse(json!({
            "id": "rail",
            "type": "line",
            "source": "p",
            "source-layer": "transportation",
            "paint": { "line-color": "#000", "line-dasharray": [3, 2] }
        }))
        .unwrap();
        let LayerKind::Line(line) = &layer.kind else {
            panic!("expected line");
        };
        assert_eq!(line.paint.line_dasharray, Some(vec![3.0, 2.0]));
    }

    #[test]
    fn line_dasharray_expression_form_is_dropped() {
        // A `step`-driven dasharray is rare but spec-legal; we silently
        // drop the expression form because the renderer's dash uniform
        // is uploaded per-frame, not per-feature. The layer still loads
        // and falls back to a solid stroke.
        let layer = parse(json!({
            "id": "rail",
            "type": "line",
            "source": "p",
            "source-layer": "transportation",
            "paint": {
                "line-color": "#000",
                "line-dasharray": ["step", ["zoom"], ["literal", [2, 2]], 14, ["literal", [4, 2]]]
            }
        }))
        .unwrap();
        let LayerKind::Line(line) = &layer.kind else {
            panic!("expected line");
        };
        assert_eq!(line.paint.line_dasharray, None);
    }

    #[test]
    fn parses_line_gap_width_literal_and_expression() {
        let lit = parse(json!({
            "id": "x", "type": "line",
            "source-layer": "roads",
            "paint": { "line-color": "#000", "line-gap-width": 4 }
        }))
        .unwrap();
        let LayerKind::Line(line) = &lit.kind else {
            panic!("expected line");
        };
        assert_eq!(line.paint.line_gap_width, Some(Property::Constant(4.0)));

        let expr = parse(json!({
            "id": "x", "type": "line",
            "source-layer": "roads",
            "paint": {
                "line-color": "#000",
                "line-gap-width": ["interpolate", ["linear"], ["zoom"], 10, 0, 16, 8]
            }
        }))
        .unwrap();
        let LayerKind::Line(line) = &expr.kind else {
            panic!("expected line");
        };
        assert!(matches!(
            line.paint.line_gap_width,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
    }

    #[test]
    fn parses_line_offset_literal_with_negative_value() {
        // Negative offsets are spec-legal: they shift the line the
        // opposite way from the positive convention. The parser must
        // round-trip the sign rather than clamping to zero.
        let layer = parse(json!({
            "id": "x", "type": "line",
            "source-layer": "roads",
            "paint": { "line-color": "#000", "line-offset": -3 }
        }))
        .unwrap();
        let LayerKind::Line(line) = &layer.kind else {
            panic!("expected line");
        };
        assert_eq!(line.paint.line_offset, Some(Property::Constant(-3.0)));
    }

    #[test]
    fn property_resolve_constant_color() {
        let p = Property::Constant(Color {
            r: 1,
            g: 2,
            b: 3,
            a: 4,
        });
        let c = p
            .resolve(
                &EvalContext {
                    zoom: 0.0,
                    canonical: None,
                    ..Default::default()
                },
                &empty_feature(),
            )
            .unwrap();
        assert_eq!(
            c,
            Color {
                r: 1,
                g: 2,
                b: 3,
                a: 4
            }
        );
    }

    // ==============================================================================
    // Phase 4: symbol text polish (halo / anchor / offset / wrap / spacing /
    // transform / justify / padding)
    // ==============================================================================
    //
    // The new fields land in two camps:
    //
    //   * Scalar / colour properties (`text-halo-*`, `text-max-width`,
    //     `text-letter-spacing`, `text-line-height`) go through the
    //     standard `prop_de::opt_f32` / `opt_color` dispatch — same
    //     literal-or-expression handling as `text-size`. One spot-check
    //     per field is enough.
    //   * Enum properties (`text-anchor`, `text-transform`,
    //     `text-justify`) follow the `line-cap` / `line-join`
    //     precedent: literal accepted, expression silently dropped.
    //   * Array-shaped properties (`text-offset`, `text-padding`)
    //     accept a 2-vec / scalar-or-4-vec; expression form dropped.

    #[test]
    fn parses_text_halo_paint_literals() {
        let layer = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": { "text-field": ["get", "name"] },
            "paint": {
                "text-color": "#222",
                "text-halo-color": "#ffffff",
                "text-halo-width": 1.5,
                "text-halo-blur": 0.5
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert_eq!(
            sym.paint.text_halo_color,
            Some(Property::Constant(Color {
                r: 0xff,
                g: 0xff,
                b: 0xff,
                a: 0xff
            }))
        );
        assert_eq!(
            sym.paint.text_halo_width,
            Some(Property::Constant(1.5))
        );
        assert_eq!(sym.paint.text_halo_blur, Some(Property::Constant(0.5)));
    }

    #[test]
    fn parses_text_halo_paint_expressions() {
        // Halo width / blur are scalar properties — the standard
        // expression dispatch must accept `interpolate` shapes so
        // production styles that fade the halo across zoom load.
        let layer = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": { "text-field": ["get", "name"] },
            "paint": {
                "text-halo-color": ["interpolate", ["linear"], ["zoom"], 8, "#000", 12, "#fff"],
                "text-halo-width": ["interpolate", ["linear"], ["zoom"], 8, 0, 12, 2],
                "text-halo-blur": ["interpolate", ["linear"], ["zoom"], 8, 0, 12, 1]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert!(matches!(
            sym.paint.text_halo_color,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
        assert!(matches!(
            sym.paint.text_halo_width,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
        assert!(matches!(
            sym.paint.text_halo_blur,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
    }

    #[test]
    fn parses_text_anchor_transform_justify_literals() {
        // All three enum-valued layout properties decode via serde's
        // `rename_all` machinery. We test each variant string round-
        // trips through the literal path — the kebab-case anchors
        // (`top-left` etc.) are the spec-sensitive shape because
        // `serde(rename_all = "kebab-case")` is what makes them match.
        let layer = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-anchor": "top-left",
                "text-transform": "uppercase",
                "text-justify": "right"
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_anchor, Some(TextAnchor::TopLeft));
        assert_eq!(sym.layout.text_transform, Some(TextTransform::Uppercase));
        assert_eq!(sym.layout.text_justify, Some(TextJustify::Right));
    }

    #[test]
    fn text_anchor_transform_justify_expression_form_is_dropped() {
        // Same accept-and-drop policy as `line-cap` / `text-font`. The
        // layer still loads cleanly so the rest of the document
        // renders; the renderer falls back to the spec default.
        let layer = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-anchor": ["step", ["zoom"], "center", 14, "top"],
                "text-transform": ["literal", "uppercase"],
                "text-justify": ["step", ["zoom"], "left", 14, "center"]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_anchor, None);
        assert_eq!(sym.layout.text_transform, None);
        assert_eq!(sym.layout.text_justify, None);
    }

    #[test]
    fn parses_text_offset_literal_and_drops_expression() {
        // Two-element numeric array is the literal form.
        let lit = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-offset": [0.5, -1.2]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_offset, Some([0.5, -1.2]));

        // Expression form (`literal` wrap) is dropped — same precedent
        // as `text-font` / `line-dasharray`. The renderer falls back
        // to the spec default of `[0, 0]`.
        let dropped = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-offset": ["literal", [0.5, -1.2]]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &dropped.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_offset, None);
    }

    #[test]
    fn parses_text_max_width_letter_spacing_line_height_literal_and_expression() {
        // `text-max-width` / `text-letter-spacing` / `text-line-height`
        // are scalar `Property<f32>` — the dispatch is the standard
        // `prop_de::opt_f32`. One literal-and-expression test pins the
        // wiring; the renderer-side test handles end-to-end behaviour.
        let lit = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-max-width": 8,
                "text-letter-spacing": 0.1,
                "text-line-height": 1.4
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_max_width, Some(Property::Constant(8.0)));
        assert_eq!(
            sym.layout.text_letter_spacing,
            Some(Property::Constant(0.1))
        );
        assert_eq!(
            sym.layout.text_line_height,
            Some(Property::Constant(1.4))
        );

        let expr = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-max-width": ["interpolate", ["linear"], ["zoom"], 8, 6, 14, 12]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &expr.kind else {
            panic!("expected symbol");
        };
        assert!(matches!(
            sym.layout.text_max_width,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
    }

    #[test]
    fn parses_text_padding_scalar_and_array() {
        // `text-padding` is parse-only (collision is Phase 7). Both
        // shapes the spec allows decode into `Vec<f32>` so the future
        // collision pass can pattern-match on length.
        let scalar = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": { "text-field": ["get", "name"], "text-padding": 2 }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &scalar.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_padding, Some(vec![2.0]));

        let array = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-padding": [4, 2, 4, 2]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &array.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_padding, Some(vec![4.0, 2.0, 4.0, 2.0]));
    }

    // ==============================================================================
    // Phase 5: icon-* properties on SymbolLayer
    // ==============================================================================
    //
    // Scalar / colour properties (`icon-size`, `icon-rotate`,
    // `icon-color`, `icon-halo-*`) follow the standard `prop_de::opt_f32`
    // / `opt_color` dispatch — one literal + one expression spot-check
    // per field pins the field name and the expression path. Enum
    // (`icon-anchor`) and array (`icon-offset`) properties follow the
    // accept-literal-drop-expression precedent from `text-anchor` /
    // `text-offset`. `icon-image` is `Property<String>` *without* the
    // `{token}` shorthand `text-field` carries — verified explicitly
    // because the spec quirk is easy to miss.

    #[test]
    fn parses_icon_image_literal_and_expression() {
        // Literal sprite name: round-trips as a Constant. The
        // `{token}` shorthand `text-field` honours does NOT apply
        // here — `"airport_11"` is a literal sprite key.
        let lit = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": { "icon-image": "airport_11" }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(
            sym.layout.icon_image,
            Some(Property::Constant("airport_11".to_string()))
        );

        // Expression form: `["get", "kind"]` is by far the most common
        // shape in real-world styles (Protomaps, MapTiler).
        let expr = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": { "icon-image": ["get", "kind"] }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &expr.kind else {
            panic!("expected symbol");
        };
        assert!(matches!(
            sym.layout.icon_image,
            Some(Property::Expression(Expr::Get(ref n))) if n == "kind"
        ));
    }

    #[test]
    fn icon_image_does_not_normalize_legacy_tokens() {
        // Critical regression: `text-field: "{name}"` becomes
        // `["get", "name"]` but `icon-image: "{name}"` must stay as
        // a *literal* "{name}" sprite name (which will simply miss
        // in the atlas, matching upstream MapLibre's behaviour).
        let layer = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": { "icon-image": "{name}" }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert_eq!(
            sym.layout.icon_image,
            Some(Property::Constant("{name}".to_string())),
        );
    }

    #[test]
    fn parses_icon_size_rotate_color_literal_and_expression() {
        let lit = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": { "icon-image": "x", "icon-size": 1.5, "icon-rotate": 45 },
            "paint":  { "icon-color": "#ff0000" }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.icon_size, Some(Property::Constant(1.5)));
        assert_eq!(sym.layout.icon_rotate, Some(Property::Constant(45.0)));
        assert_eq!(
            sym.paint.icon_color,
            Some(Property::Constant(Color {
                r: 0xff,
                g: 0x00,
                b: 0x00,
                a: 0xff
            }))
        );

        let expr = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "icon-image": "x",
                "icon-size": ["interpolate", ["linear"], ["zoom"], 10, 0.5, 16, 1.5],
                "icon-rotate": ["get", "angle"]
            },
            "paint": {
                "icon-color": ["interpolate", ["linear"], ["zoom"], 8, "#000", 14, "#fff"]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &expr.kind else {
            panic!("expected symbol");
        };
        assert!(matches!(
            sym.layout.icon_size,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
        assert!(matches!(
            sym.layout.icon_rotate,
            Some(Property::Expression(Expr::Get(_)))
        ));
        assert!(matches!(
            sym.paint.icon_color,
            Some(Property::Expression(Expr::Interpolate { .. }))
        ));
    }

    #[test]
    fn parses_icon_anchor_literal_and_drops_expression() {
        let lit = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": { "icon-image": "x", "icon-anchor": "top-left" }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.icon_anchor, Some(IconAnchor::TopLeft));

        // Expression form silently drops — same accept-and-drop
        // precedent as `text-anchor`.
        let dropped = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "icon-image": "x",
                "icon-anchor": ["step", ["zoom"], "center", 14, "top"]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &dropped.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.icon_anchor, None);
    }

    #[test]
    fn parses_icon_offset_literal_and_drops_expression() {
        let lit = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": { "icon-image": "x", "icon-offset": [4.0, -3.0] }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        // Pixels, not em — see `opt_icon_offset` docs.
        assert_eq!(sym.layout.icon_offset, Some([4.0, -3.0]));

        let dropped = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "icon-image": "x",
                "icon-offset": ["literal", [4.0, -3.0]]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &dropped.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.icon_offset, None);
    }

    #[test]
    fn parses_icon_halo_paint_and_padding() {
        // Halo paint properties are parse-only this phase but the
        // fields exist so styles parse cleanly.
        let layer = parse(json!({
            "id": "icons", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "icon-image": "x",
                "icon-padding": [2, 4, 2, 4]
            },
            "paint": {
                "icon-halo-color": "#fff",
                "icon-halo-width": 1.0,
                "icon-halo-blur": 0.5
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert!(sym.paint.icon_halo_color.is_some());
        assert_eq!(sym.paint.icon_halo_width, Some(Property::Constant(1.0)));
        assert_eq!(sym.paint.icon_halo_blur, Some(Property::Constant(0.5)));
        assert_eq!(sym.layout.icon_padding, Some(vec![2.0, 4.0, 2.0, 4.0]));
    }

    #[test]
    fn parses_text_radial_offset_literal_and_expression() {
        // Parsed-only — accepted so styles that *only* set radial
        // (rare but spec-legal) load cleanly. The renderer doesn't
        // read this field today.
        let layer = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": {
                "text-field": ["get", "name"],
                "text-radial-offset": 1.5
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert_eq!(
            sym.layout.text_radial_offset,
            Some(Property::Constant(1.5))
        );
    }

    // ==========================================================================
    // Phase 6: line-following label properties
    // ==========================================================================
    //
    // Five new fields land on `SymbolLayout`:
    //
    //   - `symbol-placement` (enum)             — literal + expression-drop
    //   - `symbol-spacing`   (Property<f32>)    — literal + expression
    //   - `text-max-angle`   (Property<f32>)    — literal + expression
    //   - `text-keep-upright` (bool)            — literal + expression-drop
    //   - `text-writing-mode` (enum, parse-only)— literal + expression-drop
    //
    // One literal-form test per field pins the spelling; one expression
    // spot-check per field demonstrates either pass-through (for the
    // `Property<f32>` shapes) or accept-and-drop (for the enums and the
    // bool).

    #[test]
    fn parses_symbol_placement_literals_and_zoom_functions() {
        use crate::style::expression::EvalContext;
        let placement_at = |layout: serde_json::Value, zoom: f64| {
            let layer = parse(json!({
                "id": "labels", "type": "symbol",
                "source-layer": "roads",
                "layout": layout
            }))
            .unwrap();
            let LayerKind::Symbol(sym) = &layer.kind else {
                panic!("expected symbol");
            };
            let feature = crate::mvt::Feature {
                id: None,
                properties: Default::default(),
                geometry: crate::mvt::Geometry::Unknown,
            };
            sym.layout.symbol_placement.as_ref().map(|p| {
                p.resolve(&EvalContext { zoom, ..Default::default() }, &feature)
                    .expect("placement resolves")
            })
        };

        for (json_v, expected) in [
            ("point", SymbolPlacement::Point),
            ("line", SymbolPlacement::Line),
            ("line-center", SymbolPlacement::LineCenter),
        ] {
            assert_eq!(placement_at(json!({ "symbol-placement": json_v }), 12.0), Some(expected));
        }
        assert_eq!(placement_at(json!({}), 12.0), None);

        // MapTiler's street-name layers switch from point to line labels
        // at z11 with a legacy zoom function.
        let stepped = json!({ "symbol-placement": { "base": 1, "stops": [[10, "point"], [11, "line"]] } });
        assert_eq!(placement_at(stepped.clone(), 10.5), Some(SymbolPlacement::Point));
        assert_eq!(placement_at(stepped, 11.0), Some(SymbolPlacement::Line));
    }

    #[test]
    fn parses_symbol_spacing_literal_and_expression() {
        // Literal: a `Property::Constant(f32)` lands on the field.
        let lit = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "roads",
            "layout": {
                "symbol-placement": "line",
                "symbol-spacing": 180
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.symbol_spacing, Some(Property::Constant(180.0)));

        // Expression form: `interpolate` resolves through the standard
        // `opt_f32` path.
        let expr = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "roads",
            "layout": {
                "symbol-placement": "line",
                "symbol-spacing": ["interpolate", ["linear"], ["zoom"], 10, 100, 14, 250]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &expr.kind else {
            panic!("expected symbol");
        };
        assert!(matches!(
            sym.layout.symbol_spacing,
            Some(Property::Expression(_))
        ));
    }

    #[test]
    fn parses_text_max_angle_literal_and_expression() {
        let lit = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "roads",
            "layout": {
                "symbol-placement": "line",
                "text-max-angle": 30
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_max_angle, Some(Property::Constant(30.0)));

        let expr = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "roads",
            "layout": {
                "symbol-placement": "line",
                "text-max-angle": ["interpolate", ["linear"], ["zoom"], 6, 22, 14, 45]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &expr.kind else {
            panic!("expected symbol");
        };
        assert!(matches!(
            sym.layout.text_max_angle,
            Some(Property::Expression(_))
        ));
    }

    #[test]
    fn parses_text_keep_upright_literal_and_drops_expression() {
        let lit = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "roads",
            "layout": {
                "symbol-placement": "line",
                "text-keep-upright": false
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_keep_upright, Some(false));

        // Expression form silently dropped (data-driven keep-upright
        // would require a per-feature flip pass; same trade-off as
        // text-anchor).
        let dropped = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "roads",
            "layout": {
                "symbol-placement": "line",
                "text-keep-upright": ["get", "two_way"]
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &dropped.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_keep_upright, None);
    }

    #[test]
    fn parses_text_writing_mode_literal_and_drops_expression() {
        // Parse-only: every literal value lands on the field for
        // future use, but the renderer ignores it (vertical CJK is
        // Phase 13+).
        for (json_v, expected) in [
            ("horizontal", TextWritingMode::Horizontal),
            ("vertical", TextWritingMode::Vertical),
        ] {
            let layer = parse(json!({
                "id": "labels", "type": "symbol",
                "source-layer": "places",
                "layout": { "text-writing-mode": json_v }
            }))
            .unwrap();
            let LayerKind::Symbol(sym) = &layer.kind else {
                panic!("expected symbol");
            };
            assert_eq!(sym.layout.text_writing_mode, Some(expected));
        }

        let dropped = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "places",
            "layout": { "text-writing-mode": ["get", "script"] }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &dropped.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_writing_mode, None);
    }

    // ==========================================================================
    // Phase 7: collision-related layout properties
    // ==========================================================================
    //
    // Eight new layout fields land on `SymbolLayout`:
    //
    //   - `symbol-sort-key`         (`Property<f32>`, literal + expression)
    //   - `symbol-avoid-edges`      (`Option<bool>`, literal-only)
    //   - `text-allow-overlap`      (`Option<bool>`, literal-only)
    //   - `text-ignore-placement`   (`Option<bool>`, literal-only)
    //   - `text-optional`           (`Option<bool>`, literal-only)
    //   - `icon-allow-overlap`      (`Option<bool>`, literal-only)
    //   - `icon-ignore-placement`   (`Option<bool>`, literal-only)
    //   - `icon-optional`           (`Option<bool>`, literal-only)
    //
    // One literal-form assert per field pins the spelling and decoded
    // shape; the bool group also includes one expression-drop spot
    // check (the sort-key test exercises both forms because it's the
    // only one with full expression support).

    #[test]
    fn parses_symbol_sort_key_literal_and_expression() {
        let lit = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "pois",
            "layout": { "symbol-sort-key": 5.0 }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &lit.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.symbol_sort_key, Some(Property::Constant(5.0)));

        let expr = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "pois",
            "layout": { "symbol-sort-key": ["get", "priority"] }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &expr.kind else {
            panic!("expected symbol");
        };
        assert!(matches!(
            sym.layout.symbol_sort_key,
            Some(Property::Expression(_))
        ));
    }

    #[test]
    fn parses_collision_bool_flags() {
        // One test that exercises every literal-form bool knob in a
        // single layout block — the parser is purely structural so a
        // single decode lands each field on its expected side.
        let layer = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "pois",
            "layout": {
                "symbol-avoid-edges": true,
                "text-allow-overlap": true,
                "text-ignore-placement": false,
                "text-optional": true,
                "icon-allow-overlap": false,
                "icon-ignore-placement": true,
                "icon-optional": true
            }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &layer.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.symbol_avoid_edges, Some(true));
        assert_eq!(sym.layout.text_allow_overlap, Some(true));
        assert_eq!(sym.layout.text_ignore_placement, Some(false));
        assert_eq!(sym.layout.text_optional, Some(true));
        assert_eq!(sym.layout.icon_allow_overlap, Some(false));
        assert_eq!(sym.layout.icon_ignore_placement, Some(true));
        assert_eq!(sym.layout.icon_optional, Some(true));
    }

    #[test]
    fn collision_bool_flags_drop_expression_form() {
        // Spot-check: data-driven booleans follow the
        // `text-keep-upright` precedent (silently drop). One field
        // is enough to lock the policy.
        let dropped = parse(json!({
            "id": "labels", "type": "symbol",
            "source-layer": "pois",
            "layout": { "text-allow-overlap": ["get", "important"] }
        }))
        .unwrap();
        let LayerKind::Symbol(sym) = &dropped.kind else {
            panic!("expected symbol");
        };
        assert_eq!(sym.layout.text_allow_overlap, None);
    }

    // ==============================================================================
    // Phase 8: fill-pattern, background-pattern, fill/line-translate
    // ==============================================================================
    //
    // The new properties land in two camps:
    //
    //   * `fill-pattern` / `background-pattern` / `line-pattern` are
    //     `Property<String>` (sprite name). Same dispatch as
    //     `icon-image` — literal-or-expression — but without the
    //     `{token}` shorthand rewrite. `line-pattern` is *parsed-only*
    //     this phase (see the TODO breadcrumb on `LinePaint`).
    //   * `fill-translate` / `line-translate` are `[dx, dy]` in
    //     pixels — same accept-literal / drop-expression policy as
    //     `text-offset` / `icon-offset`.
    //   * `fill-translate-anchor` / `line-translate-anchor` are
    //     enum literals (`Map` / `Viewport`) — same policy as
    //     `text-anchor`.
    //   * `fill-antialias` is a parse-only bool.

    #[test]
    fn parses_fill_pattern_literal_and_expression() {
        let lit = parse(json!({
            "id": "water", "type": "fill",
            "source": "x", "source-layer": "water",
            "paint": { "fill-pattern": "water_hatch" }
        }))
        .unwrap();
        let LayerKind::Fill(f) = &lit.kind else {
            panic!("expected fill");
        };
        assert_eq!(
            f.paint.fill_pattern,
            Some(Property::Constant("water_hatch".to_string()))
        );

        let expr = parse(json!({
            "id": "landuse", "type": "fill",
            "source": "x", "source-layer": "landuse",
            "paint": { "fill-pattern": ["match", ["get", "kind"],
                "forest", "forest_stipple",
                "sand", "sand_stipple",
                "default_pattern"]
            }
        }))
        .unwrap();
        let LayerKind::Fill(f) = &expr.kind else {
            panic!("expected fill");
        };
        assert!(matches!(
            f.paint.fill_pattern,
            Some(Property::Expression(_))
        ));
    }

    #[test]
    fn parses_background_pattern_literal() {
        let layer = parse(json!({
            "id": "bg", "type": "background",
            "paint": { "background-pattern": "paper_tex" }
        }))
        .unwrap();
        let LayerKind::Background(bg) = &layer.kind else {
            panic!("expected background");
        };
        assert_eq!(
            bg.paint.background_pattern,
            Some(Property::Constant("paper_tex".to_string()))
        );
    }

    #[test]
    fn parses_fill_translate_and_anchor_literals() {
        let layer = parse(json!({
            "id": "shadow", "type": "fill",
            "source": "x", "source-layer": "landuse",
            "paint": {
                "fill-color": "#000",
                "fill-translate": [2.0, -3.0],
                "fill-translate-anchor": "viewport"
            }
        }))
        .unwrap();
        let LayerKind::Fill(f) = &layer.kind else {
            panic!("expected fill");
        };
        assert_eq!(f.paint.fill_translate, Some([2.0, -3.0]));
        assert_eq!(
            f.paint.fill_translate_anchor,
            Some(TranslateAnchor::Viewport)
        );
    }

    #[test]
    fn fill_translate_anchor_defaults_to_map_when_absent() {
        // The renderer reads `unwrap_or_default()` from the parser
        // output; `Default` for `TranslateAnchor` is `Map`.
        let default_anchor = TranslateAnchor::default();
        assert_eq!(default_anchor, TranslateAnchor::Map);
    }

    #[test]
    fn fill_translate_and_anchor_drop_expression_form() {
        let dropped = parse(json!({
            "id": "shadow", "type": "fill",
            "source": "x", "source-layer": "landuse",
            "paint": {
                "fill-color": "#000",
                "fill-translate": ["literal", [2.0, -3.0]],
                "fill-translate-anchor": ["step", ["zoom"], "map", 12, "viewport"]
            }
        }))
        .unwrap();
        let LayerKind::Fill(f) = &dropped.kind else {
            panic!("expected fill");
        };
        // Same accept-and-drop policy as `text-offset` / `text-anchor`.
        assert_eq!(f.paint.fill_translate, None);
        assert_eq!(f.paint.fill_translate_anchor, None);
    }

    #[test]
    fn parses_line_translate_and_pattern() {
        let layer = parse(json!({
            "id": "shadow_roads", "type": "line",
            "source": "x", "source-layer": "roads",
            "paint": {
                "line-color": "#000",
                "line-translate": [0.0, 1.5],
                "line-translate-anchor": "map",
                "line-pattern": "road_dashes"
            }
        }))
        .unwrap();
        let LayerKind::Line(l) = &layer.kind else {
            panic!("expected line");
        };
        assert_eq!(l.paint.line_translate, Some([0.0, 1.5]));
        assert_eq!(l.paint.line_translate_anchor, Some(TranslateAnchor::Map));
        // `line-pattern` is parse-only this phase — it lands on the
        // paint struct so styles load cleanly, but the renderer
        // ignores it. See the TODO on `LinePaint::line_pattern`.
        assert_eq!(
            l.paint.line_pattern,
            Some(Property::Constant("road_dashes".to_string()))
        );
    }

    #[test]
    fn parses_fill_antialias_parse_only_bool() {
        let layer = parse(json!({
            "id": "x", "type": "fill",
            "source": "x", "source-layer": "y",
            "paint": { "fill-color": "#abc", "fill-antialias": false }
        }))
        .unwrap();
        let LayerKind::Fill(f) = &layer.kind else {
            panic!("expected fill");
        };
        // The renderer never reads this field today; the test just
        // pins the parser surface so styles that set it load cleanly.
        assert_eq!(f.paint.fill_antialias, Some(false));
    }
}
