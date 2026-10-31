// ==============================================================================
// Legacy function conversion — Phase 11
// ==============================================================================
//
// Pre-spec-v8 MapLibre styles describe data-driven values with "stops-
// based functions" — JSON objects of shape `{ type: …, property?, stops:
// [[in, out], …], base?, default?, colorSpace? }` — rather than the
// modern expression operators. Upstream's `convertFunction.ts` rewrites
// those shapes into equivalent expressions before evaluation; we follow
// the same approach here. The runner detects the legacy shape on the
// fixture's `expression` field and routes it through
// [`convert_legacy_function`] before handing it to [`Expr::from_json`].
//
// We don't construct AST nodes directly: emitting the equivalent JSON
// (and then re-parsing it) reuses every existing parser-level check,
// keeps the converter readable as a small JSON-rewriting pipeline, and
// guarantees the output respects the same type rules as a hand-authored
// modern expression. The downside — re-parsing — is negligible at test
// time and a non-issue at runtime where production styles wouldn't
// re-convert per evaluation.
//
// The four legacy "types" we handle here are:
//
// - `identity` — return the named feature property, coerced/typed
//   according to the declared property type. With a `default` we
//   fall back to it whenever the property is missing or fails the
//   coercion.
// - `categorical` — exact-match lookup keyed off either the feature
//   property (data-driven) or the zoom level. Translated to `match`,
//   except for boolean labels which need a `case`/`==` chain because
//   the modern `match` operator only accepts string / integer labels.
// - `interval` (alias `piecewise-constant`) — discrete piecewise
//   selection by either property or zoom. Translates to `step`.
// - `exponential` — non-linear ramp over property and/or zoom.
//   Translates to `interpolate` with an exponential parameterization.
//   The optional `colorSpace` field picks `interpolate-hcl` /
//   `interpolate-lab`. `base == 1` collapses to a plain linear ramp,
//   matching upstream's special-case.
//
// All four support a "composite" form (data-driven *and* zoom-aware) in
// which each stop input is itself `{ zoom, value }`. The conversion
// nests the two axes: an outer zoom curve whose stops are inner
// property curves, mirroring `convertFunction.ts`'s output shape.

use serde_json::{Map as JsonMap, Value as Json, json};

use crate::style::expression::{Expr, ExprError};

// ------------------------------------------------------------------------------
// Recognition
// ------------------------------------------------------------------------------

/// Returns `true` when `expression` looks like a pre-spec-v8 function
/// shape: a JSON *object* (not an array) that carries at least one of
/// the function-defining keys. The fixture corpus is consistent here —
/// every legacy function fixture's expression is an object with
/// `stops`/`property`/`type` keys, and no modern expression is an
/// object at the top level.
pub fn is_legacy_function(expression: &Json) -> bool {
    let Some(obj) = expression.as_object() else {
        return false;
    };
    obj.contains_key("stops") || obj.contains_key("property") || obj.contains_key("type")
}

// ------------------------------------------------------------------------------
// Property-spec view
// ------------------------------------------------------------------------------

/// Compact view of the bits of `propertySpec` the converter consults.
/// We keep the borrow at the JSON level for `default` (legacy defaults
/// can be structural — e.g. `[-1]` for an `array` property) and resolve
/// the other fields up front.
struct PropertySpec<'a> {
    /// The declared property type — `"number"`, `"color"`, `"array"`,
    /// `"colorArray"`, `"padding"`, `"enum"`, …. We pass this through
    /// verbatim and let the per-type branch decide what assertion or
    /// coercion to wrap the legacy stop output in.
    kind: &'a str,
    /// `propertySpec.value` — the per-element type of an `array`
    /// property. Used to shape the modern `["array", T, x]` assertion
    /// wrapping array outputs so the inferred type matches the
    /// fixture's `compile.type` exactly.
    array_value_type: Option<&'a str>,
    /// `propertySpec.length` — explicit array length when the property
    /// pins a fixed shape. Threaded into the 3-arg `array` assertion
    /// when present.
    array_length: Option<usize>,
    /// `propertySpec.values` — the set of legal labels for an `enum`
    /// property. With a default, identity-enum needs to fall back when
    /// the actual value isn't one of these keys.
    enum_values: Vec<String>,
    /// `propertySpec.default`, used when the legacy expression doesn't
    /// carry its own `default`. Falls back to the modern expression's
    /// `null` default at the outer layer (`coalesce` returns the last
    /// child if everything before errored).
    default: Option<&'a Json>,
    /// `propertySpec.tokens` — when true and the function produces a
    /// string output, single-stop-output `{prop}` references in string
    /// stops are expanded into `concat` calls over `to-string ∘ get`.
    /// Tokens are only expanded on the zoom axis; the property axis
    /// of a property-driven function passes strings through unchanged
    /// (mirroring upstream's `convertTokenString`-only-on-zoom rule).
    tokens: bool,
    /// `parameters` from `propertySpec.expression.parameters`. We only
    /// care whether the function is allowed to be zoom-aware,
    /// feature-aware, or both — the converter's main fork keys off the
    /// stops' shape, not this slot, but we keep it around for the
    /// `is_zoom_only` heuristic.
    parameters: Vec<String>,
}

impl<'a> PropertySpec<'a> {
    fn from_json(spec: &'a Json) -> Self {
        let obj = spec.as_object();
        let kind = obj
            .and_then(|o| o.get("type"))
            .and_then(Json::as_str)
            .unwrap_or("value");
        let array_value_type = obj.and_then(|o| o.get("value")).and_then(Json::as_str);
        let array_length = obj
            .and_then(|o| o.get("length"))
            .and_then(Json::as_u64)
            .map(|n| n as usize);
        let enum_values = obj
            .and_then(|o| o.get("values"))
            .and_then(Json::as_object)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        let default = obj.and_then(|o| o.get("default"));
        let tokens = obj
            .and_then(|o| o.get("tokens"))
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let parameters = obj
            .and_then(|o| o.get("expression"))
            .and_then(|e| e.as_object())
            .and_then(|e| e.get("parameters"))
            .and_then(Json::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        PropertySpec {
            kind,
            array_value_type,
            array_length,
            enum_values,
            default,
            tokens,
            parameters,
        }
    }

    /// The function is zoom-only when `parameters` doesn't mention
    /// `feature` (so `["zoom"]` only). Legacy zoom-only functions never
    /// have a `property`, and the per-axis branch logic uses both this
    /// signal and the presence/absence of `property` in the expression.
    fn allows_feature(&self) -> bool {
        self.parameters.iter().any(|p| p == "feature")
    }
}

// ------------------------------------------------------------------------------
// Top-level converter
// ------------------------------------------------------------------------------

/// Translate a legacy function shape into a modern expression and parse
/// it. The result is suitable for [`Expr::eval`] like any other parsed
/// expression. Errors at this stage are limited to malformed shapes
/// (missing `stops`, non-array `stops`, etc.); the caller should
/// surface them just like a parse error from `from_json`.
pub fn convert_legacy_function(
    expression: &Json,
    property_spec: Option<&Json>,
) -> Result<Expr, ExprError> {
    let spec = property_spec
        .map(PropertySpec::from_json)
        .unwrap_or(PropertySpec {
            kind: "value",
            array_value_type: None,
            array_length: None,
            enum_values: Vec::new(),
            default: None,
            tokens: false,
            parameters: Vec::new(),
        });

    let obj = expression
        .as_object()
        .ok_or(ExprError::LegacyMalformed("expected object"))?;

    // The `type` field is optional. Upstream's converter falls back
    // to `exponential` when the property spec marks the function as
    // interpolatable (`propertySpec.expression.interpolated: true`) and
    // `interval` otherwise. The two `legacy/{exponential,interval}/
    // implicit` fixtures pin both branches.
    let kind = obj.get("type").and_then(Json::as_str).map(str::to_owned);
    let kind = match kind {
        Some(k) => k,
        None => {
            let interpolated = property_spec
                .and_then(|s| s.as_object())
                .and_then(|s| s.get("expression"))
                .and_then(|e| e.as_object())
                .and_then(|e| e.get("interpolated"))
                .and_then(Json::as_bool)
                .unwrap_or(false);
            if interpolated {
                "exponential".to_owned()
            } else {
                "interval".to_owned()
            }
        }
    };

    let modern = match kind.as_str() {
        "identity" => convert_identity(obj, &spec)?,
        "categorical" => convert_categorical(obj, &spec)?,
        // `piecewise-constant` is the pre-v8 alias for `interval` — the
        // two have identical semantics (lower-clamp piecewise output).
        "interval" | "piecewise-constant" => convert_interval(obj, &spec)?,
        "exponential" => convert_exponential(obj, &spec)?,
        _ => return Err(ExprError::LegacyMalformed("unknown legacy function type")),
    };

    Expr::from_json(&modern)
}

// ------------------------------------------------------------------------------
// identity
// ------------------------------------------------------------------------------

/// `{ type: "identity", property: P }` → return the named property,
/// asserted/coerced to the declared property type. With a default
/// (either on the expression itself or on the propertySpec) we wrap in
/// a typed-assertion-with-fallback so the default takes over both when
/// the property is missing and when it fails the type assertion.
///
/// `enum`-typed properties are treated as strings because the corpus's
/// `legacy/identity/enum*` fixtures pin `compile.type` as `string`.
/// `colorArray` / `padding` aren't modeled by the modern expression
/// runtime and stay deferred to the manifest.
fn convert_identity(
    obj: &JsonMap<String, Json>,
    spec: &PropertySpec<'_>,
) -> Result<Json, ExprError> {
    let property = obj
        .get("property")
        .and_then(Json::as_str)
        .ok_or(ExprError::LegacyMalformed("identity needs `property`"))?;
    let default = obj.get("default").or(spec.default);

    // Enum properties with a default need an explicit membership check
    // — `["string", get, default]` would happily pass any string value
    // through, including one outside the enum's `values` map. We emit
    // a `match` that whitelists the allowed labels and falls back to
    // the default for anything else. Without a default we stay on the
    // plain string-assertion path so the corpus's no-default fixture
    // round-trips unknown values verbatim.
    if let (Some(default_val), "enum") = (default, spec.kind) {
        if spec.enum_values.is_empty() {
            // Fall through to the plain string-assertion path below;
            // without an enum value list there's nothing to match
            // against.
        } else {
            return Ok(build_enum_match(
                property,
                default_val.clone(),
                &spec.enum_values,
            ));
        }
    }
    let get_expr = json!(["get", property]);
    let typed = type_assertion_or_coercion(spec, get_expr, default)?;
    Ok(typed)
}

/// `["match", ["string", ["get", prop], default], "a", "a", "b", "b",
/// default]` — whitelist the legal enum labels and fall back to the
/// default for anything else. The inner `string` assertion handles the
/// "property is null / wrong type" branch up front.
fn build_enum_match(property: &str, default_val: Json, enum_values: &[String]) -> Json {
    let mut call: Vec<Json> = vec![
        Json::String("match".into()),
        json!(["string", ["get", property], default_val.clone()]),
    ];
    for v in enum_values {
        call.push(Json::String(v.clone()));
        call.push(Json::String(v.clone()));
    }
    call.push(default_val);
    Json::Array(call)
}

/// Wrap a value-producing expression in the assertion/coercion
/// appropriate for the declared property type. `default` (if any) is
/// appended as a fallback so the wrapper transparently absorbs both
/// "property missing" and "property has the wrong type" cases.
fn type_assertion_or_coercion(
    spec: &PropertySpec<'_>,
    value: Json,
    default: Option<&Json>,
) -> Result<Json, ExprError> {
    let kind = spec.kind;
    // Defaults can be structural (`[-1]`, an object literal, …) — we
    // wrap them in `["literal", …]` so the modern parser doesn't try
    // to re-interpret them as operator calls.
    let default_expr = default.map(as_value_expr);
    match kind {
        // Scalar assertions chain through their fallback list. With a
        // default we surface it after the value; the assertion will
        // return the default if the first arg fails the type check.
        "number" | "string" | "boolean" | "object" | "enum" => {
            // `enum` is rendered as `string` in expression-land —
            // upstream's `enum` propertySpec type compiles down to a
            // plain `string` assertion. No corpus fixture distinguishes
            // the two at the expression-result-type level.
            let op = if kind == "enum" { "string" } else { kind };
            let mut call = vec![Json::String(op.into()), value];
            if let Some(d) = default_expr {
                call.push(d);
            }
            Ok(Json::Array(call))
        }
        // Color falls under the same "first arg wins, else fallback"
        // pattern but uses `to-color` instead of an assertion because
        // we accept any color-shaped value (CSS string, `[r,g,b,a]`
        // array, etc.) — not just one that's literally typed `color`.
        "color" => {
            let mut call = vec![Json::String("to-color".into()), value];
            if let Some(d) = default_expr {
                call.push(d);
            }
            Ok(Json::Array(call))
        }
        // Typed arrays — `["array", T, N, value, fallback?]` matches
        // the property spec's pinned shape. We thread `value`/`length`
        // through whenever the spec declares them. Without a length we
        // still need the length slot when a fallback is present
        // (otherwise the parser interprets `value` as the length); we
        // surface `null` for that case, matching the modern parser's
        // "length may be null" branch.
        "array" => {
            let mut call: Vec<Json> = vec![Json::String("array".into())];
            if let Some(t) = spec.array_value_type {
                call.push(Json::String(t.into()));
                match (spec.array_length, &default_expr) {
                    (Some(n), _) => call.push(Json::Number(n.into())),
                    (None, Some(_)) => call.push(Json::Null),
                    (None, None) => {}
                }
            }
            call.push(value);
            if let Some(d) = default_expr {
                call.push(d);
            }
            Ok(Json::Array(call))
        }
        // numberArray / colorArray / padding — modern expression-land
        // has no first-class operator for these aggregate types. Fall
        // through to a bare `get`; the runner's propertySpec check will
        // reject the shape and the fixture stays in the manifest. We
        // make sure not to lose the default though — wrap it in
        // `coalesce`.
        _ => {
            let call = if let Some(d) = default_expr {
                json!(["coalesce", value, d])
            } else {
                value
            };
            Ok(call)
        }
    }
}

/// Convert a raw JSON value into an expression that *evaluates* to that
/// value. Structural shapes (arrays, objects) need `["literal", …]` to
/// keep the parser from re-interpreting them as operator calls;
/// scalars round-trip as themselves.
fn as_value_expr(v: &Json) -> Json {
    match v {
        Json::Array(_) | Json::Object(_) => json!(["literal", v]),
        other => other.clone(),
    }
}

// ------------------------------------------------------------------------------
// categorical
// ------------------------------------------------------------------------------

/// `{ type: "categorical", property?: P, stops: [[label, out], …],
/// default? }` → a `match` (string/number labels) or `case` (boolean
/// labels) over the property or zoom input. The default falls back to
/// the propertySpec's default when the expression itself doesn't carry
/// one; if neither is present we emit `null`, matching upstream.
fn convert_categorical(
    obj: &JsonMap<String, Json>,
    spec: &PropertySpec<'_>,
) -> Result<Json, ExprError> {
    let stops = require_stops(obj)?;
    let default = obj.get("default").or(spec.default);
    let property = obj.get("property").and_then(Json::as_str);

    // The categorical function's input is the property value if one is
    // declared, else the zoom level. The corpus only exercises the
    // property-driven form, but the zoom form falls out cleanly.
    let input = match property {
        Some(p) => json!(["get", p]),
        None => json!(["zoom"]),
    };

    // Decide between `match` and a `case` chain by looking at the
    // first stop's label kind. Booleans need the `case` chain because
    // the modern `match` operator only takes string / safe-integer
    // labels (upstream's `Match.parse` rejects boolean labels with
    // "Branch labels must be integers or strings.").
    let first_label = stops
        .first()
        .and_then(|s| s.as_array())
        .and_then(|p| p.first())
        .ok_or(ExprError::LegacyMalformed("categorical needs ≥ 1 stop"))?;

    let fallback = wrap_output(spec, default.cloned().unwrap_or(Json::Null), false);

    if matches!(first_label, Json::Bool(_)) {
        // `case` shape: `["case", ["==", input, lab1], out1, ["==",
        // input, lab2], out2, …, fallback]`.
        let mut call: Vec<Json> = vec![Json::String("case".into())];
        for stop in stops {
            let (label, out) = stop_pair(stop)?;
            call.push(json!(["==", input.clone(), label.clone()]));
            call.push(wrap_output(spec, out.clone(), false));
        }
        call.push(fallback);
        return Ok(Json::Array(call));
    }

    // `match` shape: `["match", input, lab1, out1, lab2, out2, …,
    // fallback]`. Numeric labels round-trip as numbers, string labels
    // as strings; we leave them untouched because the modern parser
    // does the validation.
    let mut call: Vec<Json> = vec![Json::String("match".into()), input];
    for stop in stops {
        let (label, out) = stop_pair(stop)?;
        call.push(label.clone());
        call.push(wrap_output(spec, out.clone(), false));
    }
    call.push(fallback);
    Ok(Json::Array(call))
}

// ------------------------------------------------------------------------------
// interval
// ------------------------------------------------------------------------------

/// `{ type: "interval", property?, stops: [[in, out], …], default? }`
/// → a `step` expression over the property or zoom input, or a nested
/// outer-zoom-step / inner-property-step pair for the composite form.
///
/// The first stop's output becomes `step`'s default (the value
/// returned when `input < firstStop.in`); the remaining stops become
/// `(in, out)` pairs. Duplicate-in stops are merged keeping the
/// earliest output — modern `step` rejects equal-or-decreasing stops
/// at parse time, but upstream's legacy converter tolerates them by
/// silently dropping the duplicate.
fn convert_interval(
    obj: &JsonMap<String, Json>,
    spec: &PropertySpec<'_>,
) -> Result<Json, ExprError> {
    let stops = require_stops(obj)?;
    let default = obj.get("default").or(spec.default);
    let property = obj.get("property").and_then(Json::as_str);

    // Composite form: stop inputs are `{zoom, value}` objects. We
    // detect by looking at the first stop's input — the legacy shape
    // requires every stop to use the same form.
    if is_composite_stops(stops) {
        return build_composite(
            stops,
            spec,
            default,
            property.ok_or(ExprError::LegacyMalformed(
                "composite interval needs `property`",
            ))?,
            CurveKind::Step,
        );
    }

    let input = pick_axis_input(property, spec);
    let expand_tokens = property.is_none() && spec.tokens && is_string_kind(spec.kind);
    let step = build_step(&input, stops, spec, expand_tokens)?;
    Ok(maybe_wrap_default(step, default, spec))
}

// ------------------------------------------------------------------------------
// exponential
// ------------------------------------------------------------------------------

/// `{ type: "exponential", property?, stops, base?, colorSpace? }` → an
/// `interpolate` (or `interpolate-hcl` / `interpolate-lab`) expression.
/// Composite forms nest exactly like `convert_interval`'s — the outer
/// axis is zoom, the inner axis is property — but with `interpolate` on
/// both axes instead of `step`.
fn convert_exponential(
    obj: &JsonMap<String, Json>,
    spec: &PropertySpec<'_>,
) -> Result<Json, ExprError> {
    let stops = require_stops(obj)?;
    let default = obj.get("default").or(spec.default);
    let property = obj.get("property").and_then(Json::as_str);
    let base = obj.get("base").and_then(Json::as_f64).unwrap_or(1.0);
    let color_space = obj
        .get("colorSpace")
        .and_then(Json::as_str)
        .unwrap_or("rgb");

    let curve = CurveKind::Interpolate { base, color_space };

    if is_composite_stops(stops) {
        return build_composite(
            stops,
            spec,
            default,
            property.ok_or(ExprError::LegacyMalformed(
                "composite exponential needs `property`",
            ))?,
            curve,
        );
    }

    let input = pick_axis_input(property, spec);
    let expand_tokens = property.is_none() && spec.tokens && is_string_kind(spec.kind);
    let interp = build_interpolate(&input, stops, spec, expand_tokens, curve)?;
    Ok(maybe_wrap_default(interp, default, spec))
}

// ------------------------------------------------------------------------------
// Curve construction
// ------------------------------------------------------------------------------

/// Which curve operator a sub-axis uses. The composite builder picks
/// the same curve for both axes (per upstream); we keep the variant
/// here so the per-axis builder is one function.
#[derive(Debug, Clone, Copy)]
enum CurveKind<'a> {
    Step,
    Interpolate { base: f64, color_space: &'a str },
}

/// Build a `step` over the given stops. `input` is the already-typed
/// axis input (`["get", p]`, `["number", ["get", p]]`, `["zoom"]`,
/// …). `output_kind` selects the per-output coercion (e.g. `to-color`
/// for color-typed properties). `expand_tokens` turns string outputs
/// into `concat`-of-tokens calls.
fn build_step(
    input: &Json,
    stops: &[Json],
    spec: &PropertySpec<'_>,
    expand_tokens: bool,
) -> Result<Json, ExprError> {
    let (first_out, tail) = split_stops(stops)?;
    // A single stop has no breakpoints, so the output is constant —
    // emit it directly. Modern `step` rejects the 2-arg `(input,
    // default)` shape, and upstream's converter applies the same
    // collapse.
    if tail.is_empty() {
        return Ok(wrap_output(spec, first_out, expand_tokens));
    }
    let mut call: Vec<Json> = vec![Json::String("step".into()), input.clone()];
    call.push(wrap_output(spec, first_out, expand_tokens));
    for (stop_in, stop_out) in tail {
        call.push(stop_in);
        call.push(wrap_output(spec, stop_out, expand_tokens));
    }
    Ok(Json::Array(call))
}

/// Build an `interpolate` (linear / exponential / hcl / lab) over the
/// given stops. Boolean / null / non-color string outputs are rejected
/// by the parser at the call site, so we don't pre-screen them here.
fn build_interpolate(
    input: &Json,
    stops: &[Json],
    spec: &PropertySpec<'_>,
    expand_tokens: bool,
    curve: CurveKind<'_>,
) -> Result<Json, ExprError> {
    let CurveKind::Interpolate { base, color_space } = curve else {
        unreachable!("build_interpolate only called for Interpolate");
    };

    // The interpolation kind — `["exponential", base]` for `base != 1`,
    // `["linear"]` otherwise. Upstream's converter applies the same
    // collapse; `base == 1` is mathematically linear and using
    // `exponential` would force the runtime through a redundant
    // branch.
    let interp_kind = if (base - 1.0).abs() < f64::EPSILON {
        json!(["linear"])
    } else {
        json!(["exponential", base])
    };

    // The HCL / LAB color spaces use dedicated operators, not a
    // `colorSpace` field. The modern expression form rolls the color
    // space into the operator name itself.
    let op = match color_space {
        "hcl" => "interpolate-hcl",
        "lab" => "interpolate-lab",
        _ => "interpolate",
    };

    let pairs = deduplicate_stops(stops)?;
    // A single stop has no second point to interpolate toward, so the
    // output is constant — emit it directly. Modern `interpolate`
    // rejects the 3-arg `(kind, input)` shape, and upstream's
    // converter applies the same collapse.
    if pairs.len() == 1 {
        let (_, out) = pairs.into_iter().next().expect("len == 1");
        return Ok(wrap_output(spec, out, expand_tokens));
    }
    let mut call: Vec<Json> = vec![Json::String(op.into()), interp_kind, input.clone()];
    for (stop_in, stop_out) in pairs {
        call.push(stop_in);
        call.push(wrap_output(spec, stop_out, expand_tokens));
    }
    Ok(Json::Array(call))
}

// ------------------------------------------------------------------------------
// Composite (zoom + property) curves
// ------------------------------------------------------------------------------

/// Stop input shape for composite functions: `{ zoom: Z, value: V }`.
/// Returned by `parse_composite_stops` when we group by zoom.
struct CompositeStop {
    zoom: f64,
    value: Json,
    output: Json,
}

/// Returns `true` if the first stop's input is an object — the
/// canonical composite shape.
fn is_composite_stops(stops: &[Json]) -> bool {
    stops
        .first()
        .and_then(|s| s.as_array())
        .and_then(|p| p.first())
        .map(|v| v.is_object())
        .unwrap_or(false)
}

/// Pull the `{zoom, value}` blocks out of a composite stops list. We
/// preserve source order; downstream grouping relies on consecutive
/// same-zoom entries already being adjacent (the legacy convention).
fn parse_composite_stops(stops: &[Json]) -> Result<Vec<CompositeStop>, ExprError> {
    let mut out = Vec::with_capacity(stops.len());
    for stop in stops {
        let pair = stop.as_array().ok_or(ExprError::LegacyMalformed(
            "composite stop must be a 2-tuple",
        ))?;
        if pair.len() != 2 {
            return Err(ExprError::LegacyMalformed(
                "composite stop must have input and output",
            ));
        }
        let input_obj = pair[0].as_object().ok_or(ExprError::LegacyMalformed(
            "composite stop input must be {zoom, value}",
        ))?;
        let zoom = input_obj
            .get("zoom")
            .and_then(Json::as_f64)
            .ok_or(ExprError::LegacyMalformed("composite stop needs `zoom`"))?;
        let value = input_obj
            .get("value")
            .cloned()
            .ok_or(ExprError::LegacyMalformed("composite stop needs `value`"))?;
        out.push(CompositeStop {
            zoom,
            value,
            output: pair[1].clone(),
        });
    }
    Ok(out)
}

/// Build the nested curve for a composite legacy function: outer axis
/// is zoom, inner axis is the property value. Both axes use the same
/// curve operator (so an `interval` composite is `step`-of-`step`, an
/// `exponential` composite is `interpolate`-of-`interpolate`).
fn build_composite(
    stops: &[Json],
    spec: &PropertySpec<'_>,
    default: Option<&Json>,
    property: &str,
    curve: CurveKind<'_>,
) -> Result<Json, ExprError> {
    let composite = parse_composite_stops(stops)?;

    // Group adjacent-by-zoom: each group becomes one inner curve. The
    // legacy stops list is required to be sorted by `(zoom, value)`,
    // so grouping by consecutive equal zooms gives the right layout.
    let mut groups: Vec<(f64, Vec<(Json, Json)>)> = Vec::new();
    for stop in composite {
        if let Some((last_zoom, entries)) = groups.last_mut()
            && *last_zoom == stop.zoom {
                entries.push((stop.value, stop.output));
                continue;
            }
        groups.push((stop.zoom, vec![(stop.value, stop.output)]));
    }

    // Property-axis input: the inner step/interpolate keys off a
    // numeric value, so we always wrap with `["number", ["get", p]]`.
    // The output-type wrapping happens per stop, not on the input.
    let inner_input = axis_input_for_property(property);

    // The inner axis builder needs each stop as a 2-element JSON
    // array — recreate that shape from the grouped entries.
    let inner_stops_json = |entries: &[(Json, Json)]| -> Vec<Json> {
        entries
            .iter()
            .map(|(v, o)| Json::Array(vec![v.clone(), o.clone()]))
            .collect()
    };

    // Outer-axis stops: each (zoom, inner_curve) becomes a stop on the
    // zoom axis. The inner curve uses the same kind as the outer.
    let outer_stops: Vec<Json> = groups
        .iter()
        .map(|(zoom, entries)| {
            let inner_stops = inner_stops_json(entries);
            let inner = match curve {
                CurveKind::Step => build_step(&inner_input, &inner_stops, spec, false)?,
                CurveKind::Interpolate { .. } => {
                    build_interpolate(&inner_input, &inner_stops, spec, false, curve)?
                }
            };
            Ok(Json::Array(vec![
                Json::Number(serde_json::Number::from_f64(*zoom).ok_or(
                    ExprError::LegacyMalformed("non-finite zoom in composite stop"),
                )?),
                inner,
            ]))
        })
        .collect::<Result<_, ExprError>>()?;

    // The outer (zoom-axis) builder shouldn't double-wrap the per-stop
    // outputs — each stop's value is already a fully-built inner
    // curve. A "passthrough" spec leaves the value untouched (no
    // assertion, no token expansion).
    let passthrough = PropertySpec {
        kind: "passthrough",
        array_value_type: None,
        array_length: None,
        enum_values: Vec::new(),
        default: None,
        tokens: false,
        parameters: Vec::new(),
    };
    let zoom_input = json!(["zoom"]);
    let outer = match curve {
        CurveKind::Step => build_step(&zoom_input, &outer_stops, &passthrough, false)?,
        CurveKind::Interpolate { .. } => {
            build_interpolate(&zoom_input, &outer_stops, &passthrough, false, curve)?
        }
    };
    Ok(maybe_wrap_default(outer, default, spec))
}

// ------------------------------------------------------------------------------
// Shared helpers
// ------------------------------------------------------------------------------

/// The axis input for a property-driven non-composite function. Step
/// and interpolate both key off a *numeric* stop input, so the property
/// is always asserted as `number` regardless of the function's output
/// type — that surfaces the spec-pinned "Expected value to be of type
/// number, but found null instead." message on missing or wrong-type
/// properties.
fn axis_input_for_property(property: &str) -> Json {
    json!(["number", ["get", property]])
}

/// Pick the axis input expression for a non-composite legacy function.
/// Property-driven uses a numeric assertion over `["get", p]`; zoom-
/// only uses `["zoom"]`.
fn pick_axis_input(property: Option<&str>, spec: &PropertySpec<'_>) -> Json {
    match property {
        Some(p) => axis_input_for_property(p),
        None => {
            // Zoom-only function. We still consult the spec to decide
            // whether the function is *allowed* to be feature-aware
            // (and so whether `propertySpec.parameters` agrees), but
            // the axis input itself is always `zoom`.
            let _ = spec.allows_feature();
            json!(["zoom"])
        }
    }
}

/// Wrap a stop output in the coercion that matches the declared
/// property type — `to-color` for colors, `concat` for token-bearing
/// strings, an explicit `array` assertion for typed arrays, the value
/// unchanged otherwise.
///
/// Numeric / boolean outputs round-trip through `from_json` as
/// literals, so no wrapping is needed there. Array outputs need both
/// the `literal` wrapper (so the parser doesn't reinterpret them as
/// operator calls) and the `["array", T, …]` assertion when the
/// property spec pins an item type — without the assertion the
/// inferred type carries the literal's exact length, which diverges
/// from the corpus's `array<T>` expectation on length-free shapes.
fn wrap_output(spec: &PropertySpec<'_>, value: Json, expand_tokens: bool) -> Json {
    // Token expansion: only on the zoom axis, only for string outputs.
    // Non-string values pass through unchanged.
    if expand_tokens
        && let Some(s) = value.as_str() {
            return expand_token_string(s);
        }
    // The composite builder's outer axis passes already-built inner
    // curves (themselves JSON arrays) through this helper. Wrapping
    // them in `["literal", …]` would turn the curve into an opaque
    // value; the passthrough kind escapes the wrapping rules.
    if spec.kind == "passthrough" {
        return value;
    }
    match spec.kind {
        // Color stops can be CSS strings, `[r, g, b, a]` arrays, or any
        // already-color-typed expression. `to-color` handles all three
        // and is cheap when given a value that already parses as a
        // color literal.
        "color" => match &value {
            Json::String(_) => json!(["to-color", value]),
            Json::Array(_) => json!(["to-color", literal_wrap(value)]),
            _ => json!(["to-color", value]),
        },
        // Typed array stops — wrap the JSON array as a `literal`, then
        // assert its item type (and length, when the spec pins one)
        // via the `array` operator. This lines the inferred type up
        // with `propertySpec.value` regardless of the per-stop
        // length variance.
        "array" => {
            let lit = match &value {
                Json::Array(_) => literal_wrap(value),
                _ => value,
            };
            assert_array_with(spec.array_value_type, spec.array_length, lit)
        }
        // Default branch: pass the value through, wrapping bare arrays
        // in `literal` for the parser's benefit. Aggregate types like
        // `colorArray`, `numberArray`, `padding` flow here — without
        // dedicated assertions they stay typed `value`, which the
        // outer propertySpec check will reject. Those fixtures stay
        // in the manifest until a downstream phase models the types.
        _ => match value {
            Json::Array(_) => literal_wrap(value),
            other => other,
        },
    }
}

/// `["array", T, N, value]` / `["array", T, value]` / `["array",
/// value]` depending on which slots are present. The modern parser
/// rejects unrecognized item-type strings, so an item-type slot we
/// don't recognize falls back to a bare assertion.
fn assert_array_with(item_type: Option<&str>, length: Option<usize>, value: Json) -> Json {
    match (item_type, length) {
        (Some(t), Some(n)) => json!(["array", t, n, value]),
        (Some(t), None) => json!(["array", t, value]),
        _ => json!(["array", value]),
    }
}

/// `["literal", value]`. Required to pass an array/object as a value
/// to the modern expression parser without it being interpreted as an
/// operator call.
fn literal_wrap(value: Json) -> Json {
    json!(["literal", value])
}

/// Token expansion for string outputs on the zoom axis. A token
/// `{prop}` becomes `["to-string", ["get", "prop"]]`; surrounding
/// literal text is preserved as a string segment of `concat`. The
/// pattern mirrors upstream's `convertTokenString`.
fn expand_token_string(s: &str) -> Json {
    let mut sections: Vec<Json> = Vec::new();
    let mut literal_start = 0;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // Find the closing brace. Tokens with no closing brace
            // pass through verbatim (matching the upstream walker).
            if let Some(rel) = s[i + 1..].find('}') {
                let end = i + 1 + rel;
                if literal_start < i {
                    sections.push(Json::String(s[literal_start..i].to_owned()));
                }
                let name = &s[i + 1..end];
                sections.push(json!(["to-string", ["get", name]]));
                i = end + 1;
                literal_start = i;
                continue;
            }
        }
        i += 1;
    }
    if literal_start < s.len() {
        sections.push(Json::String(s[literal_start..].to_owned()));
    }

    match sections.len() {
        0 => Json::String(String::new()),
        // A single literal section is the original string verbatim; no
        // `concat` needed.
        1 if matches!(&sections[0], Json::String(_)) => sections.pop().expect("len 1"),
        _ => {
            let mut call = vec![Json::String("concat".into())];
            call.extend(sections);
            Json::Array(call)
        }
    }
}

/// Wrap an expression in `coalesce(value, default)` when a default is
/// present. `coalesce` swallows errors from its leading children, so
/// any runtime failure inside the curve (a missing property, a wrong-
/// type input) falls through to the default. With no default we return
/// `value` unchanged.
///
/// The default value goes through the property-spec's output coercion
/// (so e.g. `"red"` for a color property becomes `["to-color", "red"]`)
/// — without that, color defaults would surface as plain strings and
/// the runner's type comparison would diverge from upstream.
fn maybe_wrap_default(value: Json, default: Option<&Json>, spec: &PropertySpec<'_>) -> Json {
    match default {
        Some(d) => json!(["coalesce", value, wrap_output(spec, d.clone(), false)]),
        None => value,
    }
}

/// Stop list accessor with the "must be an array" check inline.
fn require_stops(obj: &JsonMap<String, Json>) -> Result<&[Json], ExprError> {
    let stops = obj
        .get("stops")
        .and_then(Json::as_array)
        .ok_or(ExprError::LegacyMalformed("missing `stops` array"))?;
    if stops.is_empty() {
        return Err(ExprError::LegacyMalformed("`stops` must be non-empty"));
    }
    Ok(stops)
}

/// Pull `(label, output)` out of a `[label, output]` stop pair.
fn stop_pair(stop: &Json) -> Result<(&Json, &Json), ExprError> {
    let arr = stop
        .as_array()
        .ok_or(ExprError::LegacyMalformed("stop must be an array"))?;
    if arr.len() != 2 {
        return Err(ExprError::LegacyMalformed(
            "stop must be a [label, output] pair",
        ));
    }
    Ok((&arr[0], &arr[1]))
}

/// Split scalar-input stops into `(firstOutput, [(in, out), …])`.
/// `step`'s shape is "default, then ascending (in, out) pairs"; the
/// first stop's output becomes the default and the rest become pairs.
/// We deduplicate as we go: legacy converters accept duplicate-in
/// stops, modern `step` doesn't.
fn split_stops(stops: &[Json]) -> Result<(Json, Vec<(Json, Json)>), ExprError> {
    let mut deduped = deduplicate_stops(stops)?.into_iter();
    let (_first_in, first_out) = deduped
        .next()
        .ok_or(ExprError::LegacyMalformed("step needs ≥ 1 stop"))?;
    let tail = deduped.collect();
    Ok((first_out, tail))
}

/// Drop duplicate-in stops, keeping the earliest output. Upstream's
/// converter silently elides duplicates so the modern parser doesn't
/// trip its strict ascending-input check on a legacy function that's
/// otherwise valid (`legacy/{interval,exponential}/duplicate-stops`).
fn deduplicate_stops(stops: &[Json]) -> Result<Vec<(Json, Json)>, ExprError> {
    let mut out: Vec<(Json, Json)> = Vec::with_capacity(stops.len());
    for stop in stops {
        let (label, output) = stop_pair(stop)?;
        // The stop's input must be a numeric scalar — we can compare
        // by `f64` equality because the legacy spec requires literal
        // numeric inputs (the composite form goes through a different
        // code path).
        let label_num = label.as_f64();
        if let Some(num) = label_num
            && let Some((prev, _)) = out.last()
                && prev.as_f64() == Some(num) {
                    continue;
                }
        out.push((label.clone(), output.clone()));
    }
    Ok(out)
}

fn is_string_kind(kind: &str) -> bool {
    matches!(kind, "string" | "enum" | "formatted" | "resolvedImage")
}

// ==============================================================================
// Legacy filter conversion
// ==============================================================================
//
// Pre-spec-v8 styles describe layer filters with a different mini-
// language than modern expressions:
//
// - Comparison ops take a *bare property name* in the second slot:
//   `["==", "kind", "highway"]` is the legacy form of
//   `["==", ["get", "kind"], "highway"]`.
// - Two pseudo-properties `$type` and `$id` map to `["geometry-type"]`
//   and `["id"]`.
// - The `in` / `!in` ops are variadic: `["in", k, v1, v2, …]` reads
//   "property k equals one of v1, v2, …". Modern `in` is strictly
//   `[op, needle, haystack]`.
// - The `none` combinator (legacy-only) is `not(any(…))`.
// - `has` / `!has` exist in both dialects and pass through.
//
// Detection is signature-based: any array whose head matches one of
// the legacy operator names is treated as legacy. The conversion
// recurses through `all` / `any` / `none` so nested filters are
// rewritten consistently. Modern expression filters (where the second
// slot is itself an expression array) are left alone.

/// Returns `true` if the top-level shape of `filter` looks like a
/// pre-spec-v8 filter. Used by [`Layer::deserialize`] to route through
/// the converter before handing the value to the modern expression
/// parser. Conservative on purpose — when in doubt we leave the
/// filter alone and let the modern parser do its thing.
pub fn is_legacy_filter(filter: &Json) -> bool {
    let Some(arr) = filter.as_array() else {
        return false;
    };
    let Some(head) = arr.first().and_then(Json::as_str) else {
        return false;
    };
    match head {
        // Operators that only exist in the legacy dialect — any array
        // headed by these is unambiguously legacy.
        "in" | "!in" | "none" | "!has" => true,
        // Combinators are dialect-agnostic; recurse into children.
        "all" | "any" => arr.iter().skip(1).any(is_legacy_filter),
        // Comparison ops are legacy iff the key slot is a bare string
        // (modern would wrap it in `["get", …]` or similar).
        "==" | "!=" | ">" | ">=" | "<" | "<=" => arr.get(1).is_some_and(Json::is_string),
        // `has` is identical in both dialects, but a legacy doc may
        // still embed it inside `all` etc., so we don't claim it on
        // its own.
        _ => false,
    }
}

/// Rewrite a legacy filter as an equivalent modern expression. Modern
/// shapes pass through unchanged. The recursion handles `all` / `any`
/// / `none` regardless of whether each child individually looked
/// legacy — once we've decided the outer is legacy, children get the
/// same treatment for consistency.
pub fn convert_legacy_filter(filter: &Json) -> Json {
    let Some(arr) = filter.as_array() else {
        return filter.clone();
    };
    let Some(head) = arr.first().and_then(Json::as_str) else {
        return filter.clone();
    };
    let args = &arr[1..];

    match head {
        "all" | "any" => {
            let mut out = vec![json!(head)];
            for child in args {
                out.push(convert_legacy_filter(child));
            }
            Json::Array(out)
        }
        // Legacy `none` = `all(not(child1), not(child2), …)`.
        "none" => {
            let mut out = vec![json!("all")];
            for child in args {
                out.push(json!(["!", convert_legacy_filter(child)]));
            }
            Json::Array(out)
        }
        "has" => {
            // `has` is identical in both dialects but accept its legacy
            // bare-string form anywhere we're already in legacy mode.
            let key = args.first().cloned().unwrap_or(Json::Null);
            json!(["has", key])
        }
        "!has" => {
            let key = args.first().cloned().unwrap_or(Json::Null);
            json!(["!", ["has", key]])
        }
        "in" => convert_legacy_in(args, false),
        "!in" => convert_legacy_in(args, true),
        "==" | "!=" | ">" | ">=" | "<" | "<=" => {
            // Legacy comparison: `[op, key, value]`. The key may be a
            // bare property name or one of the `$type` / `$id` pseudo-
            // properties; in either case we lift it into the
            // corresponding expression node.
            let Some(key) = args.first().and_then(Json::as_str) else {
                return filter.clone();
            };
            let value = args.get(1).cloned().unwrap_or(Json::Null);
            let lhs = legacy_key_to_expr(key);
            json!([head, lhs, value])
        }
        _ => filter.clone(),
    }
}

/// Rewrite legacy `["in", k, v1, v2, …]` (or its negation) as a modern
/// `["in", needle, ["literal", [v1, …]]]`. The variadic legacy form
/// becomes a single haystack literal; `!in` wraps the result in `!`.
fn convert_legacy_in(args: &[Json], negate: bool) -> Json {
    let Some(key) = args.first().and_then(Json::as_str) else {
        // Malformed; let the modern parser surface the error.
        return Json::Array({
            let mut v = vec![json!(if negate { "!in" } else { "in" })];
            v.extend_from_slice(args);
            v
        });
    };
    let needle = legacy_key_to_expr(key);
    let haystack_values: Vec<Json> = args.iter().skip(1).cloned().collect();
    let modern = json!(["in", needle, ["literal", haystack_values]]);
    if negate { json!(["!", modern]) } else { modern }
}

/// Map a legacy filter key — a bare property name or one of the two
/// `$type` / `$id` pseudo-properties — to the equivalent modern
/// expression sub-tree.
fn legacy_key_to_expr(key: &str) -> Json {
    match key {
        "$type" => json!(["geometry-type"]),
        "$id" => json!(["id"]),
        other => json!(["get", other]),
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;

    #[test]
    fn recognises_legacy_filters() {
        assert!(is_legacy_filter(&json!(["==", "kind", "highway"])));
        assert!(is_legacy_filter(&json!(["==", "$type", "Polygon"])));
        assert!(is_legacy_filter(&json!(["in", "kind", "a", "b"])));
        assert!(is_legacy_filter(&json!([
            "all",
            ["==", "kind", "highway"],
            ["has", "name"]
        ])));
        assert!(is_legacy_filter(&json!(["none", ["==", "kind", "x"]])));
        // Modern: `==` with an expression in the key slot.
        assert!(!is_legacy_filter(&json!([
            "==",
            ["get", "kind"],
            "highway"
        ])));
        assert!(!is_legacy_filter(&json!(["has", "name"])));
    }

    #[test]
    fn rewrites_comparison() {
        let out = convert_legacy_filter(&json!(["==", "kind", "highway"]));
        assert_eq!(out, json!(["==", ["get", "kind"], "highway"]));
    }

    #[test]
    fn rewrites_dollar_type() {
        let out = convert_legacy_filter(&json!(["==", "$type", "Polygon"]));
        assert_eq!(out, json!(["==", ["geometry-type"], "Polygon"]));
    }

    #[test]
    fn rewrites_in() {
        let out = convert_legacy_filter(&json!(["in", "kind", "river", "stream"]));
        assert_eq!(
            out,
            json!(["in", ["get", "kind"], ["literal", ["river", "stream"]]])
        );
    }

    #[test]
    fn rewrites_not_in() {
        let out = convert_legacy_filter(&json!(["!in", "kind", "a"]));
        assert_eq!(
            out,
            json!(["!", ["in", ["get", "kind"], ["literal", ["a"]]]])
        );
    }

    #[test]
    fn rewrites_none_to_not_any() {
        let out = convert_legacy_filter(&json!(["none", ["==", "kind", "a"], ["==", "kind", "b"]]));
        assert_eq!(
            out,
            json!([
                "all",
                ["!", ["==", ["get", "kind"], "a"]],
                ["!", ["==", ["get", "kind"], "b"]]
            ])
        );
    }

    #[test]
    fn rewrites_nested_filters() {
        let out = convert_legacy_filter(&json!([
            "all",
            ["has", "is_tunnel"],
            ["in", "kind", "other", "path"]
        ]));
        assert_eq!(
            out,
            json!([
                "all",
                ["has", "is_tunnel"],
                ["in", ["get", "kind"], ["literal", ["other", "path"]]]
            ])
        );
    }
}
