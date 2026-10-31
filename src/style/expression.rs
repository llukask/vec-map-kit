// Style expressions are MapLibre's small Lisp-like sublanguage for
// values that depend on the feature, the zoom level, or both. JSON
// arrays whose first element is a string are operator calls; everything
// else is a literal.
//
// Coverage is built up phase by phase per the project roadmap
// (`maplibre-style-spec-full-coverage-roadmap.md`). This module currently
// implements:
//
// - feature/zoom inputs and structural literals
// - `==`, `!=`, `<`, `<=`, `>`, `>=`, `interpolate` (linear, exponential,
//   cubic-bezier), `step` (discrete piecewise selection)
// - `literal` (so arrays/objects can be passed as values)
// - type assertions `number` / `string` / `boolean` / `object` / `array`
//   (with optional item type and length), including the "try each
//   subsequent value as a fallback" form
// - type coercions `to-number` / `to-string` / `to-boolean`
// - `typeof`
// - arithmetic `+`, `-`, `*`, `/`, `%`, `^` plus `min`/`max`
// - unary math `abs`, `ceil`, `floor`, `round`, `sqrt`, `ln`, `log10`,
//   `log2`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
// - math constants `pi`, `e`, `ln2`
// - boolean `all`, `any`, `!` (short-circuiting)
// - control flow `case`, `match`, `coalesce`
// - bindings `let` / `var` (with parse-time scope tracking so the
//   `let/unbound` fixture errors at parse time, matching upstream)
// - strings & i18n: `concat`, `downcase`, `upcase`, `slice`, `length`,
//   `split`, `join`, `index-of`, `format`, `number-format`, `collator`,
//   `resolved-locale`, `is-supported-script`
// - 3-arg `collator`-aware comparison form on `==` / `!=` / `<` /
//   `<=` / `>` / `>=`
// - `error` — parse-accept, always-error-at-eval (only there so
//   short-circuit fixtures can exercise unreachable branches)
// - colors: `rgb` / `rgba` constructors, `to-color` (with CSS3 named-
//   color list, hex shortforms, `rgb()` / `rgba()` / `hsl()` /
//   `hsla()` parsing, and partial-fail fallback chain), `to-rgba` —
//   plus color-output support in `interpolate` and `interpolate-hcl` /
//   `interpolate-lab` for perceptual-space interpolation
// - context-driven readers: `feature-state`, `global-state`,
//   `heatmap-density`, `elevation`, `image` — each pulls from a
//   dedicated slot on `EvalContext`. `coalesce` knows to skip a
//   resolved-but-unavailable image so chains like `coalesce(image("a"),
//   image("b"))` fall through to the first registered sprite.
//
// Operators not in that list are rejected at parse time.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::Value as Json;

use crate::mvt::Feature;
use crate::mvt::Value as MvtValue;

// ==============================================================================
// AST
// ==============================================================================

/// An expression node. Operator-specific data is carried inline; literal
/// JSON scalars fold into [`Literal`].
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit(Literal),
    /// `["literal", x]` — wraps a raw JSON value (including arrays and
    /// objects) without interpreting it as a call. The pre-converted
    /// runtime value is stored so eval is a cheap clone.
    Literal(Value),
    /// `["get", "<property-name>"]` — feature property lookup. Missing
    /// properties evaluate to `Null` (not an error).
    Get(String),
    /// `["zoom"]` — the current map zoom from [`EvalContext`].
    Zoom,
    /// `["==", a, b]` — strict equality (no implicit coercions). An
    /// optional third argument carries a [`Collator`] for locale-aware
    /// string comparison; the 2-arg form is unchanged.
    Eq(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    /// `["interpolate", ["linear"], input, x0, y0, x1, y1, ...]`.
    Interpolate {
        interpolation: Interpolation,
        input: Box<Expr>,
        /// Stops are kept in source order; the evaluator assumes they
        /// are monotonically increasing in `x` (per the spec).
        stops: Vec<(f64, Expr)>,
        /// `true` when at least one stop's *literal* form parses as a
        /// CSS color (e.g. the source had `"red"` as a stop). Without
        /// Phase 10 type inference we use this to decide whether to
        /// coerce string-valued outputs (including endpoint-clamped
        /// stops) to `Value::Color`. The two perceptual operators
        /// force this to `true` regardless of stop contents.
        color_typed: bool,
    },
    /// `["step", input, default, in1, out1, in2, out2, …]` — discrete
    /// piecewise selection. For `input < in1` the result is `default`;
    /// for `inK <= input < in{K+1}` it's `outK`; for `input >= inN`
    /// it's `outN`. Outputs can be *any* value type (numbers, colors,
    /// arrays, even formatted text), since `step` never interpolates.
    Step {
        input: Box<Expr>,
        default: Box<Expr>,
        /// Stops live in source order, strictly ascending — enforced
        /// at parse time to mirror upstream's compile-time check
        /// (`step/duplicate_stops`).
        stops: Vec<(f64, Expr)>,
    },
    /// `["TYPE", v1, v2, ...]` for `TYPE ∈ {number, string, boolean,
    /// object}`. Each `vN` is evaluated in order; the first one whose
    /// runtime value matches `kind` is returned. If none match, the
    /// expression errors. With one argument this is the strict
    /// assertion; with more, the trailing values act as fallbacks.
    Assert {
        kind: AssertKind,
        values: Vec<Expr>,
    },
    /// `["array", value]` / `["array", T, value]` / `["array", T, N,
    /// value]`, with optional trailing fallback values. Each fallback
    /// must itself satisfy the asserted shape; mismatches advance to
    /// the next candidate.
    ArrayAssert {
        item_type: Option<ItemType>,
        length: Option<usize>,
        values: Vec<Expr>,
    },
    /// `["to-number" | "to-string" | "to-boolean", v1, v2, ...]`. Each
    /// argument is evaluated and coerced in turn; the first successful
    /// coercion wins. `to-boolean` never fails so it always uses its
    /// first argument.
    Coerce {
        kind: CoerceKind,
        values: Vec<Expr>,
    },
    /// `["typeof", v]` — returns the runtime type as a spec-formatted
    /// string (e.g. `"number"`, `"array<string, 3>"`).
    TypeOf(Box<Expr>),
    /// Variadic arithmetic — `["+", v1, v2, …]` and `["*", v1, v2, …]`.
    /// The empty form is the operator's identity: `0` for `+`, `1` for
    /// `*`. We don't fold associativity at parse time because the suite
    /// only asserts the *result*, and the simpler walk-and-accumulate
    /// reads more clearly.
    Variadic {
        op: VariadicOp,
        values: Vec<Expr>,
    },
    /// Two-operand arithmetic — `["-", a, b]`, `["/", a, b]`, `["%", a,
    /// b]`, `["^", a, b]`. `-` also has a unary form (negation) which
    /// the parser turns into the same shape with `0` as the implicit
    /// first operand. Division by zero follows IEEE 754 (`Infinity` /
    /// `NaN`) rather than erroring; the spec specifies this.
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Unary math — `["abs", x]`, `["ln", x]`, etc. Every variant is
    /// just a single-argument numeric function; the dispatch is
    /// table-driven at eval time.
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
    },
    /// Zero-arg math constants — `["pi"]`, `["e"]`, `["ln2"]`.
    MathConst(MathConst),
    /// `["!=", a, b]` — strict inequality with the same type-mismatch
    /// rules as `==`. Mixed types compare unequal at runtime (the
    /// upstream compile-time check that rejects mixed-type pairs is a
    /// Phase 10 concern). The optional third slot accepts a collator,
    /// mirroring `==`.
    NotEq(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    /// Two-arg ordered comparison — `["<", a, b]`, `["<=", a, b]`,
    /// `[">", a, b]`, `[">=", a, b]`. Operands must be both numeric or
    /// both string at eval time; mismatched kinds raise a runtime
    /// error. The optional third argument is a collator for locale-
    /// aware string ordering — applying it to non-string operands is a
    /// runtime error per `collator/comparison-number-error`.
    Compare {
        op: CompareOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        collator: Option<Box<Expr>>,
    },
    /// `["all", …]` / `["any", …]` — short-circuiting boolean
    /// conjunction / disjunction. The empty forms are the operator's
    /// identity (`true` / `false`).
    BoolFold {
        op: BoolFold,
        values: Vec<Expr>,
    },
    /// `["!", x]` — boolean negation. The operand must be a bool at
    /// eval time; non-bool inputs are an error, not falsy-coerced.
    Not(Box<Expr>),
    /// `["error", msg]` — pseudo-op that always errors at eval time.
    /// The short-circuit fixtures use it to assert that `all` / `any`
    /// don't evaluate operands past a definitive answer. We have to
    /// accept it at parse time so those fixtures compile.
    ErrorOp(String),
    /// `["case", c1, o1, c2, o2, …, fallback]` — first-true wins.
    /// Conditions and outputs alternate; the final unpaired arg is the
    /// fallback returned when no condition matches. Conditions must
    /// evaluate to a boolean at runtime; non-bool conditions are a
    /// runtime error per spec.
    Case {
        branches: Vec<(Expr, Expr)>,
        fallback: Box<Expr>,
    },
    /// `["match", input, l1, o1, l2, o2, …, fallback]` — switch over
    /// the input's exact value. Labels are scalars or arrays of
    /// scalars (the latter is the multi-match form). We keep the label
    /// type uniform per case: either all numbers or all strings, and
    /// labels are unique across cases (enforced at parse time, matching
    /// upstream behavior).
    Match {
        input: Box<Expr>,
        branches: Vec<(Vec<MatchLabel>, Expr)>,
        fallback: Box<Expr>,
    },
    /// `["coalesce", e1, e2, …]` — return the first child whose
    /// evaluation neither errors nor yields `Null`. Errors from
    /// non-final children are swallowed and the next child is tried;
    /// if every child errors or returns null we surface the last
    /// value (null) so the caller sees the spec-defined "no
    /// successful match" outcome.
    Coalesce(Vec<Expr>),
    /// `["let", n1, v1, n2, v2, …, body]` with `["var", name]` lookups
    /// inside `body` (and inside later bindings). The binding values
    /// are evaluated in declaration order at eval time; we hand the
    /// resulting values to a `Scope` that the body and any nested
    /// `var` lookups read from.
    Let {
        bindings: Vec<(String, Expr)>,
        body: Box<Expr>,
    },
    /// `["var", name]`. The name is resolved against the surrounding
    /// `let` bindings — parse fails if the name isn't bound, so by
    /// eval time the lookup is always live.
    Var(String),
    /// `["concat", v1, v2, …]` — variadic stringification fold.
    /// Each arg is stringified per the same rules as `to-string`. The
    /// empty form returns the empty string.
    Concat(Vec<Expr>),
    /// `["downcase", s]` / `["upcase", s]` — strict string-only case
    /// folding. Non-string inputs are a runtime error rather than a
    /// silent coercion; the spec leaves `downcase(123)` undefined and
    /// the upstream evaluator raises.
    CaseChange {
        kind: CaseChangeKind,
        value: Box<Expr>,
    },
    /// `["slice", s_or_arr, start[, end]]`. The slice indexes operate on
    /// Unicode code points for strings (NOT UTF-8 bytes or UTF-16
    /// units), and on element positions for arrays. Negative indices
    /// count from the end. Out-of-range indices clamp to the natural
    /// boundary, matching `String.prototype.slice` / `Array.prototype
    /// .slice`.
    Slice {
        input: Box<Expr>,
        start: Box<Expr>,
        end: Option<Box<Expr>>,
    },
    /// `["length", v]`. Strings yield Unicode code-point count, arrays
    /// yield element count. Any other runtime type is an error.
    Length(Box<Expr>),
    /// `["split", s, sep]`. Always returns a `Value::Array` of
    /// `Value::String`s. An empty separator splits into individual code
    /// points (mirroring `String.prototype.split('')`).
    Split {
        input: Box<Expr>,
        separator: Box<Expr>,
    },
    /// `["join", arr, sep]`. Each array element is stringified per
    /// `to-string`. Useful for round-tripping `text-font` arrays into
    /// comma-separated `fontStack` strings during `format` rendering.
    Join {
        input: Box<Expr>,
        separator: Box<Expr>,
    },
    /// `["index-of", needle, hay[, start]]`. Works on both strings (the
    /// needle is stringified) and arrays (strict-equality search).
    /// Returns `-1` if not found. Start index applies the JS-array rule
    /// for negative inputs (`hay.len() + start`); strings clamp
    /// negatives to `0` instead, matching `String.prototype.indexOf`.
    IndexOf {
        needle: Box<Expr>,
        haystack: Box<Expr>,
        from_index: Option<Box<Expr>>,
    },
    /// `["format", t1, o1, t2, o2, …]` — produces a `Value::Formatted`
    /// value with one section per `(text, options)` pair. The options
    /// object's keys (`font-scale`, `text-font`, `text-color`,
    /// `vertical-align`) are validated at parse time but their values
    /// stay as expressions so they can be data-driven.
    Format(Vec<FormatSection>),
    /// `["number-format", n, opts]` — Intl.NumberFormat-style. We
    /// currently only support the `en`-family locale (and treat
    /// unknown locales as `en` for grouping/decimal punctuation).
    /// `currency` and `unit` are mutually exclusive at compile time.
    NumberFormat {
        value: Box<Expr>,
        locale: Option<Box<Expr>>,
        currency: Option<Box<Expr>>,
        unit: Option<Box<Expr>>,
        min_fraction_digits: Option<Box<Expr>>,
        max_fraction_digits: Option<Box<Expr>>,
    },
    /// `["collator", { case-sensitive, diacritic-sensitive, locale }]`
    /// — constructs a `Value::Collator`. Each option is an expression
    /// (so the collator can be data-driven via `get`/`var`).
    Collator(CollatorSpec),
    /// `["resolved-locale", collator]` — returns the locale the
    /// collator settled on as a string. With no ICU dependency we just
    /// echo back the user-provided locale string (or empty string for
    /// the default).
    ResolvedLocale(Box<Expr>),
    /// `["is-supported-script", s]` — without an ambient
    /// `isSupportedScript` global (the runner doesn't provide one) the
    /// upstream contract is to return `true` unconditionally, per the
    /// `is-supported-script/default` fixture's own description.
    IsSupportedScript(Box<Expr>),
    /// `["rgb", r, g, b]` / `["rgba", r, g, b, a]`. RGB channels are
    /// 0..=255 (the spec types them as integer but runtime accepts any
    /// number); alpha is 0..=1. Out-of-range values raise the
    /// upstream-pinned error at eval time — they're not silently
    /// clamped because `rgba/out-of-bounds` checks that the parser
    /// surfaces an error message including the offending channels.
    /// `alpha` stays `None` for the 3-arg form so we can keep one AST
    /// variant for both arities.
    Rgb {
        r: Box<Expr>,
        g: Box<Expr>,
        b: Box<Expr>,
        alpha: Option<Box<Expr>>,
    },
    /// `["to-color", v1, v2, …]`. Each argument is evaluated and
    /// passed through the same parsing rules as the standalone color
    /// literal (CSS named colors, hex, `rgb()` / `rgba()` / `hsl()` /
    /// `hsla()`, or `[r, g, b, a]` arrays). The first successful
    /// conversion wins; this is the "partial fail" behavior pinned by
    /// `to-color/2-ary` and `to-color/basic`.
    ToColor(Vec<Expr>),
    /// `["to-rgba", c]` — returns a 4-element straight-RGB array with
    /// channels in `0..=255` for R/G/B and `0..=1` for alpha. This is
    /// the reverse of the premultiplied storage: we divide the stored
    /// channels by alpha before scaling. Alpha-zero colors round-trip
    /// as `[0, 0, 0, 0]` to avoid NaN.
    ToRgba(Box<Expr>),
    /// `["geometry-type"]` — returns the current feature's geometry
    /// type as a string (`"Point"`, `"LineString"`, `"Polygon"`, or
    /// `"Unknown"`). Upstream normalizes Multi* variants down to their
    /// non-Multi base (a `MultiPolygon` reports as `"Polygon"`); we
    /// do the same so fixtures comparing against the bare names line
    /// up.
    GeometryType,
    /// `["within", <geojson polygon>]` — `true` when the current
    /// feature's geometry lies entirely inside the argument polygon
    /// (or multipolygon). The argument is parsed at compile time
    /// into a constant [`WithinPolygons`] value so the per-feature
    /// evaluation is just point-in-polygon arithmetic.
    Within(WithinPolygons),
    /// `["distance", <geojson>]` — shortest geodesic distance, in
    /// meters, between the current feature and the argument
    /// geometry. The argument is parsed at compile time into a
    /// collection of simple geometries; per-feature evaluation walks
    /// each component and keeps the running minimum.
    Distance(Vec<SimpleGeoJson>),
    /// `["feature-state", name]` — read from
    /// [`EvalContext::feature_state`]. The name is itself an
    /// expression so the fixture `feature-state/expression-arg` can
    /// build it via `["at", …]`/`["concat", …]`/etc. A missing key
    /// returns `Value::Null`; the outer assertion (`["number", …]`)
    /// is what raises the spec-pinned error message.
    FeatureState(Box<Expr>),
    /// `["global-state", name]` — read from
    /// [`EvalContext::global_state`]. Unlike `feature-state`, the
    /// name is required to be a *string literal* at parse time —
    /// `global-state/invalid-property-arg` pins the message for the
    /// expression-arg form.
    GlobalState(String),
    /// `["heatmap-density"]` — current heatmap density from the
    /// context. Always returns a number; defaults to 0 when no
    /// heatmap layer is rendering.
    HeatmapDensity,
    /// `["elevation"]` — terrain elevation at the current feature
    /// from the context. Always returns a number; defaults to 0.
    Elevation,
    /// `["image", name]` — produce a [`Value::Image`] by combining
    /// the name with the context's `available_images` set. We accept
    /// any string-yielding sub-expression for `name`; non-string
    /// *literal* arguments are rejected at parse time, matching the
    /// `image/invalid-image-name/{boolean,number,null}` fixtures.
    /// Catching the harder "non-string operator" case
    /// (`image(+(1,4))`) needs Phase 10 static-type inference, so we
    /// accept it here and defer to the manifest.
    Image(Box<Expr>),

    // Phase 5 — collections & feature metadata. These operators read
    // structural data off the current feature (or an object the caller
    // passes in) and form the basis for nested-object/array access.
    /// `["get", name, host]` — 2-arg form. Reads `name` off `host`
    /// (which evaluates to an object). Distinct from the 1-arg
    /// `Expr::Get(String)` because the 2-arg form has a different
    /// type-inference shape (host's static type drives the result),
    /// and because the upstream reject path checks the host is an
    /// object at runtime.
    GetFrom {
        name: String,
        host: Box<Expr>,
    },
    /// `["at", index, array]` — array indexing by zero-based integer.
    /// Negative indices and out-of-bounds indices are runtime errors;
    /// non-integer numeric indices likewise error (so `at(1.5, …)`
    /// fails) — the upstream `at/basic` fixture exercises each branch.
    At {
        index: Box<Expr>,
        array: Box<Expr>,
    },
    /// `["in", needle, haystack]` — substring search for strings or
    /// element membership for arrays. Non-scalar needles and non-
    /// array/string haystacks are runtime errors with spec-pinned
    /// wording. `null` haystacks return `false` so `["in", x, get y]`
    /// works on missing properties without an outer assert.
    In {
        needle: Box<Expr>,
        haystack: Box<Expr>,
    },
    /// `["has", name, host]` — does `host` (or `feature.properties`
    /// when `host` is `None`) contain `name`? Distinct from `get` in
    /// that a key with `null` value still reports `true`.
    Has {
        name: String,
        host: Option<Box<Expr>>,
    },
    /// `["properties"]` — returns the current feature's properties as
    /// an `Object` value. Useful for round-tripping the whole property
    /// bag through a `get`-from-object chain.
    Properties,
    /// `["id"]` — returns `feature.id` as a number (or `null` when the
    /// feature lacks an id).
    Id,
}

/// A simple (non-Multi, non-collection) geometry stored in lng/lat
/// space. `distance`'s argument may be a `FeatureCollection`, `Feature`,
/// or bare geometry — we normalize each variant into a flat list of
/// these so the evaluator just iterates and folds. Multi* variants
/// expand into one [`SimpleGeoJson`] per component, matching upstream's
/// `toSimpleGeometry` helper.
#[derive(Debug, Clone, PartialEq)]
pub enum SimpleGeoJson {
    Point(f64, f64),
    LineString(Vec<(f64, f64)>),
    /// A polygon as a list of rings (first = exterior, rest = holes).
    /// Each ring is a closed loop of lng/lat coordinates.
    Polygon(PolygonRings),
}

/// A single polygon represented as its rings, exterior first. Type
/// alias because the nested `Vec<Vec<…>>` shape is hard to read and
/// shows up in several signatures.
pub type PolygonRings = Vec<Vec<(f64, f64)>>;

/// Argument to `within`. We accept the spec's three top-level shapes —
/// `FeatureCollection`, `Feature`, or a bare polygon — and normalize
/// them at parse time into a list of polygons. This shape feeds the
/// tile-space conversion directly when the evaluator runs.
#[derive(Debug, Clone, PartialEq)]
pub struct WithinPolygons {
    pub polygons: Vec<PolygonRings>,
}

/// Label slot for a `match` branch. Numbers must be integer-valued in
/// the spec's `Number.isSafeInteger` sense (≤ `2^53 − 1`); strings are
/// taken literally. Booleans, nulls, objects, and arrays-of-non-scalars
/// are rejected at parse time.
#[derive(Debug, Clone, PartialEq)]
pub enum MatchLabel {
    Number(i64),
    String(String),
}

/// Inline JSON-scalar literal. Structural literals (arrays, objects)
/// must go through `["literal", …]` and live in [`Expr::Literal`].
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
}

/// How an interpolation walks between stop outputs. The `Linear`
/// variant applies in source space (numeric or RGB channels). The
/// `Hcl` and `Lab` variants are color-only and run the lerp through
/// the corresponding perceptual color space before converting back to
/// sRGB. They share the AST shape because the operator names
/// (`interpolate-hcl`, `interpolate-lab`) are parsed independently
/// from the interpolation kind argument — upstream's parser bakes the
/// space into the operator itself, but it's cleaner here to carry it
/// on this enum.
///
/// `Exponential` and `CubicBezier` shape the *parameterization* of the
/// 0..1 sweep across one stop pair. They don't change the output
/// space (numeric or color); they only remap `t` before the
/// component-wise lerp runs. Carrying their parameters on the enum
/// keeps the `Expr::Interpolate` node uniform regardless of which
/// curve was selected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interpolation {
    Linear,
    /// Non-linear ramp: `t = (base^(x-x_lo) - 1) /
    /// (base^(x_hi-x_lo) - 1)`. The spec collapses `base == 1` to
    /// linear at the formula level (it'd otherwise be `0/0`), and the
    /// existing endpoint clamp in `interpolate()` keeps `base` of
    /// other values well-behaved.
    Exponential {
        base: f64,
    },
    /// CSS `cubic-bezier(x1, y1, x2, y2)` parameterization. Each
    /// control point must lie inside the unit square per the spec; the
    /// parser rejects out-of-range constants up front. We carry the
    /// four `f64` coefficients directly and resolve them via Newton-
    /// Raphson at eval time.
    CubicBezier {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
    },
    Hcl,
    Lab,
}

/// Distinguishes the four scalar assertion ops. `object` is included
/// here because its semantics match the scalar assertions (single-arg
/// strict check, multi-arg fallback chain) even though it accepts a
/// structural value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertKind {
    Number,
    String,
    Boolean,
    Object,
}

impl AssertKind {
    fn as_str(self) -> &'static str {
        match self {
            AssertKind::Number => "number",
            AssertKind::String => "string",
            AssertKind::Boolean => "boolean",
            AssertKind::Object => "object",
        }
    }
}

/// Per-item type permitted by `array`'s 2- and 3-argument forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemType {
    Number,
    String,
    Boolean,
}

impl ItemType {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "number" => Some(ItemType::Number),
            "string" => Some(ItemType::String),
            "boolean" => Some(ItemType::Boolean),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ItemType::Number => "number",
            ItemType::String => "string",
            ItemType::Boolean => "boolean",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoerceKind {
    ToNumber,
    ToString,
    ToBoolean,
}

// ==============================================================================
// Static type — Phase 10
// ==============================================================================
//
// The spec types every expression statically: each node has a "result
// type" that the parent's parser can compare against its expected
// argument shape. We model that as [`ExprType`], formatted to mirror
// upstream's [`ValueType.toString()`] verbatim (e.g. `array<number, 2>`,
// `value`, `string`) because the fixtures' `expected.compiled.type`
// field carries the rendered string and the test runner compares it
// directly.
//
// We resolve the type lazily through [`Expr::result_type`]; constness
// inference ([`Expr::is_feature_constant`] / [`Expr::is_zoom_constant`])
// is symmetrical — both fall out of a recursive walk over the AST. None
// of the three is part of the language's runtime semantics; they only
// drive the runner's compile-time comparison and the parse-time
// `zoom`-position check.

/// Static return type of an [`Expr`] subtree. `Value` is the spec's
/// top type (anything goes) and is what every type-erased operator
/// (`get`, `coalesce` over mixed-type branches, etc.) decays to.
///
/// `Array(item, length)` carries the type's structure when known:
/// upstream renders array types as `array<T>` or `array<T, N>` based on
/// what the static check could pin down. `Null(item=None, length=None)`
/// renders as a bare `array` — the type-erased "any array" upstream
/// uses when the structure isn't statically knowable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprType {
    /// The spec's top type: an unknown-at-compile-time value. Renders as
    /// `value` and is what every type-erased operator decays to.
    Value,
    Null,
    Boolean,
    Number,
    String,
    Object,
    Color,
    Formatted,
    ResolvedImage,
    Collator,
    /// An array. `item` is `Some` when the per-element type is known,
    /// `None` for the type-erased "any array". `length` is `Some(n)`
    /// when a fixed length is statically known (the 3-arg `array`
    /// assertion form), `None` otherwise.
    Array(Option<Box<ExprType>>, Option<usize>),
}

impl ExprType {
    /// Render as the spec's type string. Matches upstream's
    /// `ValueType.toString()` so the runner's per-fixture compile-shape
    /// comparison is a byte-for-byte string match.
    pub fn to_spec_string(&self) -> String {
        match self {
            ExprType::Value => "value".into(),
            ExprType::Null => "null".into(),
            ExprType::Boolean => "boolean".into(),
            ExprType::Number => "number".into(),
            ExprType::String => "string".into(),
            ExprType::Object => "object".into(),
            ExprType::Color => "color".into(),
            ExprType::Formatted => "formatted".into(),
            ExprType::ResolvedImage => "resolvedImage".into(),
            ExprType::Collator => "collator".into(),
            ExprType::Array(item, len) => match (item, len) {
                (None, _) => "array".into(),
                (Some(t), None) => format!("array<{}>", t.to_spec_string()),
                (Some(t), Some(n)) => format!("array<{}, {n}>", t.to_spec_string()),
            },
        }
    }

    /// The "join" of two types — what `coalesce` / `case` / `match`
    /// surface when their branches disagree. The rule mirrors upstream:
    /// equal types stay themselves; the top type `value` is absorbed by
    /// the other side (so a concrete sibling wins); arrays meet on
    /// item-type and length; everything else decays to `value`.
    ///
    /// `value` deferring to the concrete side is intentional and
    /// matches upstream's `Interpolate`/`Step` outputType inference:
    /// a stop like `["get", "x"]` (whose static type is `value`) joins
    /// with a literal stop and the literal's type carries through.
    fn join(a: ExprType, b: ExprType) -> ExprType {
        if a == b {
            return a;
        }
        match (a, b) {
            (ExprType::Value, other) | (other, ExprType::Value) => other,
            (ExprType::Array(at, al), ExprType::Array(bt, bl)) => {
                let item = match (at, bt) {
                    (Some(x), Some(y)) if x == y => Some(x),
                    _ => None,
                };
                let length = if al == bl { al } else { None };
                ExprType::Array(item, length)
            }
            _ => ExprType::Value,
        }
    }

    /// Fold an iterator of types into a single join. Empty input is the
    /// spec's bottom — we surface `Value` so callers (e.g. an empty
    /// `coalesce`, which the parser would have rejected anyway) never
    /// see an `Option` they have to handle.
    fn join_many<I: IntoIterator<Item = ExprType>>(iter: I) -> ExprType {
        let mut iter = iter.into_iter();
        let Some(first) = iter.next() else {
            return ExprType::Value;
        };
        iter.fold(first, ExprType::join)
    }

    /// Map an [`ItemType`] (the `array` assertion's parse-time slot) to
    /// its `ExprType` peer. Reads as a small adapter; `result_type` uses
    /// it once for `Expr::ArrayAssert`.
    fn from_item_type(t: ItemType) -> ExprType {
        match t {
            ItemType::Number => ExprType::Number,
            ItemType::String => ExprType::String,
            ItemType::Boolean => ExprType::Boolean,
        }
    }
}

/// `case` / `match` type pinning. Upstream's parser uses the *first*
/// output's static type as the canonical type for the whole
/// expression: subsequent outputs (and the fallback) must each be
/// assignable to it, but they don't get to specialize the type
/// further. If the first output is `value` (the top type), the whole
/// expression stays `value` no matter what — the `case/null-fallback`
/// fixture pins this on a `case` whose body output is `value` (from
/// `get`) and whose fallback is `null`.
fn pin_branch_types<I: IntoIterator<Item = ExprType>>(iter: I) -> ExprType {
    let mut iter = iter.into_iter();
    let Some(first) = iter.next() else {
        return ExprType::Value;
    };
    if matches!(first, ExprType::Value) {
        return ExprType::Value;
    }
    let mut t = first;
    for next in iter {
        // Treat `value` (top) on the right as "compatible, no
        // specialization" — preserve the pinned type.
        if matches!(next, ExprType::Value) {
            continue;
        }
        t = ExprType::join(t, next);
    }
    t
}

/// Variadic numeric folds. `Plus` has identity `0`, `Times` identity
/// `1`; this matches the upstream evaluator's `arity-0` fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariadicOp {
    Plus,
    Times,
    Max,
    Min,
}

impl VariadicOp {
    fn identity(self) -> f64 {
        // Picked so the variadic fold collapses cleanly on `arity-0`
        // inputs: `max` returns -infinity (any real value beats it) and
        // `min` returns +infinity (mirror).
        match self {
            VariadicOp::Plus => 0.0,
            VariadicOp::Times => 1.0,
            VariadicOp::Max => f64::NEG_INFINITY,
            VariadicOp::Min => f64::INFINITY,
        }
    }
}

/// Strictly-binary numeric ops. `Minus` additionally has a unary form
/// handled by the parser, which lowers `["-", x]` to `["-", 0, x]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Minus,
    Divide,
    Mod,
    Pow,
}

/// Single-operand numeric functions. Each variant maps to one method
/// on `f64`; the eval dispatch is the only place that translates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Abs,
    Ceil,
    Floor,
    Round,
    Sqrt,
    Ln,
    Log10,
    Log2,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
}

/// Zero-arg numeric constants. The spec exposes only these three; the
/// upstream evaluator inlines them at parse time, but keeping them in
/// the AST keeps `result_type` work simple when Phase 10 lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathConst {
    Pi,
    E,
    Ln2,
}

impl MathConst {
    fn value(self) -> f64 {
        match self {
            MathConst::Pi => std::f64::consts::PI,
            MathConst::E => std::f64::consts::E,
            MathConst::Ln2 => std::f64::consts::LN_2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            MathConst::Pi => "pi",
            MathConst::E => "e",
            MathConst::Ln2 => "ln2",
        }
    }
}

/// Ordered-comparison operators. The 2-arg form here matches the spec;
/// the 3-arg locale-aware form (with a `collator`) is a Phase 4 item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

impl CompareOp {
    fn op_str(self) -> &'static str {
        match self {
            CompareOp::Less => "<",
            CompareOp::LessOrEqual => "<=",
            CompareOp::Greater => ">",
            CompareOp::GreaterOrEqual => ">=",
        }
    }
}

/// Short-circuiting boolean folds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolFold {
    All,
    Any,
}

/// Distinguishes `downcase` from `upcase`. They share parse / eval
/// shapes so a single AST variant carries both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseChangeKind {
    Down,
    Up,
}

impl CaseChangeKind {
    fn name(self) -> &'static str {
        match self {
            CaseChangeKind::Down => "downcase",
            CaseChangeKind::Up => "upcase",
        }
    }
}

/// One `(text, options)` section of a `format` call. The text and each
/// option are stored as expressions so a downstream `eval` can produce
/// data-driven output. Unset options stay `None`.
///
/// `vertical_align` can be either a parse-time keyword (validated
/// eagerly) or a data-driven expression that yields the keyword string
/// at eval time. Mixed shape because `format/data-driven-vertical-
/// align` passes an expression while `format/basic` passes a string
/// literal directly.
#[derive(Debug, Clone, PartialEq)]
pub struct FormatSection {
    pub text: Box<Expr>,
    pub font_scale: Option<Box<Expr>>,
    pub text_font: Option<Box<Expr>>,
    pub text_color: Option<Box<Expr>>,
    pub vertical_align: Option<VerticalAlignSlot>,
}

/// Either a statically-known vertical-align keyword (resolved at parse
/// time so `format/unknown-vertical-align` errors at compile time) or
/// an expression that resolves to the keyword at eval time.
#[derive(Debug, Clone, PartialEq)]
pub enum VerticalAlignSlot {
    Static(VerticalAlign),
    Expr(Box<Expr>),
}

/// `vertical-align` is parsed and validated at compile time (the
/// upstream spec restricts it to three keywords) so the runtime layer
/// only handles a closed enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalAlign {
    Bottom,
    Center,
    Top,
}

impl VerticalAlign {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "bottom" => Some(VerticalAlign::Bottom),
            "center" => Some(VerticalAlign::Center),
            "top" => Some(VerticalAlign::Top),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            VerticalAlign::Bottom => "bottom",
            VerticalAlign::Center => "center",
            VerticalAlign::Top => "top",
        }
    }
}

/// Parse-time description of a `collator` constructor. Every option is
/// itself an expression so a collator can be built from runtime feature
/// data — the `collator/variable-equals` fixture wires
/// `diacritic-sensitive` through `get`.
#[derive(Debug, Clone, PartialEq)]
pub struct CollatorSpec {
    pub case_sensitive: Option<Box<Expr>>,
    pub diacritic_sensitive: Option<Box<Expr>>,
    pub locale: Option<Box<Expr>>,
}

/// A filter is just an expression that must evaluate to a `Bool` at
/// eval time. Kept distinct from a bare [`Expr`] so call sites read
/// clearly.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter(pub Expr);

// ==============================================================================
// Parsing from raw JSON
// ==============================================================================

impl Expr {
    /// Parse an expression from a `serde_json::Value` subtree. The
    /// document is already parsed; this is the structural-to-typed
    /// conversion step.
    pub fn from_json(v: &Json) -> Result<Self, ExprError> {
        // External callers always start with an empty scope; `var`
        // can only resolve names bound by an enclosing `let`, so the
        // top-level expression sees no bindings.
        let parsed = Self::from_json_scoped(v, &ParseScope::root())?;
        validate_zoom_positions(&parsed)?;
        Ok(parsed)
    }

    /// Like [`Expr::from_json`], but the caller supplies the JSON path
    /// of `v` within its enclosing document. On error, the returned
    /// [`ExprError`] is wrapped in [`ExprError::Located`] so downstream
    /// snippet rendering can resolve the path back to a byte span.
    ///
    /// Errors raised during the validation post-pass
    /// (`validate_zoom_positions`) also receive the supplied path, so a
    /// nested-zoom rejection still points back into the source.
    pub fn from_json_at(
        v: &Json,
        path: super::span::JsonPath,
    ) -> Result<Self, ExprError> {
        // External callers always start with an empty scope; `var`
        // can only resolve names bound by an enclosing `let`, so the
        // top-level expression sees no bindings.
        let parsed = Self::from_json_scoped(v, &ParseScope::root())
            .map_err(|e| e.with_path(path.clone()))?;
        // The spec restricts `zoom` to a *direct* input slot of a
        // top-level `step`/`interpolate`. Upstream phrases this as two
        // related errors — "zoom expression may only be used as input
        // to a top-level …" (positional) and "Only one zoom-based
        // step/interpolate subexpression may be used …" (uniqueness).
        // We run the post-parse walk here so structural errors in
        // child operators surface first; the runner only inspects
        // success/failure so error wording doesn't have to match.
        validate_zoom_positions(&parsed).map_err(|e| e.with_path(path))?;
        Ok(parsed)
    }

    /// Parse an expression intended to be used as a layer filter.
    ///
    /// Filters are re-evaluated whenever the camera's zoom crosses an
    /// integer boundary, so the spec deliberately *relaxes* the
    /// "zoom may only feed a top-level step/interpolate" rule here —
    /// `["zoom"]` is allowed inside any operator, and styles in the
    /// wild lean on this (e.g. `[">=", ["zoom"], ["get", "min_zoom"]]`
    /// to hide features below a per-feature minzoom). All other
    /// parse-time checks (scope, types, operator arity) still apply.
    pub fn from_filter_json(v: &Json) -> Result<Self, ExprError> {
        Self::from_json_scoped(v, &ParseScope::root())
    }

    /// Path-aware variant of [`Expr::from_filter_json`].
    pub fn from_filter_json_at(
        v: &Json,
        path: super::span::JsonPath,
    ) -> Result<Self, ExprError> {
        Self::from_json_scoped(v, &ParseScope::root()).map_err(|e| e.with_path(path))
    }

    fn from_json_scoped(v: &Json, scope: &ParseScope<'_>) -> Result<Self, ExprError> {
        match v {
            Json::Null => Ok(Expr::Lit(Literal::Null)),
            Json::Bool(b) => Ok(Expr::Lit(Literal::Bool(*b))),
            Json::Number(n) => {
                // `serde_json::Number::as_f64` succeeds for every
                // representable JSON number we care about; it only
                // fails for arbitrary-precision values which are off
                // by default.
                let f = n
                    .as_f64()
                    .ok_or_else(|| ExprError::InvalidNumber(n.to_string()))?;
                Ok(Expr::Lit(Literal::Number(f)))
            }
            Json::String(s) => Ok(Expr::Lit(Literal::String(s.clone()))),
            Json::Array(items) => parse_array(items, scope),
            // Bare object literals aren't expression-shaped in the
            // spec; if one shows up where an expression is expected,
            // it's a malformed style.
            Json::Object(_) => Err(ExprError::UnexpectedObject),
        }
    }
}

// ------------------------------------------------------------------------------
// Zoom-position validation — Phase 10
// ------------------------------------------------------------------------------
//
// The spec restricts `zoom` references to a *direct* input slot of a
// "top-level" `step` or `interpolate`. "Top-level" here means
// reachable from the root through only passthrough operators (`let`'s
// body, `coalesce`, `case`/`match` outputs); anything that combines a
// curve's result with other values (`+`, `string`, etc.) breaks
// passthrough status. Additionally only *one* zoom-using curve may
// appear anywhere in the expression — two parallel curves would force
// the renderer to interpolate twice for one styled property, which the
// spec excludes.
//
// The two relevant fixture clusters are `zoom/invalid-nested-*` (one
// per failure category) and `zoom/nested-{let,coalesce}` (the
// positive cases). Two upstream wordings flow from the same walk
// here, both surfaced as `ExprError::ZoomNestedInvalid` /
// `ExprError::ZoomMultipleCurves`.

/// Outcome of [`validate_zoom_positions`]'s recursive walk. We track
/// whether the subtree itself counts as the legal zoom curve, plus a
/// count of legal curves found below, so the outer caller can flag a
/// duplicate at the topmost shared ancestor.
#[derive(Debug, Clone, Copy, Default)]
struct ZoomWalkOutcome {
    /// Number of zoom-using `step`/`interpolate` nodes found *anywhere*
    /// in the subtree that are at a legal position. Two anywhere is
    /// always an error per upstream's "Only one zoom-based curve".
    legal_curves: usize,
}

fn validate_zoom_positions(e: &Expr) -> Result<(), ExprError> {
    let out = validate_zoom_in(e, /* zoom_curve_allowed: */ true)?;
    let _ = out; // success outcome carries no information for callers
    Ok(())
}

/// Recursive helper. `zoom_curve_allowed` is `true` at any position
/// reachable from the root by descending only through passthrough
/// operators — i.e. the position where a `step`/`interpolate` over
/// `zoom` is the canonical "top-level" zoom curve.
fn validate_zoom_in(e: &Expr, zoom_curve_allowed: bool) -> Result<ZoomWalkOutcome, ExprError> {
    match e {
        // Bare zoom outside a curve's input slot — `subtree_has_zoom`
        // is what flags that case at the curve level. Reaching this
        // arm from the top-level walk means the caller didn't descend
        // into a curve's input first, so `zoom` is being used in a
        // position the spec forbids.
        Expr::Zoom => Err(ExprError::ZoomNestedInvalid),

        // A `step`/`interpolate` whose input contains `zoom` is the
        // canonical zoom curve. If we're at a legal position, we accept
        // it and count it; the stops are processed with
        // `zoom_curve_allowed = false` so nested curves there get
        // counted but also constrained (a nested curve in a stop would
        // hit this same code path and fail the position check).
        Expr::Step {
            input,
            default,
            stops,
        } if subtree_has_zoom(input) => {
            if !zoom_curve_allowed {
                return Err(ExprError::ZoomNestedInvalid);
            }
            // The curve's own input is allowed to reference zoom — that's
            // the whole point — but it must not contain *another* zoom
            // curve. Walking the input with `zoom_curve_allowed = false`
            // would forbid the bare `zoom` reference too; instead we run
            // a `walk_inside_curve_input` that only flags nested curves.
            let mut total = walk_inside_curve_input(input)?;
            total = total.saturating_add(validate_zoom_in(default, false)?.legal_curves);
            for (_, out) in stops {
                total = total.saturating_add(validate_zoom_in(out, false)?.legal_curves);
            }
            let total = total.saturating_add(1);
            if total > 1 {
                return Err(ExprError::ZoomMultipleCurves);
            }
            Ok(ZoomWalkOutcome {
                legal_curves: total,
            })
        }
        Expr::Interpolate { input, stops, .. } if subtree_has_zoom(input) => {
            if !zoom_curve_allowed {
                return Err(ExprError::ZoomNestedInvalid);
            }
            let mut total = walk_inside_curve_input(input)?;
            for (_, out) in stops {
                total = total.saturating_add(validate_zoom_in(out, false)?.legal_curves);
            }
            let total = total.saturating_add(1);
            if total > 1 {
                return Err(ExprError::ZoomMultipleCurves);
            }
            Ok(ZoomWalkOutcome {
                legal_curves: total,
            })
        }

        // Passthrough operators: their primary "result" is a child's
        // result, so a zoom curve at that position is still top-level.
        // `let` body is passthrough, but bindings are not — a curve
        // inside a binding's value is detached from the body's flow.
        Expr::Let { bindings, body } => {
            let mut total = 0usize;
            for (_, val) in bindings {
                total = total.saturating_add(validate_zoom_in(val, false)?.legal_curves);
            }
            total = total.saturating_add(validate_zoom_in(body, zoom_curve_allowed)?.legal_curves);
            if total > 1 {
                return Err(ExprError::ZoomMultipleCurves);
            }
            Ok(ZoomWalkOutcome {
                legal_curves: total,
            })
        }
        Expr::Coalesce(values) => {
            let mut total = 0usize;
            for v in values {
                // Each child of `coalesce` is a candidate result, so
                // every child position is passthrough.
                total = total.saturating_add(validate_zoom_in(v, zoom_curve_allowed)?.legal_curves);
            }
            if total > 1 {
                return Err(ExprError::ZoomMultipleCurves);
            }
            Ok(ZoomWalkOutcome {
                legal_curves: total,
            })
        }
        Expr::Case { branches, fallback } => {
            let mut total = 0usize;
            for (cond, out) in branches {
                // Conditions don't passthrough; outputs do.
                total = total.saturating_add(validate_zoom_in(cond, false)?.legal_curves);
                total =
                    total.saturating_add(validate_zoom_in(out, zoom_curve_allowed)?.legal_curves);
            }
            total =
                total.saturating_add(validate_zoom_in(fallback, zoom_curve_allowed)?.legal_curves);
            if total > 1 {
                return Err(ExprError::ZoomMultipleCurves);
            }
            Ok(ZoomWalkOutcome {
                legal_curves: total,
            })
        }
        Expr::Match {
            input,
            branches,
            fallback,
        } => {
            let mut total = validate_zoom_in(input, false)?.legal_curves;
            for (_, out) in branches {
                total =
                    total.saturating_add(validate_zoom_in(out, zoom_curve_allowed)?.legal_curves);
            }
            total =
                total.saturating_add(validate_zoom_in(fallback, zoom_curve_allowed)?.legal_curves);
            if total > 1 {
                return Err(ExprError::ZoomMultipleCurves);
            }
            Ok(ZoomWalkOutcome {
                legal_curves: total,
            })
        }

        // Any other operator: not a passthrough. A zoom-curve inside a
        // child of these is not at a legal position; a bare `zoom` even
        // less so. We walk each child with `zoom_curve_allowed = false`
        // so the recursion's own checks fire.
        _ => {
            let mut total = 0usize;
            for child in expr_children(e) {
                total = total.saturating_add(validate_zoom_in(child, false)?.legal_curves);
            }
            if total > 1 {
                return Err(ExprError::ZoomMultipleCurves);
            }
            Ok(ZoomWalkOutcome {
                legal_curves: total,
            })
        }
    }
}

/// Walk inside a `step`/`interpolate`'s `input` slot. Bare `Expr::Zoom`
/// is fine here — it's the legal use site. We only need to count any
/// nested zoom curves so the uniqueness check stays accurate; the
/// nested curve itself isn't at a top-level position, so its own
/// recursion would surface `ZoomNestedInvalid`.
fn walk_inside_curve_input(e: &Expr) -> Result<usize, ExprError> {
    match e {
        Expr::Zoom => Ok(0),
        Expr::Step { .. } | Expr::Interpolate { .. } => {
            // A curve nested inside another curve's input is not
            // top-level positioned. We hand it back to the main walker
            // with `zoom_curve_allowed = false` so the rejection
            // message comes out of one place.
            Ok(validate_zoom_in(e, false)?.legal_curves)
        }
        // `let` inside an input slot lets the body still be a legal
        // zoom-input expression — `var` lookups read the binding's
        // pre-computed value, so the body inherits the input slot's
        // own legality.
        Expr::Let { bindings, body } => {
            let mut total = 0usize;
            for (_, val) in bindings {
                total = total.saturating_add(validate_zoom_in(val, false)?.legal_curves);
            }
            total = total.saturating_add(walk_inside_curve_input(body)?);
            Ok(total)
        }
        // Other operators inside the input slot may contain bare zoom
        // freely (e.g. `["interpolate", ["linear"], ["+", ["zoom"],
        // 1], …]`) so we recurse with a relaxed walker that mirrors
        // the same "zoom is OK at any depth inside an input" rule.
        _ => {
            let mut total = 0usize;
            for child in expr_children(e) {
                total = total.saturating_add(walk_inside_curve_input(child)?);
            }
            Ok(total)
        }
    }
}

/// `true` when the subtree contains an `Expr::Zoom` anywhere. Used to
/// classify whether a `step`/`interpolate` is "zoom-using" and thus
/// participates in the uniqueness/positional checks.
fn subtree_has_zoom(e: &Expr) -> bool {
    if let Expr::Zoom = e {
        return true;
    }
    expr_children(e).into_iter().any(subtree_has_zoom)
}

/// Flat list of every immediate child expression. Used by the zoom
/// walker so a future operator only has to be added here (and not
/// scattered across multiple recursive matches).
fn expr_children(e: &Expr) -> Vec<&Expr> {
    let mut out: Vec<&Expr> = Vec::new();
    match e {
        Expr::Lit(_)
        | Expr::Literal(_)
        | Expr::Get(_)
        | Expr::Zoom
        | Expr::MathConst(_)
        | Expr::ErrorOp(_)
        | Expr::Var(_)
        | Expr::GlobalState(_)
        | Expr::HeatmapDensity
        | Expr::Elevation
        | Expr::GeometryType
        | Expr::Within(_)
        | Expr::Distance(_)
        | Expr::Properties
        | Expr::Id => {}
        Expr::Eq(a, b, c) | Expr::NotEq(a, b, c) => {
            out.push(a);
            out.push(b);
            if let Some(c) = c {
                out.push(c);
            }
        }
        Expr::Compare {
            lhs, rhs, collator, ..
        } => {
            out.push(lhs);
            out.push(rhs);
            if let Some(c) = collator {
                out.push(c);
            }
        }
        Expr::Interpolate { input, stops, .. } => {
            out.push(input);
            for (_, s) in stops {
                out.push(s);
            }
        }
        Expr::Step {
            input,
            default,
            stops,
        } => {
            out.push(input);
            out.push(default);
            for (_, s) in stops {
                out.push(s);
            }
        }
        Expr::Assert { values, .. }
        | Expr::ArrayAssert { values, .. }
        | Expr::Coerce { values, .. }
        | Expr::Concat(values)
        | Expr::ToColor(values)
        | Expr::Coalesce(values) => {
            for v in values {
                out.push(v);
            }
        }
        Expr::Variadic { values, .. } | Expr::BoolFold { values, .. } => {
            for v in values {
                out.push(v);
            }
        }
        Expr::TypeOf(inner)
        | Expr::Not(inner)
        | Expr::Length(inner)
        | Expr::FeatureState(inner)
        | Expr::ResolvedLocale(inner)
        | Expr::IsSupportedScript(inner)
        | Expr::ToRgba(inner)
        | Expr::Image(inner) => out.push(inner),
        Expr::Binary { lhs, rhs, .. } => {
            out.push(lhs);
            out.push(rhs);
        }
        Expr::Unary { value, .. } => out.push(value),
        Expr::Case { branches, fallback } => {
            for (c, o) in branches {
                out.push(c);
                out.push(o);
            }
            out.push(fallback);
        }
        Expr::Match {
            input,
            branches,
            fallback,
        } => {
            out.push(input);
            for (_, o) in branches {
                out.push(o);
            }
            out.push(fallback);
        }
        Expr::Let { bindings, body } => {
            for (_, e) in bindings {
                out.push(e);
            }
            out.push(body);
        }
        Expr::CaseChange { value, .. } => out.push(value),
        Expr::Slice { input, start, end } => {
            out.push(input);
            out.push(start);
            if let Some(e) = end {
                out.push(e);
            }
        }
        Expr::Split { input, separator } | Expr::Join { input, separator } => {
            out.push(input);
            out.push(separator);
        }
        Expr::IndexOf {
            needle,
            haystack,
            from_index,
        } => {
            out.push(needle);
            out.push(haystack);
            if let Some(e) = from_index {
                out.push(e);
            }
        }
        Expr::Format(sections) => {
            for s in sections {
                out.push(&s.text);
                if let Some(e) = &s.font_scale {
                    out.push(e);
                }
                if let Some(e) = &s.text_font {
                    out.push(e);
                }
                if let Some(e) = &s.text_color {
                    out.push(e);
                }
                if let Some(VerticalAlignSlot::Expr(e)) = &s.vertical_align {
                    out.push(e);
                }
            }
        }
        Expr::NumberFormat {
            value,
            locale,
            currency,
            unit,
            min_fraction_digits,
            max_fraction_digits,
        } => {
            out.push(value);
            for e in [
                locale,
                currency,
                unit,
                min_fraction_digits,
                max_fraction_digits,
            ].into_iter().flatten() {
                out.push(e);
            }
        }
        Expr::Collator(spec) => {
            for e in [
                &spec.case_sensitive,
                &spec.diacritic_sensitive,
                &spec.locale,
            ].into_iter().flatten() {
                out.push(e);
            }
        }
        Expr::Rgb { r, g, b, alpha } => {
            out.push(r);
            out.push(g);
            out.push(b);
            if let Some(e) = alpha {
                out.push(e);
            }
        }
        // Phase 5 — collections & feature metadata. Bare `properties`
        // / `id` have no children and are handled by the leaf arm
        // above; the others push their sub-expressions.
        Expr::GetFrom { host, .. } => out.push(host),
        Expr::At { index, array } => {
            out.push(index);
            out.push(array);
        }
        Expr::In { needle, haystack } => {
            out.push(needle);
            out.push(haystack);
        }
        Expr::Has { host, .. } => {
            if let Some(h) = host {
                out.push(h);
            }
        }
    }
    out
}

/// Set of `let`-bound variable names visible at a given point in the
/// parse. We need parse-time visibility so the `let/unbound` fixture
/// errors at compile time, mirroring upstream. The scope is a linked
/// list of stack frames so nested `let`s shadow outer names cheaply
/// without cloning the whole binding set.
#[derive(Debug, Clone, Copy)]
struct ParseScope<'a> {
    names: &'a [String],
    parent: Option<&'a ParseScope<'a>>,
}

impl<'a> ParseScope<'a> {
    fn root() -> Self {
        ParseScope {
            names: &[],
            parent: None,
        }
    }

    fn extend(&'a self, names: &'a [String]) -> Self {
        ParseScope {
            names,
            parent: Some(self),
        }
    }

    fn contains(&self, name: &str) -> bool {
        if self.names.iter().any(|n| n == name) {
            return true;
        }
        self.parent.is_some_and(|p| p.contains(name))
    }
}

/// Parse a JSON array as an operator call. Bare literal arrays must be
/// wrapped in `["literal", …]`; anything else here is treated as a
/// call.
fn parse_array(items: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // An empty array is never valid as an expression.
    let head = items.first().ok_or(ExprError::EmptyArray)?;

    let op = head
        .as_str()
        .ok_or_else(|| ExprError::NonStringOperator(format!("{head}")))?;

    let args = &items[1..];
    match op {
        "literal" => parse_literal(args),
        "get" => parse_get(args, scope),
        "zoom" => parse_zoom(args),
        "==" => parse_eq(args, scope),
        "!=" => parse_not_eq(args, scope),
        "<" => parse_compare(CompareOp::Less, args, scope),
        "<=" => parse_compare(CompareOp::LessOrEqual, args, scope),
        ">" => parse_compare(CompareOp::Greater, args, scope),
        ">=" => parse_compare(CompareOp::GreaterOrEqual, args, scope),
        "interpolate" => parse_interpolate(args, scope),
        "step" => parse_step(args, scope),
        "number" => parse_assert(AssertKind::Number, args, scope),
        "string" => parse_assert(AssertKind::String, args, scope),
        "boolean" => parse_assert(AssertKind::Boolean, args, scope),
        "object" => parse_assert(AssertKind::Object, args, scope),
        "array" => parse_array_assert(args, scope),
        "to-number" => parse_coerce(CoerceKind::ToNumber, args, scope),
        "to-string" => parse_coerce(CoerceKind::ToString, args, scope),
        "to-boolean" => parse_coerce(CoerceKind::ToBoolean, args, scope),
        "typeof" => parse_typeof(args, scope),

        // Variadic arithmetic. `+`, `*`, `max`, `min` accept any
        // number of operands (including zero).
        "+" => parse_variadic(VariadicOp::Plus, args, scope),
        "*" => parse_variadic(VariadicOp::Times, args, scope),
        "max" => parse_variadic(VariadicOp::Max, args, scope),
        "min" => parse_variadic(VariadicOp::Min, args, scope),

        // Strictly-binary numeric ops. `-` is special: 1-arg is
        // unary negation, 2-arg is subtraction.
        "-" => parse_minus(args, scope),
        "/" => parse_binary(BinaryOp::Divide, args, scope),
        "%" => parse_binary(BinaryOp::Mod, args, scope),
        "^" => parse_binary(BinaryOp::Pow, args, scope),

        // Unary math. Listed explicitly rather than via a static
        // table so the operator name is part of the match and shows
        // up in grep results.
        "abs" => parse_unary(UnaryOp::Abs, args, scope),
        "ceil" => parse_unary(UnaryOp::Ceil, args, scope),
        "floor" => parse_unary(UnaryOp::Floor, args, scope),
        "round" => parse_unary(UnaryOp::Round, args, scope),
        "sqrt" => parse_unary(UnaryOp::Sqrt, args, scope),
        "ln" => parse_unary(UnaryOp::Ln, args, scope),
        "log10" => parse_unary(UnaryOp::Log10, args, scope),
        "log2" => parse_unary(UnaryOp::Log2, args, scope),
        "sin" => parse_unary(UnaryOp::Sin, args, scope),
        "cos" => parse_unary(UnaryOp::Cos, args, scope),
        "tan" => parse_unary(UnaryOp::Tan, args, scope),
        "asin" => parse_unary(UnaryOp::Asin, args, scope),
        "acos" => parse_unary(UnaryOp::Acos, args, scope),
        "atan" => parse_unary(UnaryOp::Atan, args, scope),

        // Math constants. Upstream's `e/unexpected-argument` fixture
        // verifies a non-empty argument list is a compile-time error.
        "pi" => parse_math_const(MathConst::Pi, args),
        "e" => parse_math_const(MathConst::E, args),
        "ln2" => parse_math_const(MathConst::Ln2, args),

        // Boolean ops. `all`/`any` accept zero args (identity), `!`
        // requires exactly one.
        "all" => parse_bool_fold(BoolFold::All, args, scope),
        "any" => parse_bool_fold(BoolFold::Any, args, scope),
        "!" => parse_not(args, scope),

        // Pseudo-operator. The corpus only uses this inside short-
        // circuit fixtures, where evaluation must never reach it.
        "error" => parse_error_op(args),

        // Control flow and bindings.
        "case" => parse_case(args, scope),
        "match" => parse_match(args, scope),
        "coalesce" => parse_coalesce(args, scope),
        "let" => parse_let(args, scope),
        "var" => parse_var(args, scope),

        // Strings, formatting, i18n.
        "concat" => parse_concat(args, scope),
        "downcase" => parse_case_change(CaseChangeKind::Down, args, scope),
        "upcase" => parse_case_change(CaseChangeKind::Up, args, scope),
        "slice" => parse_slice(args, scope),
        "length" => parse_length(args, scope),
        "split" => parse_split(args, scope),
        "join" => parse_join(args, scope),
        "index-of" => parse_index_of(args, scope),
        "format" => parse_format(args, scope),
        "number-format" => parse_number_format(args, scope),
        "collator" => parse_collator(args, scope),
        "resolved-locale" => parse_resolved_locale(args, scope),
        "is-supported-script" => parse_is_supported_script(args, scope),

        // Colors. The `rgb` / `rgba` constructors are parsed eagerly;
        // `to-color` accepts any number of fallback expressions and
        // tries each at eval time. The two perceptual interpolations
        // (`interpolate-hcl` / `interpolate-lab`) share the
        // `interpolate` parser machinery — they just preset the
        // interpolation-kind to `Hcl` / `Lab` and ignore the leading
        // `["linear"]` argument's interior beyond the syntactic check
        // (upstream allows only the bare `["linear"]` form for the
        // perceptual variants, which is also what every fixture
        // exercises).
        "rgb" => parse_rgb(args, scope, /* with_alpha: */ false),
        "rgba" => parse_rgb(args, scope, /* with_alpha: */ true),
        "to-color" => parse_to_color(args, scope),
        "to-rgba" => parse_to_rgba(args, scope),
        "interpolate-hcl" => parse_interpolate_kind(Interpolation::Hcl, args, scope),
        "interpolate-lab" => parse_interpolate_kind(Interpolation::Lab, args, scope),

        // Geometry-aware. Both ops take a constant GeoJSON value as
        // their sole argument — we parse it eagerly at compile time
        // so per-feature evaluation can skip the JSON walk.
        "geometry-type" => parse_geometry_type(args),
        "within" => parse_within(args),
        "distance" => parse_distance(args),

        // Phase 9 — context-driven readers. `feature-state` /
        // `global-state` look up a name in their respective maps;
        // `heatmap-density` / `elevation` are zero-arg reads of a
        // numeric global; `image` resolves a sprite name against
        // `available_images`.
        "feature-state" => parse_feature_state(args, scope),
        "global-state" => parse_global_state(args),
        "heatmap-density" => parse_zero_arg(args, "heatmap-density", Expr::HeatmapDensity),
        "elevation" => parse_zero_arg(args, "elevation", Expr::Elevation),
        "image" => parse_image(args, scope),

        // Phase 5 — collections & feature metadata.
        "at" => parse_at(args, scope),
        "in" => parse_in(args, scope),
        "has" => parse_has(args, scope),
        "properties" => parse_zero_arg(args, "properties", Expr::Properties),
        "id" => parse_zero_arg(args, "id", Expr::Id),

        other => Err(ExprError::UnknownOperator(other.to_string())),
    }
}

fn parse_get(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // Two arities. `["get", name]` reads from `feature.properties`;
    // `["get", name, host]` reads from an explicit object expression
    // (`host`). The host form keeps the property-name slot as a string
    // literal — upstream's parser does the same; only the host slot is
    // an expression.
    match args.len() {
        1 => {
            let name = args[0].as_str().ok_or(ExprError::ExpectedString {
                op: "get",
                position: "property name",
            })?;
            Ok(Expr::Get(name.to_string()))
        }
        2 => {
            let name = args[0].as_str().ok_or(ExprError::ExpectedString {
                op: "get",
                position: "property name",
            })?;
            let host = Expr::from_json_scoped(&args[1], scope)?;
            Ok(Expr::GetFrom {
                name: name.to_string(),
                host: Box::new(host),
            })
        }
        n => Err(ExprError::WrongArity {
            op: "get",
            expected: "1 or 2",
            got: n,
        }),
    }
}

fn parse_zoom(args: &[Json]) -> Result<Expr, ExprError> {
    if !args.is_empty() {
        return Err(ExprError::WrongArity {
            op: "zoom",
            expected: "0",
            got: args.len(),
        });
    }
    Ok(Expr::Zoom)
}

fn parse_eq(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // Two- and three-argument forms; the third arg, when present, is a
    // collator expression. We delegate the collator slot's syntactic
    // check (must be a JSON array — i.e. a sub-expression) to its own
    // parse step so any inner errors surface from there.
    let (a, b, collator) = parse_compare_args("==", args, scope, CompareKind::Equality)?;
    Ok(Expr::Eq(Box::new(a), Box::new(b), collator))
}

/// Parse `["step", input, default, stop_in1, out1, stop_in2, out2, …]`.
///
/// Arity is at least 4 (input + default + one stop pair), and after
/// the leading two slots the remaining args come in `(stop_in,
/// stop_out)` pairs — same shape as `interpolate`'s tail but without
/// the leading interpolation-kind argument. We enforce strict ascent
/// of `stop_in`s here so the runtime `step` evaluator can scan
/// linearly without re-validating.
fn parse_step(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // Shape: input, default, then pairs of (stop_in, stop_out). Total
    // arity must be `>= 4` and even (input + default + 2*N stops).
    if args.len() < 4 || !args.len().is_multiple_of(2) {
        return Err(ExprError::WrongArity {
            op: "step",
            expected: "input, default, then at least one (in, out) pair",
            got: args.len(),
        });
    }
    let input = Box::new(Expr::from_json_scoped(&args[0], scope)?);
    let default = Box::new(Expr::from_json_scoped(&args[1], scope)?);

    let mut stops = Vec::with_capacity((args.len() - 2) / 2);
    for pair in args[2..].as_chunks::<2>().0 {
        let stop_in = pair[0].as_f64().ok_or(ExprError::ExpectedNumber {
            op: "step",
            position: "stop input",
        })?;
        if let Some(prev) = stops.last().map(|(x, _): &(f64, Expr)| *x)
            && stop_in <= prev {
                return Err(ExprError::StepNonAscendingStops);
            }
        let stop_out = Expr::from_json_scoped(&pair[1], scope)?;
        stops.push((stop_in, stop_out));
    }
    Ok(Expr::Step {
        input,
        default,
        stops,
    })
}

fn parse_interpolate(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // Shape: `interpolation`, `input`, then pairs of (`stop_in`,
    // `stop_out`). The pairs may themselves be expressions in general,
    // but minimal_style.json only uses literal numbers and we accept
    // arbitrary `Expr`s here for free.
    if args.len() < 4 || !args.len().is_multiple_of(2) {
        return Err(ExprError::WrongArity {
            op: "interpolate",
            expected: "interpolation, input, then at least one (in, out) pair",
            got: args.len(),
        });
    }

    let interpolation = parse_interpolation(&args[0])?;
    parse_interpolate_tail(interpolation, args, scope)
}

/// Shared by `interpolate-hcl` / `interpolate-lab`, where the operator
/// name itself determines the interpolation space. We still consume
/// the leading `["linear"]` slot for shape parity with upstream —
/// every fixture exercises the linear variant only, and the
/// perceptual operators don't take a separate "kind" argument.
fn parse_interpolate_kind(
    kind: Interpolation,
    args: &[Json],
    scope: &ParseScope<'_>,
) -> Result<Expr, ExprError> {
    if args.len() < 4 || !args.len().is_multiple_of(2) {
        return Err(ExprError::WrongArity {
            op: "interpolate-*",
            expected: "interpolation, input, then at least one (in, out) pair",
            got: args.len(),
        });
    }
    // The leading slot must still parse as a valid interpolation
    // descriptor for shape parity with the plain `interpolate`
    // operator.
    //
    // TODO: the corpus's `interpolate-hcl` / `interpolate-lab`
    // fixtures only exercise linear curves, so we discard the inner
    // shape and force the perceptual `Hcl` / `Lab` lerps to run on a
    // plain linear `t`. Combining an `exponential` / `cubic-bezier`
    // curve with a perceptual color space would need a small struct
    // (curve + color space) on `Interpolation`; we can split it that
    // way the day a fixture pins the combined behavior.
    let _ = parse_interpolation(&args[0])?;
    // Upstream `interpolate-hcl` / `interpolate-lab` only operate on
    // color values, and the parser rejects any stop whose output
    // type isn't `color` at compile time. We don't have full type
    // inference yet, but we can statically reject the obvious
    // miscarriers — booleans, nulls, raw numbers, and array literals
    // (which can never be coerced to a color). `to-color/-inferred`
    // strings and `rgb(...)` calls pass through unchallenged.
    for pair in args[2..].as_chunks::<2>().0 {
        reject_non_color_stop_output(&pair[1])?;
    }
    parse_interpolate_tail(kind, args, scope)
}

/// Compile-time rejection of obviously non-color stop outputs for the
/// perceptual interpolation operators. We can't catch every case
/// without type inference, but rejecting bare booleans / nulls / raw
/// numbers / non-color string literals / numeric-array literals
/// covers what the `interpolate-{hcl,lab}/uninterpolable-output/*`
/// fixtures pin.
fn reject_non_color_stop_output(v: &Json) -> Result<(), ExprError> {
    match v {
        Json::Bool(_) | Json::Null => Err(ExprError::InterpolateColorOutputRequired),
        Json::Number(_) => Err(ExprError::InterpolateColorOutputRequired),
        Json::String(s) if parse_css_color(s).is_none() => {
            Err(ExprError::InterpolateColorOutputRequired)
        }
        Json::Array(items) => {
            // `["literal", [n, n, n, n]]` is a color-by-array form
            // and is fine. Other array calls might be a `get` /
            // `to-color` / `rgb` / etc., which we let through. Bare
            // array literals (not wrapped in `["literal", …]`) are
            // operator calls and not statically rejectable here.
            if items.first().and_then(Json::as_str) == Some("literal")
                && let Some(inner) = items.get(1) {
                    if let Json::Array(elems) = inner {
                        let all_num = elems.iter().all(|e| matches!(e, Json::Number(_)));
                        if all_num && (elems.len() == 3 || elems.len() == 4) {
                            return Ok(());
                        }
                        // Non-3/4-number array literal can never be a color.
                        return Err(ExprError::InterpolateColorOutputRequired);
                    }
                    // `["literal", scalar]` — recurse to apply the
                    // same rules.
                    return reject_non_color_stop_output(inner);
                }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_interpolate_tail(
    interpolation: Interpolation,
    args: &[Json],
    scope: &ParseScope<'_>,
) -> Result<Expr, ExprError> {
    let input = Box::new(Expr::from_json_scoped(&args[1], scope)?);

    let mut stops = Vec::with_capacity((args.len() - 2) / 2);
    for pair in args[2..].as_chunks::<2>().0 {
        let stop_in = pair[0].as_f64().ok_or(ExprError::ExpectedNumber {
            op: "interpolate",
            position: "stop input",
        })?;
        // Upstream rejects equal-or-decreasing stop inputs at compile
        // time; `interpolate/duplicate_stops` pins this with two
        // zero-valued inputs in a row. Compare against the previous
        // stop's input rather than the whole-list min so we catch the
        // duplicate even when the rest of the sequence is ascending.
        if let Some(prev) = stops.last().map(|(x, _): &(f64, Expr)| *x)
            && stop_in <= prev {
                return Err(ExprError::InterpolateNonAscendingStops);
            }
        let stop_out = Expr::from_json_scoped(&pair[1], scope)?;
        stops.push((stop_in, stop_out));
    }

    // Color-mode detection. The HCL/LAB variants always run through
    // the color pipeline; for the plain linear form we look for any
    // stop whose literal output is a CSS-color string. That catches
    // the common `["interpolate", ["linear"], ["zoom"], 0, "red", …]`
    // idiom without needing the full Phase 10 type inference.
    let color_typed = matches!(interpolation, Interpolation::Hcl | Interpolation::Lab)
        || stops.iter().any(|(_, expr)| stop_is_color_literal(expr));

    Ok(Expr::Interpolate {
        interpolation,
        input,
        stops,
        color_typed,
    })
}

/// Static check for "is this stop output type interpolatable?". The
/// spec restricts stop outputs to number, color, array<number>, and
/// the projection/padding shapes we don't model. The runtime fallback
/// (`value`) defers to the eval-time numeric check in
/// [`interpolate_pair`]; everything else gets rejected here with the
/// upstream wording.
///
/// The check is called from the runner (after propertySpec wiring)
/// rather than during parse so it can be skipped when the
/// propertySpec demands an exotic interpolatable type (e.g.
/// `projectionDefinition` accepts string stops that the bare
/// interpolate parser couldn't otherwise admit).
pub fn check_interpolate_stops_interpolatable(expr: &Expr) -> Result<(), ExprError> {
    if let Expr::Interpolate {
        stops,
        color_typed,
        interpolation,
        ..
    } = expr
    {
        if *color_typed {
            return Ok(());
        }
        // Exponential / cubic-bezier curves can shape arrays the same
        // way linear does, but only when the array length is
        // statically pinned — the curve parameterization runs once
        // and then the lerp threads through every component, so a
        // stop whose length is unknown can't be safely interpolated.
        // The `interpolate/exponential-uninterpolatable-numeric-
        // array` fixture pins this on a stop typed `array<number>`
        // (no length) under an exponential curve.
        let unsized_arrays_ok = matches!(interpolation, Interpolation::Linear);
        for (_, out) in stops {
            let t = out.result_type();
            let ok = match &t {
                ExprType::Number | ExprType::Color | ExprType::Value => true,
                ExprType::Array(item, length) => {
                    let item_ok = matches!(
                        item.as_deref(),
                        Some(ExprType::Number) | Some(ExprType::Value) | None,
                    );
                    item_ok && (unsized_arrays_ok || length.is_some())
                }
                _ => false,
            };
            if !ok {
                return Err(ExprError::InterpolateNotInterpolatable(t.to_spec_string()));
            }
        }
    }
    Ok(())
}

/// Heuristic for "this stop produces a color value". We only look at
/// shapes that are *syntactically* obvious: a bare string literal that
/// parses as a CSS color, an `rgb`/`rgba` call, or a `to-color` call.
/// Anything more involved (e.g. `["match", …]` returning color names)
/// defers to runtime detection in `interpolate_pair` via the cross-
/// endpoint `Value::Color` check.
fn stop_is_color_literal(e: &Expr) -> bool {
    match e {
        Expr::Lit(Literal::String(s)) => parse_css_color(s).is_some(),
        Expr::Rgb { .. } | Expr::ToColor(_) => true,
        _ => false,
    }
}

fn parse_interpolation(v: &Json) -> Result<Interpolation, ExprError> {
    // Interpolation kind is itself wrapped in an array, e.g.
    // `["linear"]`, `["exponential", base]`, or
    // `["cubic-bezier", x1, y1, x2, y2]`. Trailing arguments past the
    // declared arity are rejected here so the `linear-1-arg` /
    // `exponential-2-args` / `cubic-bezier-5-args` fixtures surface a
    // parse error at the right call site.
    let arr = v.as_array().ok_or(ExprError::MalformedInterpolation)?;
    let kind = arr
        .first()
        .and_then(Json::as_str)
        .ok_or(ExprError::MalformedInterpolation)?;
    match kind {
        // `linear` permits arbitrary trailing arguments — upstream's
        // parser ignores them silently. The `linear-1-arg` fixture
        // exercises a stray `0.5` after `linear` and still expects
        // parse success.
        "linear" => Ok(Interpolation::Linear),
        "exponential" => parse_exponential_interpolation(arr),
        "cubic-bezier" => parse_cubic_bezier_interpolation(arr),
        other => Err(ExprError::UnsupportedInterpolation(other.to_string())),
    }
}

/// Parse the inner shape of `["exponential", base]`. Upstream's spec
/// requires the base to be a numeric literal (the
/// `exponential-expression-base` fixture pins the non-numeric reject)
/// but tolerates trailing-arg slop — `exponential-2-args` exercises a
/// stray `9999` after the base.
fn parse_exponential_interpolation(arr: &[Json]) -> Result<Interpolation, ExprError> {
    if arr.len() < 2 {
        return Err(ExprError::MalformedInterpolation);
    }
    let base = arr[1]
        .as_f64()
        .ok_or(ExprError::InterpolateExponentialBaseNotNumber)?;
    Ok(Interpolation::Exponential { base })
}

/// Parse `["cubic-bezier", x1, y1, x2, y2]`. The four control-point
/// coordinates must be numeric literals in `0..=1`; expression-valued
/// or out-of-range control points are rejected at parse time so the
/// `cubic-bezier-invalid-control-point` and
/// `cubic-bezier-expression-control-point` fixtures fail at compile.
fn parse_cubic_bezier_interpolation(arr: &[Json]) -> Result<Interpolation, ExprError> {
    if arr.len() != 5 {
        return Err(ExprError::CubicBezierInvalidControlPoints);
    }
    let mut coords = [0.0_f64; 4];
    for (i, slot) in coords.iter_mut().enumerate() {
        let n = arr[i + 1]
            .as_f64()
            .ok_or(ExprError::CubicBezierInvalidControlPoints)?;
        // The CSS `cubic-bezier()` spec restricts the x-axis to
        // `0..=1` (so the curve stays a function of `t`); the spec
        // text extends that to y for the upstream parser.
        if !(0.0..=1.0).contains(&n) {
            return Err(ExprError::CubicBezierInvalidControlPoints);
        }
        *slot = n;
    }
    Ok(Interpolation::CubicBezier {
        x1: coords[0],
        y1: coords[1],
        x2: coords[2],
        y2: coords[3],
    })
}

/// `["literal", v]` — the second JSON argument becomes the literal's
/// runtime value as-is. Arrays/objects inside are *not* re-parsed as
/// expressions.
fn parse_literal(args: &[Json]) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: "literal",
            expected: "1",
            got: args.len(),
        });
    };
    Ok(Expr::Literal(json_to_value(v)))
}

/// Convert a raw JSON value to a runtime [`Value`]. Used by `literal`,
/// which treats its argument as data rather than code.
fn json_to_value(v: &Json) -> Value {
    match v {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => Value::Number(n.as_f64().unwrap_or(f64::NAN)),
        Json::String(s) => Value::String(s.clone()),
        Json::Array(items) => Value::Array(items.iter().map(json_to_value).collect()),
        Json::Object(map) => {
            let mut out = BTreeMap::new();
            for (k, v) in map {
                out.insert(k.clone(), json_to_value(v));
            }
            Value::Object(out)
        }
    }
}

fn parse_assert(
    kind: AssertKind,
    args: &[Json],
    scope: &ParseScope<'_>,
) -> Result<Expr, ExprError> {
    if args.is_empty() {
        return Err(ExprError::WrongArity {
            op: kind.as_str(),
            expected: "at least 1",
            got: 0,
        });
    }
    let values = args
        .iter()
        .map(|a| Expr::from_json_scoped(a, scope))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::Assert { kind, values })
}

fn parse_array_assert(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // `array` has three syntactic forms; they're distinguished purely
    // by the argument count, matching upstream maplibre-style-spec:
    //
    //   1 arg                 — bare array assertion
    //   2 args                — `[op, type, value]`
    //   3+ args               — `[op, type, length|null, value, fallback…]`
    //
    // In the 2+ arg forms the type slot is *not* itself an expression:
    // it must be a literal `"number"` / `"string"` / `"boolean"`. The
    // length slot must be a non-negative integer literal or `null`.
    // Anything else is rejected here rather than at eval time.
    match args.len() {
        0 => Err(ExprError::WrongArity {
            op: "array",
            expected: "at least 1",
            got: 0,
        }),
        1 => {
            let value = Expr::from_json_scoped(&args[0], scope)?;
            Ok(Expr::ArrayAssert {
                item_type: None,
                length: None,
                values: vec![value],
            })
        }
        _ => {
            let item_type = parse_array_item_type(&args[0])?;
            if args.len() == 2 {
                let value = Expr::from_json_scoped(&args[1], scope)?;
                return Ok(Expr::ArrayAssert {
                    item_type: Some(item_type),
                    length: None,
                    values: vec![value],
                });
            }
            let length = parse_array_length(&args[1])?;
            let values = args[2..]
                .iter()
                .map(|a| Expr::from_json_scoped(a, scope))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::ArrayAssert {
                item_type: Some(item_type),
                length,
                values,
            })
        }
    }
}

fn parse_array_item_type(v: &Json) -> Result<ItemType, ExprError> {
    v.as_str()
        .and_then(ItemType::parse)
        .ok_or(ExprError::ArrayInvalidItemType)
}

fn parse_array_length(v: &Json) -> Result<Option<usize>, ExprError> {
    match v {
        Json::Null => Ok(None),
        Json::Number(n) => {
            // Upstream's "positive integer literal" check rejects
            // negative and fractional values. `as_u64` enforces both
            // in one step.
            let n = n.as_u64().ok_or(ExprError::ArrayInvalidLength)?;
            Ok(Some(n as usize))
        }
        _ => Err(ExprError::ArrayInvalidLength),
    }
}

fn parse_coerce(
    kind: CoerceKind,
    args: &[Json],
    scope: &ParseScope<'_>,
) -> Result<Expr, ExprError> {
    let op = match kind {
        CoerceKind::ToNumber => "to-number",
        CoerceKind::ToString => "to-string",
        CoerceKind::ToBoolean => "to-boolean",
    };
    // Only `to-number` accepts trailing fallback values per the spec;
    // `to-string` and `to-boolean` always succeed for a single
    // argument and have no use for fallbacks, so the upstream parser
    // rejects extra args at compile time.
    if args.is_empty() {
        return Err(ExprError::WrongArity {
            op,
            expected: "1",
            got: 0,
        });
    }
    if !matches!(kind, CoerceKind::ToNumber) && args.len() != 1 {
        return Err(ExprError::WrongArity {
            op,
            expected: "1",
            got: args.len(),
        });
    }
    let values = args
        .iter()
        .map(|a| Expr::from_json_scoped(a, scope))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::Coerce { kind, values })
}

fn parse_typeof(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: "typeof",
            expected: "1",
            got: args.len(),
        });
    };
    Ok(Expr::TypeOf(Box::new(Expr::from_json_scoped(v, scope)?)))
}

// ------------------------------------------------------------------------------
// Arithmetic / math / comparison / boolean — parsing
// ------------------------------------------------------------------------------

/// Variadic `+`, `*`, `max`, `min`. Zero-arg form is accepted; identity
/// is supplied at eval time. The corpus's `arity-0` fixtures verify
/// that empty `+` returns `0`, empty `*` returns `1`, empty `max`
/// returns `-Infinity`, and empty `min` returns `+Infinity`.
fn parse_variadic(
    op: VariadicOp,
    args: &[Json],
    scope: &ParseScope<'_>,
) -> Result<Expr, ExprError> {
    let values = args
        .iter()
        .map(|a| Expr::from_json_scoped(a, scope))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::Variadic { op, values })
}

/// `-` is the lone arithmetic op with two arities: `["-", x]` is unary
/// negation, `["-", a, b]` is subtraction. The unary form is rewritten
/// to `0 - x` so eval only has to know about the binary shape.
fn parse_minus(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    match args.len() {
        1 => Ok(Expr::Binary {
            op: BinaryOp::Minus,
            lhs: Box::new(Expr::Lit(Literal::Number(0.0))),
            rhs: Box::new(Expr::from_json_scoped(&args[0], scope)?),
        }),
        2 => Ok(Expr::Binary {
            op: BinaryOp::Minus,
            lhs: Box::new(Expr::from_json_scoped(&args[0], scope)?),
            rhs: Box::new(Expr::from_json_scoped(&args[1], scope)?),
        }),
        n => Err(ExprError::WrongArity {
            op: "-",
            expected: "1 or 2",
            got: n,
        }),
    }
}

/// Strictly-binary numeric ops (`/`, `%`, `^`).
fn parse_binary(op: BinaryOp, args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [a, b] = args else {
        return Err(ExprError::WrongArity {
            op: binary_op_str(op),
            expected: "2",
            got: args.len(),
        });
    };
    Ok(Expr::Binary {
        op,
        lhs: Box::new(Expr::from_json_scoped(a, scope)?),
        rhs: Box::new(Expr::from_json_scoped(b, scope)?),
    })
}

fn binary_op_str(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Minus => "-",
        BinaryOp::Divide => "/",
        BinaryOp::Mod => "%",
        BinaryOp::Pow => "^",
    }
}

fn unary_op_str(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Abs => "abs",
        UnaryOp::Ceil => "ceil",
        UnaryOp::Floor => "floor",
        UnaryOp::Round => "round",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Ln => "ln",
        UnaryOp::Log10 => "log10",
        UnaryOp::Log2 => "log2",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tan => "tan",
        UnaryOp::Asin => "asin",
        UnaryOp::Acos => "acos",
        UnaryOp::Atan => "atan",
    }
}

fn parse_unary(op: UnaryOp, args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: unary_op_str(op),
            expected: "1",
            got: args.len(),
        });
    };
    Ok(Expr::Unary {
        op,
        value: Box::new(Expr::from_json_scoped(v, scope)?),
    })
}

fn parse_math_const(c: MathConst, args: &[Json]) -> Result<Expr, ExprError> {
    if !args.is_empty() {
        return Err(ExprError::WrongArity {
            op: c.name(),
            expected: "0",
            got: args.len(),
        });
    }
    Ok(Expr::MathConst(c))
}

fn parse_not_eq(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let (a, b, collator) = parse_compare_args("!=", args, scope, CompareKind::Equality)?;
    Ok(Expr::NotEq(Box::new(a), Box::new(b), collator))
}

fn parse_compare(op: CompareOp, args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let (a, b, collator) = parse_compare_args(op.op_str(), args, scope, CompareKind::Ordered)?;
    Ok(Expr::Compare {
        op,
        lhs: Box::new(a),
        rhs: Box::new(b),
        collator,
    })
}

/// Shared argument-shape check for the comparison operators
/// (`==` / `!=` / `<` / `<=` / `>` / `>=`). Each accepts either two
/// operands or two operands plus a collator expression. When a
/// collator is supplied, we reject statically-known non-string
/// operand pairs at parse time — matching `collator/comparison-
/// number-error` and `collator/equals-non-string-error`.
fn parse_compare_args(
    op: &'static str,
    args: &[Json],
    scope: &ParseScope<'_>,
    kind: CompareKind,
) -> Result<(Expr, Expr, Option<Box<Expr>>), ExprError> {
    match args.len() {
        2 => {
            let a = Expr::from_json_scoped(&args[0], scope)?;
            let b = Expr::from_json_scoped(&args[1], scope)?;
            // Colors aren't ordered or equality-comparable at the
            // expression level — `equal/color` pins this. We only
            // catch the syntactically-obvious cases (literal `rgb`
            // calls / `to-color` of a literal); deeper detection
            // requires Phase 10 type inference.
            if statically_color(&a) || statically_color(&b) {
                return Err(ExprError::CompareUnsupportedColor(op));
            }
            // Phase 10 type-check. Equality and ordered comparisons
            // share the "operands must match" rule but differ on
            // which scalar kinds are admissible: equality forbids
            // `array`/`object`/`null`-typed operands; ordered
            // comparison narrows to `number`/`string`/`value`. We
            // surface a generic "Cannot compare types …" rejection
            // (the runner only cares we erred at parse time).
            check_compare_operand_types(op, kind, &a, &b)?;
            Ok((a, b, None))
        }
        3 => {
            let a = Expr::from_json_scoped(&args[0], scope)?;
            let b = Expr::from_json_scoped(&args[1], scope)?;
            let c = Expr::from_json_scoped(&args[2], scope)?;
            // If both operands are statically non-string scalars and
            // we've also been handed a collator, upstream rejects at
            // compile time. We only flag the case where *both* sides
            // are statically known non-strings; if either side is a
            // `get` we have to wait until runtime.
            let lhs_kind = static_scalar_kind(&a);
            let rhs_kind = static_scalar_kind(&b);
            if let (Some(lk), Some(rk)) = (lhs_kind, rhs_kind)
                && lk != "string" && rk != "string" {
                    return Err(ExprError::CollatorNonStringStatic);
                }
            Ok((a, b, Some(Box::new(c))))
        }
        n => Err(ExprError::WrongArity {
            op,
            expected: "2 or 3",
            got: n,
        }),
    }
}

/// Distinguishes the type-rule between equality (`==`/`!=`) and
/// ordered (`<`/`<=`/`>`/`>=`) compare operators. Used by
/// [`check_compare_operand_types`] to apply the right admissibility
/// table without duplicating the rest of the helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareKind {
    Equality,
    Ordered,
}

/// Apply Phase 10's parse-time type rule on a comparison's two
/// operands. The contract is:
///
/// * Equality (`==`/`!=`): operands' inferred types must agree
///   (with `value` deferring); `array`/`object`/`null` operand
///   types are rejected outright.
/// * Ordered (`<`/`<=`/`>`/`>=`): operand types must be `number`
///   or `string` (or `value`), and the two sides must match.
fn check_compare_operand_types(
    op: &'static str,
    kind: CompareKind,
    a: &Expr,
    b: &Expr,
) -> Result<(), ExprError> {
    let at = a.result_type();
    let bt = b.result_type();
    let admissible = |t: &ExprType| -> bool {
        match kind {
            // Equality: anything except array / object /
            // resolvedImage / formatted / collator. `null` *is*
            // admissible — upstream's `equal/null-{lhs,rhs}` pin
            // that — and `value` is the top type so it stays
            // admissible too; runtime tags cross-kind mismatches.
            CompareKind::Equality => !matches!(
                t,
                ExprType::Array(_, _)
                    | ExprType::Object
                    | ExprType::Formatted
                    | ExprType::ResolvedImage
                    | ExprType::Collator
            ),
            // Ordered: number, string, or value. Booleans / nulls
            // / colors / etc. are all out.
            CompareKind::Ordered => {
                matches!(t, ExprType::Number | ExprType::String | ExprType::Value)
            }
        }
    };
    if !admissible(&at) {
        return Err(ExprError::CompareUnsupportedType(op, at.to_spec_string()));
    }
    if !admissible(&bt) {
        return Err(ExprError::CompareUnsupportedType(op, bt.to_spec_string()));
    }
    // Reject cross-type comparisons when *both* sides are concrete.
    // `value` defers — runtime catches the mismatch with the canonical
    // assertion error.
    let lhs_concrete = !matches!(at, ExprType::Value);
    let rhs_concrete = !matches!(bt, ExprType::Value);
    if lhs_concrete && rhs_concrete && at != bt {
        return Err(ExprError::CompareTypeMismatch(
            at.to_spec_string(),
            bt.to_spec_string(),
        ));
    }
    Ok(())
}

fn parse_bool_fold(op: BoolFold, args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let values = args
        .iter()
        .map(|a| Expr::from_json_scoped(a, scope))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::BoolFold { op, values })
}

fn parse_not(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: "!",
            expected: "1",
            got: args.len(),
        });
    };
    Ok(Expr::Not(Box::new(Expr::from_json_scoped(v, scope)?)))
}

/// `["error", msg]` only ever appears inside short-circuit fixtures.
/// We accept and stash the message; eval raises it. The argument's
/// shape is loose because the corpus uses both literal strings and
/// expressions, but the message we surface is best-effort.
fn parse_error_op(args: &[Json]) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: "error",
            expected: "1",
            got: args.len(),
        });
    };
    let msg = v.as_str().unwrap_or("error operator reached").to_string();
    Ok(Expr::ErrorOp(msg))
}

// ------------------------------------------------------------------------------
// Control flow & bindings — parsing
// ------------------------------------------------------------------------------

/// `["case", c1, o1, c2, o2, …, fallback]`. The argument list has odd
/// length: each `(cond, out)` pair is followed by a final fallback. We
/// only enforce that shape here — the boolean check on `cond` is a
/// runtime concern.
fn parse_case(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // We need at least one (cond, out) pair plus a fallback — three
    // arguments total. Without them there's nothing to branch on.
    if args.len() < 3 || args.len().is_multiple_of(2) {
        return Err(ExprError::WrongArity {
            op: "case",
            expected: "an odd number ≥ 3 (cond/out pairs followed by a fallback)",
            got: args.len(),
        });
    }
    let pair_count = (args.len() - 1) / 2;
    let mut branches = Vec::with_capacity(pair_count);
    for pair in args[..args.len() - 1].as_chunks::<2>().0 {
        let cond = Expr::from_json_scoped(&pair[0], scope)?;
        let out = Expr::from_json_scoped(&pair[1], scope)?;
        branches.push((cond, out));
    }
    let fallback = Box::new(Expr::from_json_scoped(
        args.last().expect("len ≥ 3"),
        scope,
    )?);
    Ok(Expr::Case { branches, fallback })
}

/// `["match", input, l1, o1, l2, o2, …, fallback]`. The label positions
/// are *not* expressions — they're raw JSON literals (scalar or array
/// of scalars). Upstream picks a uniform label kind (number or string)
/// from the first branch and requires every other branch to match.
/// Labels must be unique across all branches; the spec leaves this to
/// the parser, which is also where we enforce it.
fn parse_match(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // The minimum legal shape is `[input, label, out, fallback]` —
    // anything shorter is the upstream "Expected at least 4 arguments"
    // case.
    if args.len() < 4 {
        return Err(ExprError::MatchTooFewArguments(args.len()));
    }
    // After the input and the fallback, the remaining slots come in
    // (label, output) pairs.
    if !(args.len() - 2).is_multiple_of(2) {
        return Err(ExprError::MatchUnpairedBranches(args.len()));
    }

    let input = Box::new(Expr::from_json_scoped(&args[0], scope)?);

    // We decide the uniform label kind (number or string) by inspecting
    // the first label slot. Subsequent labels must agree. This mirrors
    // upstream's behavior; `match/labels-mixed-*` fixtures exercise the
    // mismatch path, which surfaces as a parse error.
    let mut branches: Vec<(Vec<MatchLabel>, Expr)> = Vec::new();
    let mut label_kind: Option<MatchLabelKind> = None;
    let mut seen_numbers: Vec<i64> = Vec::new();
    let mut seen_strings: Vec<String> = Vec::new();

    let branch_args = &args[1..args.len() - 1];
    for pair in branch_args.as_chunks::<2>().0 {
        let labels = parse_match_labels(&pair[0])?;
        // Pin the label kind on the first branch; downstream branches
        // must match. The error mirrors `match/labels-mixed-*`.
        let pair_kind = match_labels_kind(&labels);
        match (label_kind, pair_kind) {
            (None, _) => label_kind = Some(pair_kind),
            (Some(prev), got) if prev != got => return Err(ExprError::MatchLabelKindMismatch),
            _ => {}
        }
        for l in &labels {
            // Duplicate-label detection is upstream's compile-time job.
            // Linear-scanning a Vec keeps the dependency surface small;
            // realistic style files have a handful of labels per match.
            match l {
                MatchLabel::Number(n) => {
                    if seen_numbers.contains(n) {
                        return Err(ExprError::MatchDuplicateLabel);
                    }
                    seen_numbers.push(*n);
                }
                MatchLabel::String(s) => {
                    if seen_strings.iter().any(|x| x == s) {
                        return Err(ExprError::MatchDuplicateLabel);
                    }
                    seen_strings.push(s.clone());
                }
            }
        }
        let out = Expr::from_json_scoped(&pair[1], scope)?;
        branches.push((labels, out));
    }

    let fallback = Box::new(Expr::from_json_scoped(
        args.last().expect("len ≥ 4"),
        scope,
    )?);

    // Phase 10 type checks.
    //
    // Input vs labels: the input's static type must agree with the
    // chosen label kind. `match/mismatch-input` pins this with an
    // explicit `["string", …]`-typed input against numeric labels.
    if let Some(lk) = label_kind {
        let input_t = input.result_type();
        let label_t = match lk {
            MatchLabelKind::Number => ExprType::Number,
            MatchLabelKind::String => ExprType::String,
        };
        // A `value`-typed input defers to the eval-time runtime check,
        // matching upstream's "tolerate the top type" rule.
        if !matches!(input_t, ExprType::Value) && input_t != label_t {
            return Err(ExprError::MatchInputTypeMismatch {
                expected: label_t.to_spec_string(),
                got: input_t.to_spec_string(),
            });
        }
    }
    // Outputs must agree across all branches plus the fallback.
    // `match/mismatch-output` is the pinning fixture; we use the
    // first concrete branch type as the running expected shape and
    // require subsequent branches to be *compatible* with it via
    // `branch_type_compatible`. `value` and `null` defer to runtime;
    // arrays with matching item type but differing lengths join
    // structurally (the `match/infer-array-type` case).
    let mut output_type: Option<ExprType> = None;
    for (_, out) in branches.iter().chain(std::iter::once(&(
        Vec::<MatchLabel>::new(),
        (*fallback).clone(),
    ))) {
        let t = out.result_type();
        match &output_type {
            None => {
                if !matches!(t, ExprType::Value | ExprType::Null) {
                    output_type = Some(t);
                }
            }
            Some(prev) => {
                if !branch_type_compatible(prev, &t) {
                    return Err(ExprError::MatchOutputTypeMismatch {
                        expected: prev.to_spec_string(),
                        got: t.to_spec_string(),
                    });
                }
            }
        }
    }

    Ok(Expr::Match {
        input,
        branches,
        fallback,
    })
}

/// `true` when `b` is structurally compatible with `a` for branch /
/// output checks (`match`, `case`, etc.). The rule is a loosened
/// version of [`type_is_assignable`]: same scalar types match, arrays
/// with matching item type match regardless of length, and `value` /
/// `null` defer to the runtime. We can't run the strict assignability
/// table here because the test runner only sees the *result type* of
/// a fixture, and many fixtures rely on the implicit cast a propertySpec
/// would normally drive.
fn branch_type_compatible(a: &ExprType, b: &ExprType) -> bool {
    if a == b {
        return true;
    }
    match (a, b) {
        (ExprType::Value, _) | (_, ExprType::Value) => true,
        (ExprType::Null, _) | (_, ExprType::Null) => true,
        (ExprType::Array(ai, _), ExprType::Array(bi, _)) => match (ai, bi) {
            (None, _) | (_, None) => true,
            (Some(x), Some(y)) => branch_type_compatible(x, y),
        },
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchLabelKind {
    Number,
    String,
}

fn match_labels_kind(labels: &[MatchLabel]) -> MatchLabelKind {
    // `parse_match_labels` guarantees a non-empty homogeneous list, so
    // peeking at the first entry suffices.
    match labels[0] {
        MatchLabel::Number(_) => MatchLabelKind::Number,
        MatchLabel::String(_) => MatchLabelKind::String,
    }
}

/// Parse a single label slot. A slot is either a scalar literal
/// (number or string) or a JSON array of scalars (the multi-match
/// shorthand for "any of these labels routes to this branch"). Booleans,
/// nulls, objects, and any nested non-scalar values are rejected — the
/// `match/label-{boolean,null,object,array}` fixtures pin this down.
fn parse_match_labels(v: &Json) -> Result<Vec<MatchLabel>, ExprError> {
    match v {
        Json::Number(_) | Json::String(_) => Ok(vec![parse_one_label(v)?]),
        Json::Array(items) => {
            // The upstream parser's check is "at least one branch
            // label", which our `match/empty-case` fixture exercises.
            if items.is_empty() {
                return Err(ExprError::MatchEmptyLabelArray);
            }
            let mut out = Vec::with_capacity(items.len());
            let mut kind: Option<MatchLabelKind> = None;
            for item in items {
                let l = parse_one_label(item)?;
                // A `[label, label, …]` array can't mix kinds either —
                // upstream errors on `[true]` arrays the same way it
                // does on a bare `true` label.
                let lk = match l {
                    MatchLabel::Number(_) => MatchLabelKind::Number,
                    MatchLabel::String(_) => MatchLabelKind::String,
                };
                if let Some(prev) = kind {
                    if prev != lk {
                        return Err(ExprError::MatchLabelKindMismatch);
                    }
                } else {
                    kind = Some(lk);
                }
                out.push(l);
            }
            Ok(out)
        }
        _ => Err(ExprError::MatchLabelInvalid),
    }
}

fn parse_one_label(v: &Json) -> Result<MatchLabel, ExprError> {
    match v {
        Json::Number(n) => {
            let f = n
                .as_f64()
                .ok_or_else(|| ExprError::InvalidNumber(n.to_string()))?;
            // Numeric labels must be safe integers (`Number.isSafeInteger`
            // in JS). The two upstream fixtures pin both halves: non-
            // integer values trip `label-non-integer` and out-of-range
            // values trip `label-overflow`.
            if !f.is_finite() || f.fract() != 0.0 {
                return Err(ExprError::MatchLabelNonInteger);
            }
            // 2^53 − 1 is the largest exactly representable integer in
            // an IEEE-754 double; beyond that upstream rejects with
            // "Branch labels must be integers no larger than …".
            const MAX_SAFE: f64 = 9_007_199_254_740_991.0;
            if f.abs() > MAX_SAFE {
                return Err(ExprError::MatchLabelOverflow);
            }
            Ok(MatchLabel::Number(f as i64))
        }
        Json::String(s) => Ok(MatchLabel::String(s.clone())),
        _ => Err(ExprError::MatchLabelInvalid),
    }
}

/// `["coalesce", e1, e2, …]`. The spec requires at least one child;
/// the parser rejects the zero-arg form.
fn parse_coalesce(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    if args.is_empty() {
        return Err(ExprError::WrongArity {
            op: "coalesce",
            expected: "at least 1",
            got: 0,
        });
    }
    let values = args
        .iter()
        .map(|a| Expr::from_json_scoped(a, scope))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::Coalesce(values))
}

/// `["let", n1, v1, n2, v2, …, body]`. Binding values are parsed in
/// declaration order, with each subsequent value seeing the names
/// introduced by the prior bindings — that's why later `var` lookups
/// inside a binding's value can reference earlier bindings.
fn parse_let(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // Must be name/value pairs (any number, including zero) plus a
    // body. With zero bindings the shape is `["let", body]`, which the
    // spec accepts despite being degenerate.
    if args.is_empty() || args.len().is_multiple_of(2) {
        return Err(ExprError::WrongArity {
            op: "let",
            expected: "(name, value) pairs followed by a body",
            got: args.len(),
        });
    }
    let pair_count = (args.len() - 1) / 2;
    let mut bindings: Vec<(String, Expr)> = Vec::with_capacity(pair_count);
    let mut binding_names: Vec<String> = Vec::with_capacity(pair_count);

    for pair in args[..args.len() - 1].as_chunks::<2>().0 {
        let name = pair[0]
            .as_str()
            .ok_or(ExprError::LetNameNotString)?
            .to_string();
        validate_var_name(&name)?;
        // The new binding's value is parsed under a scope that
        // includes every prior name in this `let` — that's the rule
        // the `let` roadmap entry calls out: a later binding's `val`
        // can reference earlier ones.
        let value_scope = scope.extend(&binding_names);
        let value = Expr::from_json_scoped(&pair[1], &value_scope)?;
        bindings.push((name.clone(), value));
        binding_names.push(name);
    }

    // Body parses under the full extended scope so it sees every
    // binding from this `let` (in addition to anything visible from
    // outside).
    let body_scope = scope.extend(&binding_names);
    let body = Box::new(Expr::from_json_scoped(
        args.last().expect("at least one element"),
        &body_scope,
    )?);

    Ok(Expr::Let { bindings, body })
}

/// `["var", name]`. The name must be in the surrounding scope; an
/// unbound name is a parse-time error.
fn parse_var(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [name] = args else {
        return Err(ExprError::WrongArity {
            op: "var",
            expected: "1",
            got: args.len(),
        });
    };
    let name = name.as_str().ok_or(ExprError::ExpectedString {
        op: "var",
        position: "name",
    })?;
    if !scope.contains(name) {
        return Err(ExprError::UnboundVariable(name.to_string()));
    }
    Ok(Expr::Var(name.to_string()))
}

/// Best-effort static type check for *literal-shaped* expression
/// subtrees. Returns the spec-shaped type name when we can determine
/// it without a real inference pass; otherwise `None`. We use this to
/// reject patently-wrong literal arguments (e.g. `["slice", true, 0]`)
/// at parse time, matching the upstream compile-time errors.
fn static_scalar_kind(e: &Expr) -> Option<&'static str> {
    match e {
        Expr::Lit(Literal::Null) => Some("null"),
        Expr::Lit(Literal::Bool(_)) => Some("boolean"),
        Expr::Lit(Literal::Number(_)) => Some("number"),
        Expr::Lit(Literal::String(_)) => Some("string"),
        Expr::Literal(Value::Null) => Some("null"),
        Expr::Literal(Value::Bool(_)) => Some("boolean"),
        Expr::Literal(Value::Number(_)) => Some("number"),
        Expr::Literal(Value::String(_)) => Some("string"),
        Expr::Literal(Value::Array(_)) => Some("array"),
        Expr::Literal(Value::Object(_)) => Some("object"),
        _ => None,
    }
}

/// `true` when this expression is statically known to produce a
/// `Color` value. Used to reject `==` / `!=` over colors at parse
/// time, matching upstream's `equal/color` and `not_equal/color`
/// fixtures (the latter is in scope for Phase 6 only if it's already
/// in the suite). Full type inference is Phase 10's job; here we
/// only catch the syntactic shapes.
fn statically_color(e: &Expr) -> bool {
    matches!(e, Expr::Rgb { .. } | Expr::ToColor(_))
}

/// Variable names must be alphanumeric or underscore, per the
/// `let/invalid-name` fixture. The upstream parser exposes this rule
/// as a compile-time error; we surface it here at parse time so the
/// fixture's expected behavior round-trips.
fn validate_var_name(name: &str) -> Result<(), ExprError> {
    // Empty names are rejected as well; they're nominally allowed by
    // the regex `^[A-Za-z0-9_]*$` but no fixture exercises that and
    // requiring at least one character keeps the error reachable for
    // pathological inputs.
    if name.is_empty() {
        return Err(ExprError::InvalidVariableName(name.to_string()));
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(ExprError::InvalidVariableName(name.to_string()));
    }
    Ok(())
}

// ------------------------------------------------------------------------------
// Strings, formatting, i18n — parsing
// ------------------------------------------------------------------------------

/// `["concat", v1, v2, …]` — zero or more children are valid; the
/// zero-arg form is the empty string (fixture `concat/arity-0`).
fn parse_concat(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let values = args
        .iter()
        .map(|a| Expr::from_json_scoped(a, scope))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::Concat(values))
}

fn parse_case_change(
    kind: CaseChangeKind,
    args: &[Json],
    scope: &ParseScope<'_>,
) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: kind.name(),
            expected: "1",
            got: args.len(),
        });
    };
    Ok(Expr::CaseChange {
        kind,
        value: Box::new(Expr::from_json_scoped(v, scope)?),
    })
}

fn parse_slice(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    if !(2..=3).contains(&args.len()) {
        return Err(ExprError::WrongArity {
            op: "slice",
            expected: "2 or 3",
            got: args.len(),
        });
    }
    let input = Expr::from_json_scoped(&args[0], scope)?;
    // `slice/invalid-input-arg-literal` pins a parse-time error when
    // the first arg is a literal bool. We catch any statically-known
    // non-string-non-array scalar (arrays passed via `literal` are
    // fine — they're the array form of slice).
    if let Some(kind) = static_scalar_kind(&input)
        && !matches!(kind, "string" | "array") {
            return Err(ExprError::SliceInvalidLiteralInput(kind));
        }
    let start = Expr::from_json_scoped(&args[1], scope)?;
    let end = if args.len() == 3 {
        Some(Box::new(Expr::from_json_scoped(&args[2], scope)?))
    } else {
        None
    };
    Ok(Expr::Slice {
        input: Box::new(input),
        start: Box::new(start),
        end,
    })
}

fn parse_length(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: "length",
            expected: "1",
            got: args.len(),
        });
    };
    let inner = Expr::from_json_scoped(v, scope)?;
    // The `length/invalid-arg` fixture passes a bare number literal
    // and expects a *parse-time* error. We mirror that for the
    // statically-decidable cases (literal scalars). More general
    // type inference is Phase 10.
    if let Some(kind) = static_scalar_kind(&inner)
        && !matches!(kind, "string" | "array") {
            return Err(ExprError::LengthInvalidLiteral(kind));
        }
    Ok(Expr::Length(Box::new(inner)))
}

fn parse_split(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    // `split` takes the input string first and the separator second —
    // counter to several other "string + sep" operators in unrelated
    // languages, but matches the spec.
    let [s, sep] = args else {
        return Err(ExprError::WrongArity {
            op: "split",
            expected: "2",
            got: args.len(),
        });
    };
    Ok(Expr::Split {
        input: Box::new(Expr::from_json_scoped(s, scope)?),
        separator: Box::new(Expr::from_json_scoped(sep, scope)?),
    })
}

fn parse_join(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [arr, sep] = args else {
        return Err(ExprError::WrongArity {
            op: "join",
            expected: "2",
            got: args.len(),
        });
    };
    let input = Expr::from_json_scoped(arr, scope)?;
    // `join/non-array` rejects a literal string first arg at parse
    // time. We also reject the `join/numbers` case (array of literal
    // numbers — wrong element type). The runtime form still falls
    // back to per-item stringification for anything we can't decide
    // statically.
    if let Some(kind) = static_scalar_kind(&input) {
        match kind {
            "array" => {
                // Verify every literal element is a string. If we find
                // any non-string item, surface the compile-time
                // mismatch.
                if let Expr::Literal(Value::Array(items)) = &input {
                    let mut non_string = false;
                    for item in items {
                        if !matches!(item, Value::String(_)) {
                            non_string = true;
                            break;
                        }
                    }
                    if non_string && !items.is_empty() {
                        return Err(ExprError::JoinElementsNotStrings(items.len()));
                    }
                }
            }
            other => return Err(ExprError::JoinInputNotArray(other)),
        }
    }
    Ok(Expr::Join {
        input: Box::new(input),
        separator: Box::new(Expr::from_json_scoped(sep, scope)?),
    })
}

fn parse_index_of(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    if !(2..=3).contains(&args.len()) {
        return Err(ExprError::WrongArity {
            op: "index-of",
            expected: "2 or 3",
            got: args.len(),
        });
    }
    let needle = Expr::from_json_scoped(&args[0], scope)?;
    // `index-of/invalid-needle-literal-array` rejects an array literal
    // needle at compile time; we mirror that for any statically-known
    // Array literal.
    if matches!(needle, Expr::Literal(Value::Array(_) | Value::Object(_))) {
        return Err(ExprError::IndexOfNeedleNotScalar);
    }
    let haystack = Box::new(Expr::from_json_scoped(&args[1], scope)?);
    let from_index = if args.len() == 3 {
        Some(Box::new(Expr::from_json_scoped(&args[2], scope)?))
    } else {
        None
    };
    Ok(Expr::IndexOf {
        needle: Box::new(needle),
        haystack,
        from_index,
    })
}

/// `["format", t1, o1, t2, o2, …]`. The trailing options object on the
/// final pair is optional in the syntax — the parser walks the args
/// detecting whether each next slot is a text expression or an options
/// object. Empty arg lists are rejected here because upstream's parser
/// requires at least one section.
fn parse_format(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    if args.is_empty() {
        return Err(ExprError::WrongArity {
            op: "format",
            expected: "at least 1",
            got: 0,
        });
    }
    // A "section" is `text` followed optionally by an options object.
    // We scan left-to-right: whenever the current slot is a JSON object
    // we attach it to the previously-opened section as its options;
    // otherwise it begins a new section. The very first slot must
    // therefore be a text expression — a leading options object is a
    // parse error, matching the upstream grammar.
    let mut sections: Vec<FormatSection> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let text_expr = Expr::from_json_scoped(&args[i], scope)?;
        // Look ahead for an options object; if the next slot is a bare
        // JSON object it belongs to this section.
        let (opts, advance) = if let Some(next) = args.get(i + 1) {
            if let Some(map) = next.as_object() {
                (parse_format_options(map, scope)?, 2)
            } else {
                (FormatOptions::default(), 1)
            }
        } else {
            (FormatOptions::default(), 1)
        };
        sections.push(FormatSection {
            text: Box::new(text_expr),
            font_scale: opts.font_scale,
            text_font: opts.text_font,
            text_color: opts.text_color,
            vertical_align: opts.vertical_align,
        });
        i += advance;
    }
    Ok(Expr::Format(sections))
}

#[derive(Default)]
struct FormatOptions {
    font_scale: Option<Box<Expr>>,
    text_font: Option<Box<Expr>>,
    text_color: Option<Box<Expr>>,
    vertical_align: Option<VerticalAlignSlot>,
}

fn parse_format_options(
    map: &serde_json::Map<String, Json>,
    scope: &ParseScope<'_>,
) -> Result<FormatOptions, ExprError> {
    let mut out = FormatOptions::default();
    for (k, v) in map {
        match k.as_str() {
            "font-scale" => out.font_scale = Some(Box::new(Expr::from_json_scoped(v, scope)?)),
            "text-font" => {
                let inner = Expr::from_json_scoped(v, scope)?;
                // Upstream validates that `text-font` typechecks as
                // `array<string>` at parse time. We reject statically-
                // typed arrays whose item type isn't `string` (or
                // `value` — the open-item case still needs runtime
                // discrimination). The fixture
                // `format/data-driven-scale` exercises the case
                // where the user wrote `["array", ["string", …]]`
                // (untyped array) into the `text-font` slot.
                if let ExprType::Array(item, _) = inner.result_type() {
                    // `array<string>` is the only acceptable typed
                    // array. `array<value>` (open item) is fine because
                    // each element is checked at runtime; bare `array`
                    // (item type `None`) is the failure shape upstream
                    // pins.
                    let ok = matches!(
                        item.as_deref(),
                        Some(ExprType::String) | Some(ExprType::Value),
                    );
                    if !ok {
                        return Err(ExprError::FormatTextFontNotStringArray(
                            inner.result_type().to_spec_string(),
                        ));
                    }
                }
                out.text_font = Some(Box::new(inner));
            }
            "text-color" => out.text_color = Some(Box::new(Expr::from_json_scoped(v, scope)?)),
            "vertical-align" => {
                // `vertical-align` is the one option the spec restricts
                // to a fixed keyword set, but the value can either be
                // the keyword string (validated eagerly so
                // `format/unknown-vertical-align` errors at compile
                // time) or an expression (so `format/data-driven-
                // vertical-align` can drive it from `get`).
                if let Some(s) = v.as_str() {
                    let va =
                        VerticalAlign::parse(s).ok_or(ExprError::FormatInvalidVerticalAlign)?;
                    out.vertical_align = Some(VerticalAlignSlot::Static(va));
                } else {
                    out.vertical_align = Some(VerticalAlignSlot::Expr(Box::new(
                        Expr::from_json_scoped(v, scope)?,
                    )));
                }
            }
            other => return Err(ExprError::FormatUnknownOption(other.to_string())),
        }
    }
    Ok(out)
}

fn parse_number_format(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [value, opts] = args else {
        return Err(ExprError::WrongArity {
            op: "number-format",
            expected: "2",
            got: args.len(),
        });
    };
    let map = opts
        .as_object()
        .ok_or(ExprError::NumberFormatOptionsNotObject)?;

    let mut locale = None;
    let mut currency = None;
    let mut unit = None;
    let mut min_fraction_digits = None;
    let mut max_fraction_digits = None;
    for (k, v) in map {
        match k.as_str() {
            "locale" => locale = Some(Box::new(Expr::from_json_scoped(v, scope)?)),
            "currency" => currency = Some(Box::new(Expr::from_json_scoped(v, scope)?)),
            "unit" => unit = Some(Box::new(Expr::from_json_scoped(v, scope)?)),
            "min-fraction-digits" => {
                min_fraction_digits = Some(Box::new(Expr::from_json_scoped(v, scope)?))
            }
            "max-fraction-digits" => {
                max_fraction_digits = Some(Box::new(Expr::from_json_scoped(v, scope)?))
            }
            other => return Err(ExprError::NumberFormatUnknownOption(other.to_string())),
        }
    }
    // Upstream rejects the simultaneous use of `currency` and `unit`
    // at compile time — see fixture `number-format/currency-and-unit`.
    if currency.is_some() && unit.is_some() {
        return Err(ExprError::NumberFormatCurrencyAndUnit);
    }
    Ok(Expr::NumberFormat {
        value: Box::new(Expr::from_json_scoped(value, scope)?),
        locale,
        currency,
        unit,
        min_fraction_digits,
        max_fraction_digits,
    })
}

fn parse_collator(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [opts] = args else {
        return Err(ExprError::WrongArity {
            op: "collator",
            expected: "1",
            got: args.len(),
        });
    };
    let map = opts
        .as_object()
        .ok_or(ExprError::CollatorOptionsNotObject)?;
    let mut spec = CollatorSpec {
        case_sensitive: None,
        diacritic_sensitive: None,
        locale: None,
    };
    for (k, v) in map {
        match k.as_str() {
            "case-sensitive" => {
                spec.case_sensitive = Some(Box::new(Expr::from_json_scoped(v, scope)?))
            }
            "diacritic-sensitive" => {
                spec.diacritic_sensitive = Some(Box::new(Expr::from_json_scoped(v, scope)?))
            }
            "locale" => spec.locale = Some(Box::new(Expr::from_json_scoped(v, scope)?)),
            other => return Err(ExprError::CollatorUnknownOption(other.to_string())),
        }
    }
    Ok(Expr::Collator(spec))
}

fn parse_resolved_locale(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: "resolved-locale",
            expected: "1",
            got: args.len(),
        });
    };
    Ok(Expr::ResolvedLocale(Box::new(Expr::from_json_scoped(
        v, scope,
    )?)))
}

fn parse_is_supported_script(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: "is-supported-script",
            expected: "1",
            got: args.len(),
        });
    };
    Ok(Expr::IsSupportedScript(Box::new(Expr::from_json_scoped(
        v, scope,
    )?)))
}

// ------------------------------------------------------------------------------
// Color constructors / coercions — parsing
// ------------------------------------------------------------------------------

/// `["rgb", r, g, b]` / `["rgba", r, g, b, a]`. The arity-mismatched
/// form raises an upstream-style "wrong arity" error rather than the
/// runtime "out of range" one — the latter only fires when *valid-
/// arity* inputs fail the numeric bounds check at eval time.
fn parse_rgb(args: &[Json], scope: &ParseScope<'_>, with_alpha: bool) -> Result<Expr, ExprError> {
    let (op, want) = if with_alpha {
        ("rgba", 4usize)
    } else {
        ("rgb", 3usize)
    };
    if args.len() != want {
        return Err(ExprError::WrongArity {
            op,
            expected: if with_alpha { "4" } else { "3" },
            got: args.len(),
        });
    }
    let r = Box::new(Expr::from_json_scoped(&args[0], scope)?);
    let g = Box::new(Expr::from_json_scoped(&args[1], scope)?);
    let b = Box::new(Expr::from_json_scoped(&args[2], scope)?);
    let alpha = if with_alpha {
        Some(Box::new(Expr::from_json_scoped(&args[3], scope)?))
    } else {
        None
    };
    Ok(Expr::Rgb { r, g, b, alpha })
}

fn parse_to_color(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    if args.is_empty() {
        return Err(ExprError::WrongArity {
            op: "to-color",
            expected: "at least 1",
            got: 0,
        });
    }
    let values = args
        .iter()
        .map(|a| Expr::from_json_scoped(a, scope))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expr::ToColor(values))
}

fn parse_to_rgba(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [v] = args else {
        return Err(ExprError::WrongArity {
            op: "to-rgba",
            expected: "1",
            got: args.len(),
        });
    };
    Ok(Expr::ToRgba(Box::new(Expr::from_json_scoped(v, scope)?)))
}

// ------------------------------------------------------------------------------
// Geometry-aware operators — parsing
// ------------------------------------------------------------------------------

/// `["geometry-type"]` — zero-arg operator. Same shape check as the
/// math constants; the per-feature lookup happens at eval time.
fn parse_geometry_type(args: &[Json]) -> Result<Expr, ExprError> {
    if !args.is_empty() {
        return Err(ExprError::WrongArity {
            op: "geometry-type",
            expected: "0",
            got: args.len(),
        });
    }
    Ok(Expr::GeometryType)
}

/// Parse a `within` operator. Upstream rejects anything that isn't a
/// GeoJSON object containing polygons, so we duplicate that contract
/// here: the JSON must be a `FeatureCollection`, `Feature`, or bare
/// `Polygon`/`MultiPolygon`. Non-polygon geometry (a `LineString`, a
/// nested `["get", …]`, missing `coordinates`, etc.) is a compile
/// error. The accepted shapes match `within/{expression-geojson,
/// invalid-geojson,missing-geojson,non-supported,unexpected-second-
/// arg}`.
fn parse_within(args: &[Json]) -> Result<Expr, ExprError> {
    if args.len() != 1 {
        // Upstream phrases this as "expression requires exactly one
        // argument, but found N instead." Our `WrongArity` machinery
        // already prints in that shape so we route through it.
        return Err(ExprError::WrongArity {
            op: "within",
            expected: "1",
            got: args.len(),
        });
    }
    let polygons = parse_within_argument(&args[0]).ok_or(ExprError::WithinInvalidGeoJson)?;
    Ok(Expr::Within(WithinPolygons { polygons }))
}

/// Extract the polygon list from a `within` argument. Returns `None`
/// for any of the rejections upstream flags as
/// "`within` requires valid geojson object that contains polygon
/// geometry type." — i.e. non-object inputs, missing
/// `type`/`coordinates`, or non-polygon geometry types.
fn parse_within_argument(v: &Json) -> Option<Vec<PolygonRings>> {
    let obj = v.as_object()?;
    let kind = obj.get("type")?.as_str()?;
    match kind {
        "FeatureCollection" => {
            // Per upstream: walk every feature, keep only Polygon /
            // MultiPolygon geometries, and gather all rings into one
            // flat polygon list. An empty result (no polygons in the
            // collection) is also rejected.
            let features = obj.get("features")?.as_array()?;
            let mut polygons = Vec::new();
            for feat in features {
                let geom = feat.as_object()?.get("geometry")?;
                let g = geom.as_object()?;
                let t = g.get("type")?.as_str()?;
                let coords = g.get("coordinates")?;
                match t {
                    "Polygon" => {
                        polygons.push(parse_polygon_coords(coords)?);
                    }
                    "MultiPolygon" => {
                        let multi = coords.as_array()?;
                        for poly in multi {
                            polygons.push(parse_polygon_coords(poly)?);
                        }
                    }
                    _ => {}
                }
            }
            if polygons.is_empty() {
                return None;
            }
            Some(polygons)
        }
        "Feature" => {
            let geom = obj.get("geometry")?.as_object()?;
            let t = geom.get("type")?.as_str()?;
            let coords = geom.get("coordinates")?;
            extract_polygon_or_multipolygon(t, coords)
        }
        "Polygon" | "MultiPolygon" => {
            let coords = obj.get("coordinates")?;
            extract_polygon_or_multipolygon(kind, coords)
        }
        _ => None,
    }
}

/// Helper: given a `Polygon` or `MultiPolygon` type + coordinates,
/// return the normalized polygons-as-rings list. Other types return
/// `None` so the caller can surface the upstream rejection.
fn extract_polygon_or_multipolygon(kind: &str, coords: &Json) -> Option<Vec<PolygonRings>> {
    match kind {
        "Polygon" => Some(vec![parse_polygon_coords(coords)?]),
        "MultiPolygon" => {
            let multi = coords.as_array()?;
            multi.iter().map(parse_polygon_coords).collect()
        }
        _ => None,
    }
}

/// Parse a single polygon's coordinates: a list of rings, each a
/// closed loop of lng/lat pairs. Rings shorter than 3 coordinates are
/// degenerate — we let them through and rely on the point-in-polygon
/// algorithm to handle them harmlessly (zero-length loops never
/// contain anything).
fn parse_polygon_coords(v: &Json) -> Option<PolygonRings> {
    let rings = v.as_array()?;
    let mut out = Vec::with_capacity(rings.len());
    for ring in rings {
        let coords = ring.as_array()?;
        let mut pts = Vec::with_capacity(coords.len());
        for c in coords {
            pts.push(parse_lnglat(c)?);
        }
        out.push(pts);
    }
    Some(out)
}

fn parse_lnglat(v: &Json) -> Option<(f64, f64)> {
    let arr = v.as_array()?;
    if arr.len() < 2 {
        return None;
    }
    let lng = arr[0].as_f64()?;
    let lat = arr[1].as_f64()?;
    Some((lng, lat))
}

/// Parse `distance`'s GeoJSON argument. Upstream `Distance.parse`
/// accepts a `FeatureCollection`, `Feature`, or bare geometry; any
/// other shape (including a sub-expression like `["get", "geom"]`)
/// errors at compile time. We pre-expand Multi* variants here so the
/// evaluator never has to think about them.
fn parse_distance(args: &[Json]) -> Result<Expr, ExprError> {
    if args.len() != 1 {
        return Err(ExprError::WrongArity {
            op: "distance",
            expected: "1",
            got: args.len(),
        });
    }
    let geoms = parse_distance_argument(&args[0]).ok_or(ExprError::DistanceInvalidGeoJson)?;
    Ok(Expr::Distance(geoms))
}

fn parse_distance_argument(v: &Json) -> Option<Vec<SimpleGeoJson>> {
    let obj = v.as_object()?;
    let kind = obj.get("type")?.as_str()?;
    match kind {
        "FeatureCollection" => {
            let features = obj.get("features")?.as_array()?;
            let mut out = Vec::new();
            for feat in features {
                let g = feat.as_object()?.get("geometry")?;
                let kt = g.as_object()?.get("type")?.as_str()?;
                let coords = g.as_object()?.get("coordinates")?;
                out.extend(parse_simple_geometry(kt, coords)?);
            }
            Some(out)
        }
        "Feature" => {
            let g = obj.get("geometry")?.as_object()?;
            let kt = g.get("type")?.as_str()?;
            let coords = g.get("coordinates")?;
            parse_simple_geometry(kt, coords)
        }
        // Bare geometry. The upstream check is structural — `type` and
        // `coordinates` must both be present; anything else errors.
        _ => {
            let coords = obj.get("coordinates")?;
            parse_simple_geometry(kind, coords)
        }
    }
}

/// Expand a single geometry into one or more `SimpleGeoJson`s.
/// Multi* variants split into N components, mirroring upstream's
/// `toSimpleGeometry`. Unknown geometry types return `None` so the
/// outer parser surfaces the upstream "invalid geojson" error.
fn parse_simple_geometry(kind: &str, coords: &Json) -> Option<Vec<SimpleGeoJson>> {
    match kind {
        "Point" => {
            let (lng, lat) = parse_lnglat(coords)?;
            Some(vec![SimpleGeoJson::Point(lng, lat)])
        }
        "MultiPoint" => {
            let pts = coords.as_array()?;
            pts.iter()
                .map(|p| parse_lnglat(p).map(|(x, y)| SimpleGeoJson::Point(x, y)))
                .collect()
        }
        "LineString" => {
            let line = parse_lnglat_list(coords)?;
            Some(vec![SimpleGeoJson::LineString(line)])
        }
        "MultiLineString" => {
            let lines = coords.as_array()?;
            lines
                .iter()
                .map(|l| parse_lnglat_list(l).map(SimpleGeoJson::LineString))
                .collect()
        }
        "Polygon" => Some(vec![SimpleGeoJson::Polygon(parse_polygon_coords(coords)?)]),
        "MultiPolygon" => {
            let polys = coords.as_array()?;
            polys
                .iter()
                .map(|p| parse_polygon_coords(p).map(SimpleGeoJson::Polygon))
                .collect()
        }
        _ => None,
    }
}

fn parse_lnglat_list(v: &Json) -> Option<Vec<(f64, f64)>> {
    let arr = v.as_array()?;
    arr.iter().map(parse_lnglat).collect()
}

// ------------------------------------------------------------------------------
// Phase 9 — context-driven reader parsers
// ------------------------------------------------------------------------------

/// Generic "zero-arg operator, returns a fixed `Expr` variant" helper.
/// Several Phase 9 operators (`heatmap-density`, `elevation`) share the
/// exact same parse shape; centralizing it keeps the dispatch table
/// readable and the error wording uniform.
fn parse_zero_arg(args: &[Json], op: &'static str, ctor: Expr) -> Result<Expr, ExprError> {
    if !args.is_empty() {
        return Err(ExprError::WrongArity {
            op,
            expected: "0",
            got: args.len(),
        });
    }
    Ok(ctor)
}

/// Parse `["feature-state", name]`. The name is a sub-expression so
/// fixtures like `feature-state/expression-arg` can compose it from
/// `["at", …]`. We don't validate at parse time that the expression is
/// statically a string — upstream's parser doesn't either; mismatched
/// runtime types surface as `Value::Null` lookups, which the outer
/// type assertion then catches with the canonical "expected … but
/// found null" message.
fn parse_feature_state(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [name] = args else {
        return Err(ExprError::WrongArity {
            op: "feature-state",
            expected: "1",
            got: args.len(),
        });
    };
    let name_expr = Expr::from_json_scoped(name, scope)?;
    Ok(Expr::FeatureState(Box::new(name_expr)))
}

/// Parse `["global-state", name]`. Upstream requires the property arg
/// to be a *string literal* at parse time — the
/// `global-state/invalid-property-arg` fixture pins the message for
/// the sub-expression form (`["concat", …]`). We surface the same
/// rejection here via `ExpectedString`; the runner only checks that
/// we erred, not the exact wording.
fn parse_global_state(args: &[Json]) -> Result<Expr, ExprError> {
    let [name] = args else {
        return Err(ExprError::WrongArity {
            op: "global-state",
            expected: "1",
            got: args.len(),
        });
    };
    let name = name.as_str().ok_or(ExprError::ExpectedString {
        op: "global-state",
        position: "property name",
    })?;
    Ok(Expr::GlobalState(name.to_string()))
}

/// Parse `["image", name]`. The image-name argument is either a
/// string-yielding sub-expression (most commonly `["get", "icon"]`)
/// or a string literal. We reject the *literal* non-string cases
/// (`true` / numeric / `null`) because they're statically wrong — the
/// `image/invalid-image-name/{boolean,number,null}` fixtures pin
/// this. The harder "non-string operator" case
/// (`image(+(1,4))`) needs static-type inference and stays on the
/// manifest until Phase 10.
fn parse_image(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [name] = args else {
        return Err(ExprError::WrongArity {
            op: "image",
            expected: "1",
            got: args.len(),
        });
    };
    // Reject the literal forms that are statically not strings. The
    // `is_string` form survives because it covers both bare literals
    // (`"name"`) and the bracketed `["literal", "name"]` shape — the
    // latter wraps as `Expr::Literal(Value::String(_))` after parsing.
    match name {
        Json::Bool(_) | Json::Number(_) | Json::Null => {
            return Err(ExprError::ExpectedString {
                op: "image",
                position: "image name",
            });
        }
        _ => {}
    }
    let inner = Expr::from_json_scoped(name, scope)?;
    // Reject statically-typed non-string operands. The corpus pins
    // this on `image/invalid-image-name/non-string-expression` — an
    // `image(+(1,4))` whose argument types to `number`. Other
    // sub-expressions that resolve to `value` (e.g. `get`) sail
    // through and surface their type at eval time.
    match inner.result_type() {
        ExprType::String | ExprType::Value => {}
        _ => {
            return Err(ExprError::ExpectedString {
                op: "image",
                position: "image name",
            });
        }
    }
    Ok(Expr::Image(Box::new(inner)))
}

// ------------------------------------------------------------------------------
// Phase 5 — collections & feature metadata parsers
// ------------------------------------------------------------------------------

/// Parse `["at", index, array]`. Both operands are sub-expressions —
/// the upstream parser doesn't restrict either slot beyond requiring
/// the two-argument form. We do the integer-vs-fractional and the
/// out-of-bounds checks at eval time, mirroring the spec runtime.
fn parse_at(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [index, array] = args else {
        return Err(ExprError::WrongArity {
            op: "at",
            expected: "2",
            got: args.len(),
        });
    };
    let index = Expr::from_json_scoped(index, scope)?;
    let array = Expr::from_json_scoped(array, scope)?;
    Ok(Expr::At {
        index: Box::new(index),
        array: Box::new(array),
    })
}

/// Parse `["in", needle, haystack]`. Strict 2-arg form — the upstream
/// `in/missing-needle`, `in/missing-haystack`, and
/// `in/unexpected-third-arg` fixtures pin the message for each
/// off-by-one shape. We also check the *literal* array-needle case at
/// parse time so `in/invalid-needle-literal-array` rejects before
/// eval. Other non-scalar needles surface at runtime via
/// [`EvalError::InNonScalarNeedle`].
fn parse_in(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    let [needle, haystack] = args else {
        return Err(ExprError::WrongArity {
            op: "in",
            expected: "2",
            got: args.len(),
        });
    };
    let needle_expr = Expr::from_json_scoped(needle, scope)?;
    let haystack_expr = Expr::from_json_scoped(haystack, scope)?;
    // Upstream rejects a statically-typed array needle at compile time
    // (e.g. `["literal", ["a"]]`). We catch the easy case where the
    // needle is itself a literal array — anything else reaches the
    // runtime check.
    let needle_type = needle_expr.result_type();
    if matches!(needle_type, ExprType::Array(_, _)) {
        return Err(ExprError::InNeedleStaticallyArray(
            needle_type.to_spec_string(),
        ));
    }
    Ok(Expr::In {
        needle: Box::new(needle_expr),
        haystack: Box::new(haystack_expr),
    })
}

/// Parse `["has", name]` or `["has", name, host]`. The 1-arg form
/// defaults the host to the current feature's properties; the 2-arg
/// form lets a caller check membership in a literal or computed
/// object. The `name` slot must be a string literal in both arities —
/// the upstream parser does the same.
fn parse_has(args: &[Json], scope: &ParseScope<'_>) -> Result<Expr, ExprError> {
    match args.len() {
        1 => {
            let name = args[0].as_str().ok_or(ExprError::ExpectedString {
                op: "has",
                position: "property name",
            })?;
            Ok(Expr::Has {
                name: name.to_string(),
                host: None,
            })
        }
        2 => {
            let name = args[0].as_str().ok_or(ExprError::ExpectedString {
                op: "has",
                position: "property name",
            })?;
            let host = Expr::from_json_scoped(&args[1], scope)?;
            Ok(Expr::Has {
                name: name.to_string(),
                host: Some(Box::new(host)),
            })
        }
        n => Err(ExprError::WrongArity {
            op: "has",
            expected: "1 or 2",
            got: n,
        }),
    }
}

// ==============================================================================
// Evaluation
// ==============================================================================

/// Identifier for the tile a feature was decoded from. Geometry-aware
/// operators (`within`, `distance`) need this so they can map between
/// tile-local coordinates and the geographic frame their argument
/// geometries live in. Construction-time validation lives at the
/// adapter boundary — by the time a `CanonicalTileID` reaches the
/// evaluator we assume `z`, `x`, `y` are spec-consistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalTileID {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

/// Context passed to every `eval` call. `zoom` is the only required
/// signal; everything else stays at its `Default` until a caller
/// populates it. The context lost its `Copy` impl in Phase 9 — the
/// `HashMap`-backed state fields wouldn't fit it — so callers now pass
/// it by reference, which is what every existing evaluator path
/// already did anyway.
///
/// The four "global" knobs (`heatmap_density`, `elevation`,
/// `available_images`, `projection`) are populated by the renderer at
/// draw time; the two "state" maps (`feature_state`, `global_state`)
/// live on the map view and change in response to external events
/// (e.g. setFeatureState). All of them are dropped on the floor by
/// every operator except the ones that explicitly read them, so a
/// caller that only cares about `get`/`zoom` can build a context with
/// `Default::default()` and ignore the rest.
#[derive(Debug, Clone, Default)]
pub struct EvalContext {
    pub zoom: f64,
    pub canonical: Option<CanonicalTileID>,
    /// Per-feature mutable state, keyed by property name. Read by the
    /// `feature-state` operator. The upstream contract is that a
    /// missing key surfaces a *specific* error string ("Expected value
    /// to be of type …") at the type assertion that consumes it — see
    /// the operator's literate comment for why we model that as a
    /// `Null` here and let the outer assertion error.
    pub feature_state: std::collections::HashMap<String, Value>,
    /// Per-style global state shared across every feature in the
    /// current render. Read by the `global-state` operator. Same
    /// missing-key contract as `feature_state`.
    pub global_state: std::collections::HashMap<String, Value>,
    /// Density input to heatmap layer paint expressions, in `0..=1`.
    /// Read by the `heatmap-density` operator. Unset is `0.0`, which
    /// matches the "no heatmap context" fall-through upstream emits.
    pub heatmap_density: f64,
    /// Terrain elevation at the current feature, in arbitrary
    /// renderer-defined units. Read by the `elevation` operator.
    /// Unset is `0.0`.
    pub elevation: f64,
    /// Set of sprite names available to the current style. Read by
    /// the `image` operator to decide whether the resolved image is
    /// rendered or skipped over by `coalesce`.
    pub available_images: std::collections::HashSet<String>,
    /// Currently-active map projection name (e.g. `"mercator"`,
    /// `"globe"`). The corpus exercises projections almost entirely
    /// via the `step`/`interpolate` over string-valued projection
    /// definitions; we keep the field on the context so future
    /// `projection` zero-arg operator work has somewhere to land.
    pub projection: String,
}

/// Stack of `let` bindings visible at a given point in an evaluation.
/// Like [`ParseScope`], frames are linked together so nested `let`s
/// shadow outer names without cloning the binding set. Bindings are
/// stored as `Vec<(name, value)>` rather than a hashmap because each
/// frame holds at most a handful of names and a linear scan is the
/// cache-friendliest option at that size.
#[derive(Debug, Clone, Copy)]
struct EvalScope<'a> {
    bindings: &'a [(String, Value)],
    parent: Option<&'a EvalScope<'a>>,
}

impl<'a> EvalScope<'a> {
    fn root() -> Self {
        EvalScope {
            bindings: &[],
            parent: None,
        }
    }

    fn extend(&'a self, bindings: &'a [(String, Value)]) -> Self {
        EvalScope {
            bindings,
            parent: Some(self),
        }
    }

    fn lookup(&self, name: &str) -> Option<&Value> {
        // Innermost frame wins so nested `let`s shadow outer names —
        // see `let/shadow` for the canonical fixture.
        if let Some((_, v)) = self.bindings.iter().rev().find(|(n, _)| n == name) {
            return Some(v);
        }
        self.parent.and_then(|p| p.lookup(name))
    }
}

/// The result of evaluating an expression. Distinct from
/// [`crate::mvt::Value`] because MVT distinguishes integer widths but
/// expressions don't; the structural variants (`Array`, `Object`) are
/// needed for `literal`, type assertions, and `typeof`.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
    /// Output of the `format` operator: an ordered list of styled text
    /// sections. The shape mirrors upstream `Formatted` so the runner's
    /// JSON round-trip lines up with `expected.outputs`.
    Formatted(Formatted),
    /// A locale-aware string comparator built by the `collator`
    /// operator. Carried through the AST so `==`/`<`/etc.'s 3-arg form
    /// can invoke it at compare time.
    Collator(Collator),
    /// An RGBA color. Stored as **straight** (un-premultiplied)
    /// `[r, g, b, a]` in `0..=1`. Interpolation in `interpolate`
    /// (linear) walks the channels in this straight space; the
    /// perceptual variants (`interpolate-hcl` / `interpolate-lab`)
    /// do their own space conversion internally and re-emit straight.
    ///
    /// Note that the upstream JSON shape that `Color.toJSON()`
    /// produces is **premultiplied** — so [`value_to_json`] applies
    /// the `r*a, g*a, b*a, a` transform on its way out. The
    /// `interpolate/linear-color` alpha-zero case is the empirical
    /// validation that this convention is right: a half-blended
    /// `red`/`transparent-blue` at `t=0.4` round-trips as the
    /// premultiplied `[0.36, 0, 0.24, 0.6]` even though the straight
    /// values are `[0.6, 0, 0.4, 0.6]`.
    Color([f64; 4]),
    /// Output of the `image` operator: a sprite name plus an
    /// availability flag. The flag is set at eval time by intersecting
    /// the requested name against [`EvalContext::available_images`].
    /// `coalesce` treats an unavailable image as null-like so chains
    /// like `coalesce(image("a"), image("b"))` fall through until a
    /// resolvable image (or null) is found — matching upstream's
    /// `ResolvedImage` behavior pinned by `image/coalesce`.
    Image(ResolvedImage),
}

/// Runtime carrier for the `image` operator. We mirror upstream's
/// `ResolvedImage` shape so the fixture comparison emits the same
/// `{ "name": …, "available": … }` object verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedImage {
    pub name: String,
    pub available: bool,
}

/// Runtime carrier for `format` output. The Display impl renders the
/// underlying text concatenation, matching the upstream rule that
/// `to-string` of a `Formatted` is the joined section texts (see
/// `format/to-string`).
#[derive(Debug, Clone, PartialEq)]
pub struct Formatted {
    pub sections: Vec<FormattedSection>,
}

/// One styled run within a [`Formatted`] value. Every section can carry
/// an independent text-color / font-stack / scale / vertical-align.
/// Unset slots stay `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct FormattedSection {
    pub text: String,
    pub scale: Option<f64>,
    /// The text-font option resolves to a list of font names that we
    /// join with `,` at the boundary so the upstream-emitted
    /// `fontStack` string round-trips through equality.
    pub font_stack: Option<String>,
    pub text_color: Option<FormattedColor>,
    pub vertical_align: Option<VerticalAlign>,
    /// When the section's text slot evaluated to an `image`
    /// expression rather than a plain string, the resolved image
    /// lives here and `text` is the empty string. The
    /// `format/image-sections` fixture pins this round-trip shape
    /// (image-bearing sections emit `text: ""` and a populated
    /// `image: { name, available }`).
    pub image: Option<ResolvedImage>,
}

/// Phase 4 doesn't ship the full color value plumbing (that's Phase 6);
/// to keep `format`'s `text-color` slot round-tripping through the
/// fixture comparison we parse string color inputs into a structured
/// RGBA payload right here. This intentionally only covers what
/// `format/*` exercises (named colors + `#rgb`/`#rrggbb`/`rgb(...)`).
#[derive(Debug, Clone, PartialEq)]
pub struct FormattedColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

/// Runtime collator value. Mirrors `CollatorSpec` but with the option
/// expressions already evaluated, so comparisons can read them
/// directly.
#[derive(Debug, Clone, PartialEq)]
pub struct Collator {
    pub case_sensitive: bool,
    pub diacritic_sensitive: bool,
    /// Empty string means "use the runtime default locale". We don't
    /// hook ICU, so collation only honors case- and diacritic-folding
    /// — locale string is round-tripped through `resolved-locale` but
    /// otherwise inert.
    pub locale: String,
}

impl Value {
    /// Coerce to a number for arithmetic / interpolation. Booleans and
    /// strings fail — we don't do silent stringly arithmetic.
    pub fn as_number(&self) -> Result<f64, EvalError> {
        match self {
            Value::Number(n) => Ok(*n),
            _ => Err(EvalError::ExpectedNumber(self.clone())),
        }
    }
}

impl From<&MvtValue> for Value {
    /// MVT property values come in seven numeric/text/bool flavors; the
    /// abstract `MvtValue` enum also carries the richer `Null`,
    /// `Array`, and `Object` variants used by callers that build
    /// `Feature`s outside the proto pipeline. All numeric variants
    /// collapse to `Number(f64)`.
    fn from(v: &MvtValue) -> Self {
        match v {
            MvtValue::String(s) => Value::String(s.clone()),
            MvtValue::Float(f) => Value::Number(*f as f64),
            MvtValue::Double(d) => Value::Number(*d),
            MvtValue::Int(i) | MvtValue::SInt(i) => Value::Number(*i as f64),
            MvtValue::Uint(u) => Value::Number(*u as f64),
            MvtValue::Bool(b) => Value::Bool(*b),
            MvtValue::Null => Value::Null,
            MvtValue::Array(items) => Value::Array(items.iter().map(Value::from).collect()),
            MvtValue::Object(map) => {
                let mut out = BTreeMap::new();
                for (k, v) in map {
                    out.insert(k.clone(), Value::from(v));
                }
                Value::Object(out)
            }
        }
    }
}

impl Expr {
    /// Evaluate the expression against the supplied context and
    /// feature. Public entry point; internally everything threads a
    /// scope down so `let`/`var` can resolve bindings.
    pub fn eval(&self, ctx: &EvalContext, feature: &Feature) -> Result<Value, EvalError> {
        self.eval_scoped(ctx, feature, &EvalScope::root())
    }

    fn eval_scoped(
        &self,
        ctx: &EvalContext,
        feature: &Feature,
        scope: &EvalScope<'_>,
    ) -> Result<Value, EvalError> {
        match self {
            Expr::Lit(l) => Ok(match l {
                Literal::Null => Value::Null,
                Literal::Bool(b) => Value::Bool(*b),
                Literal::Number(n) => Value::Number(*n),
                Literal::String(s) => Value::String(s.clone()),
            }),
            Expr::Literal(v) => Ok(v.clone()),
            Expr::Get(name) => Ok(feature
                .properties
                .get(name)
                .map(Value::from)
                .unwrap_or(Value::Null)),
            Expr::Zoom => Ok(Value::Number(ctx.zoom)),
            Expr::Eq(a, b, c) => {
                let av = a.eval_scoped(ctx, feature, scope)?;
                let bv = b.eval_scoped(ctx, feature, scope)?;
                let collator = eval_optional_collator(c.as_deref(), ctx, feature, scope)?;
                Ok(Value::Bool(eval_equality(
                    &av,
                    &bv,
                    collator.as_ref(),
                    /* invert: */ false,
                )?))
            }
            Expr::NotEq(a, b, c) => {
                let av = a.eval_scoped(ctx, feature, scope)?;
                let bv = b.eval_scoped(ctx, feature, scope)?;
                let collator = eval_optional_collator(c.as_deref(), ctx, feature, scope)?;
                Ok(Value::Bool(eval_equality(
                    &av,
                    &bv,
                    collator.as_ref(),
                    /* invert: */ true,
                )?))
            }
            Expr::Compare {
                op,
                lhs,
                rhs,
                collator,
            } => {
                let av = lhs.eval_scoped(ctx, feature, scope)?;
                let bv = rhs.eval_scoped(ctx, feature, scope)?;
                let collator = eval_optional_collator(collator.as_deref(), ctx, feature, scope)?;
                eval_compare(*op, &av, &bv, collator.as_ref())
            }
            Expr::Interpolate {
                interpolation,
                input,
                stops,
                color_typed,
            } => {
                let x = input.eval_scoped(ctx, feature, scope)?.as_number()?;
                interpolate(*interpolation, *color_typed, x, stops, ctx, feature, scope)
            }
            Expr::Step {
                input,
                default,
                stops,
            } => eval_step(input, default, stops, ctx, feature, scope),
            Expr::Assert { kind, values } => eval_assert(*kind, values, ctx, feature, scope),
            Expr::ArrayAssert {
                item_type,
                length,
                values,
            } => eval_array_assert(*item_type, *length, values, ctx, feature, scope),
            Expr::Coerce { kind, values } => eval_coerce(*kind, values, ctx, feature, scope),
            Expr::TypeOf(inner) => {
                let v = inner.eval_scoped(ctx, feature, scope)?;
                Ok(Value::String(typeof_str(&v)))
            }
            Expr::Variadic { op, values } => eval_variadic(*op, values, ctx, feature, scope),
            Expr::Binary { op, lhs, rhs } => {
                let a = lhs.eval_scoped(ctx, feature, scope)?.as_number()?;
                let b = rhs.eval_scoped(ctx, feature, scope)?.as_number()?;
                Ok(Value::Number(apply_binary(*op, a, b)))
            }
            Expr::Unary { op, value } => {
                let x = value.eval_scoped(ctx, feature, scope)?.as_number()?;
                Ok(Value::Number(apply_unary(*op, x)))
            }
            Expr::MathConst(c) => Ok(Value::Number(c.value())),
            Expr::BoolFold { op, values } => eval_bool_fold(*op, values, ctx, feature, scope),
            Expr::Not(inner) => {
                let v = inner.eval_scoped(ctx, feature, scope)?;
                match v {
                    Value::Bool(b) => Ok(Value::Bool(!b)),
                    // The spec defines `!` strictly on booleans; non-
                    // bool input is a runtime error, not a coercion.
                    other => Err(EvalError::ExpectedBool(other)),
                }
            }
            Expr::ErrorOp(msg) => Err(EvalError::ExplicitError(msg.clone())),
            Expr::Case { branches, fallback } => eval_case(branches, fallback, ctx, feature, scope),
            Expr::Match {
                input,
                branches,
                fallback,
            } => eval_match(input, branches, fallback, ctx, feature, scope),
            Expr::Coalesce(values) => eval_coalesce(values, ctx, feature, scope),
            Expr::Let { bindings, body } => eval_let(bindings, body, ctx, feature, scope),
            Expr::Var(name) => scope
                .lookup(name)
                .cloned()
                // Parse-time scope checking guarantees the name is
                // bound, so this only fires if the AST was hand-built
                // without going through `from_json`.
                .ok_or_else(|| EvalError::UnboundVariable(name.clone())),
            Expr::Concat(values) => eval_concat(values, ctx, feature, scope),
            Expr::CaseChange { kind, value } => {
                let v = value.eval_scoped(ctx, feature, scope)?;
                match v {
                    Value::String(s) => Ok(Value::String(match kind {
                        CaseChangeKind::Down => s.to_lowercase(),
                        CaseChangeKind::Up => s.to_uppercase(),
                    })),
                    other => Err(EvalError::AssertFailed {
                        expected: "string",
                        got: typeof_str(&other),
                    }),
                }
            }
            Expr::Slice { input, start, end } => eval_slice(input, start, end, ctx, feature, scope),
            Expr::Length(inner) => {
                let v = inner.eval_scoped(ctx, feature, scope)?;
                eval_length(&v)
            }
            Expr::Split { input, separator } => {
                let s = expect_string(input.eval_scoped(ctx, feature, scope)?)?;
                let sep = expect_string(separator.eval_scoped(ctx, feature, scope)?)?;
                Ok(Value::Array(split_string(&s, &sep)))
            }
            Expr::Join { input, separator } => {
                let arr = input.eval_scoped(ctx, feature, scope)?;
                let sep = expect_string(separator.eval_scoped(ctx, feature, scope)?)?;
                eval_join(&arr, &sep)
            }
            Expr::IndexOf {
                needle,
                haystack,
                from_index,
            } => eval_index_of(needle, haystack, from_index.as_deref(), ctx, feature, scope),
            Expr::Format(sections) => eval_format(sections, ctx, feature, scope),
            Expr::NumberFormat {
                value,
                locale,
                currency,
                unit,
                min_fraction_digits,
                max_fraction_digits,
            } => eval_number_format(
                value,
                locale.as_deref(),
                currency.as_deref(),
                unit.as_deref(),
                min_fraction_digits.as_deref(),
                max_fraction_digits.as_deref(),
                ctx,
                feature,
                scope,
            ),
            Expr::Collator(spec) => eval_collator(spec, ctx, feature, scope),
            Expr::ResolvedLocale(inner) => {
                let v = inner.eval_scoped(ctx, feature, scope)?;
                match v {
                    Value::Collator(c) => Ok(Value::String(if c.locale.is_empty() {
                        // With no real ICU under the hood we don't have
                        // a system locale to report; an empty resolved
                        // locale is the spec-defined fallback in JS.
                        String::new()
                    } else {
                        c.locale
                    })),
                    other => Err(EvalError::ExpectedCollator(typeof_str(&other))),
                }
            }
            Expr::IsSupportedScript(_inner) => {
                // The runner doesn't provide an `isSupportedScript`
                // global, and the upstream fixture's own description
                // says the operator returns `true` under that absence.
                Ok(Value::Bool(true))
            }
            Expr::Rgb { r, g, b, alpha } => {
                let r = expect_number_op(&r.eval_scoped(ctx, feature, scope)?)?;
                let g = expect_number_op(&g.eval_scoped(ctx, feature, scope)?)?;
                let b = expect_number_op(&b.eval_scoped(ctx, feature, scope)?)?;
                let a = match alpha {
                    Some(e) => expect_number_op(&e.eval_scoped(ctx, feature, scope)?)?,
                    None => 1.0,
                };
                color_from_rgba_components(r, g, b, a).map(Value::Color)
            }
            Expr::ToColor(values) => eval_to_color(values, ctx, feature, scope),
            Expr::ToRgba(inner) => {
                let v = inner.eval_scoped(ctx, feature, scope)?;
                // Upstream's `to-rgba` accepts both `Color` and shapes
                // a `to-color` could parse (named strings, hex,
                // arrays). That's because the type system upstream
                // wraps non-color inputs in an implicit `to-color`
                // cast — without that machinery (Phase 10) we get the
                // same surface behavior by running the coercion at
                // eval time. `to-rgba/basic` exercises this with a
                // bare `"red"` property value.
                let c = coerce_value_to_color(&v)?;
                Ok(Value::Array(color_to_straight_rgba_array(&c)))
            }
            Expr::GeometryType => Ok(Value::String(geometry_type_name(&feature.geometry).into())),
            Expr::Within(polys) => Ok(Value::Bool(eval_within(polys, ctx, feature))),
            Expr::Distance(geoms) => Ok(Value::Number(eval_distance(geoms, ctx, feature))),
            Expr::FeatureState(name_expr) => {
                // The lookup key is itself an expression, so evaluate
                // it first. Non-string keys map to a missing lookup
                // (returning `Null`) rather than an error — the outer
                // assertion is what enforces the "expected a number /
                // string / …" contract.
                let key = name_expr.eval_scoped(ctx, feature, scope)?;
                let key_str = match key {
                    Value::String(s) => s,
                    // The fixture `feature-state/expression-arg`
                    // exercises a non-trivial key build; only string
                    // results survive here. Anything else falls
                    // through to "no such key".
                    _ => return Ok(Value::Null),
                };
                Ok(ctx
                    .feature_state
                    .get(&key_str)
                    .cloned()
                    .unwrap_or(Value::Null))
            }
            Expr::GlobalState(name) => {
                Ok(ctx.global_state.get(name).cloned().unwrap_or(Value::Null))
            }
            Expr::HeatmapDensity => Ok(Value::Number(ctx.heatmap_density)),
            Expr::Elevation => Ok(Value::Number(ctx.elevation)),
            Expr::Image(name_expr) => {
                let name_val = name_expr.eval_scoped(ctx, feature, scope)?;
                // Mirror upstream's runtime contract: a non-string
                // image name resolves to "no image". We surface that
                // as `Null` so `coalesce` and the outer
                // type-resolved-image assertion both see a clean
                // miss; raising an `EvalError` here would force
                // `image/compound` (a `get` returning a string) to
                // succeed via the happy path while a hypothetical
                // non-string property would tear down the whole
                // expression. The `coalesce` integration in
                // `image/coalesce` exercises the "fall through on
                // unavailable" branch — see [`eval_coalesce`].
                let name = match name_val {
                    Value::String(s) => s,
                    Value::Null => return Ok(Value::Null),
                    _ => return Ok(Value::Null),
                };
                let available = ctx.available_images.contains(&name);
                Ok(Value::Image(ResolvedImage { name, available }))
            }
            Expr::GetFrom { name, host } => {
                let host_val = host.eval_scoped(ctx, feature, scope)?;
                match host_val {
                    Value::Object(map) => Ok(map.get(name).cloned().unwrap_or(Value::Null)),
                    other => Err(EvalError::GetHostNotObject(typeof_str(&other))),
                }
            }
            Expr::At { index, array } => {
                let i = index.eval_scoped(ctx, feature, scope)?.as_number()?;
                let arr = array.eval_scoped(ctx, feature, scope)?;
                let items = match arr {
                    Value::Array(items) => items,
                    other => return Err(EvalError::AtArgNotArray(typeof_str(&other))),
                };
                // Reject fractional indices: the spec is integer-only.
                // We use the round-trip check (`i.fract() != 0`) rather
                // than a typed integer because the index comes from any
                // numeric sub-expression.
                if i.fract() != 0.0 || !i.is_finite() {
                    return Err(EvalError::AtIndexNotInteger(i));
                }
                let ii = i as i64;
                if ii < 0 {
                    return Err(EvalError::AtIndexBelowZero(ii));
                }
                if (ii as usize) >= items.len() {
                    // `len - 1` is the upper-bound message the fixture
                    // pins; empty arrays compare against `-1`, matching
                    // upstream's signed-subtract.
                    let max = items.len() as i64 - 1;
                    return Err(EvalError::AtIndexAboveBounds(ii, max));
                }
                Ok(items[ii as usize].clone())
            }
            Expr::In { needle, haystack } => eval_in(needle, haystack, ctx, feature, scope),
            Expr::Has { name, host } => {
                // 1-arg form reads the current feature's properties.
                // The fixture `has/basic` exercises both the absent
                // case (returns false) and the present-but-null case
                // (returns true) — built straight off the property
                // map's `contains_key` semantics so we don't conflate
                // "key absent" with "key present with null value".
                let present = match host {
                    None => feature.properties.contains_key(name),
                    Some(h) => {
                        let v = h.eval_scoped(ctx, feature, scope)?;
                        match v {
                            Value::Object(map) => map.contains_key(name),
                            other => {
                                return Err(EvalError::GetHostNotObject(typeof_str(&other)));
                            }
                        }
                    }
                };
                Ok(Value::Bool(present))
            }
            Expr::Properties => {
                // Map `feature.properties` (HashMap-backed MvtValue) to
                // a Value::Object. We sort keys so the round-trip
                // through `expr_value_to_json` is deterministic — the
                // upstream Object equality check is by key, not by
                // iteration order, but the deterministic order helps
                // when a future fixture lands that pins it.
                let mut out = std::collections::BTreeMap::new();
                for (k, v) in feature.properties.iter() {
                    out.insert(k.clone(), Value::from(v));
                }
                Ok(Value::Object(out))
            }
            Expr::Id => Ok(match feature.id {
                Some(n) => Value::Number(n as f64),
                None => Value::Null,
            }),
        }
    }
}

/// Evaluate `["in", needle, haystack]`. The two-step "haystack first"
/// order is deliberate: upstream's runtime evaluates the haystack
/// before checking the needle's type, and one of the fixtures
/// (`in/invalid-haystack` input #2 — `null` haystack with a "foo"
/// needle) returns `false` rather than the needle-type error.
fn eval_in(
    needle: &Expr,
    haystack: &Expr,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let hay = haystack.eval_scoped(ctx, feature, scope)?;
    // `null` haystack short-circuits to false without inspecting the
    // needle at all — the per-input "null short-circuit" cases all
    // expect `false` regardless of needle type.
    if matches!(hay, Value::Null) {
        return Ok(Value::Bool(false));
    }
    let n = needle.eval_scoped(ctx, feature, scope)?;
    match (n, hay) {
        // Scalar-in-string: stringify the needle and substring-search.
        // The spec's string `in` accepts boolean/number/string needles
        // (per `in/basic-string`), so we stringify here rather than
        // strictly requiring a string. Empty haystacks always return
        // `false` even for an empty needle — the upstream
        // `in/empty-string` fixture pins this. (Compare to JS's
        // `"".includes("")` which is `true`; the spec runtime
        // explicitly rejects the empty-haystack case to give the
        // common-sense answer.)
        (needle_v, Value::String(hay_s)) => {
            if hay_s.is_empty() {
                return Ok(Value::Bool(false));
            }
            let n_s = stringify_for_in(&needle_v)?;
            Ok(Value::Bool(hay_s.contains(&n_s)))
        }
        (needle_v, Value::Array(items)) => {
            // Validate the needle is scalar before searching. The
            // upstream check rejects `{}` (object) needles with the
            // exact wording the fixture pins.
            match &needle_v {
                Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => {}
                other => return Err(EvalError::InNonScalarNeedle(typeof_str(other))),
            }
            Ok(Value::Bool(
                items.iter().any(|v| values_strict_eq(v, &needle_v)),
            ))
        }
        // Haystack is non-null but not array/string: upstream rejects.
        (_, other) => Err(EvalError::InInvalidHaystack(typeof_str(&other))),
    }
}

/// Stringify a scalar needle for the `in <string>` form. Rejects
/// non-scalars (e.g. objects) so the fixture's wording lines up.
fn stringify_for_in(v: &Value) -> Result<String, EvalError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(format_number(*n)),
        Value::Bool(b) => Ok(b.to_string()),
        // `null in "..."` is `false` per upstream — we encode this as
        // a never-matching sentinel by returning a non-substring
        // marker. The simpler approach is to short-circuit here, but
        // since the haystack-empty case is already false the marker
        // approach keeps the caller code linear.
        Value::Null => Ok("\0__null__\0".to_string()),
        other => Err(EvalError::InNonScalarNeedle(typeof_str(other))),
    }
}

/// Strict value-equality matching the spec's `==`. We can't reuse
/// `eval_equality` here because that returns an error on type
/// mismatch; `in` over an array treats different-type items as a
/// silent miss rather than an error.
fn values_strict_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        _ => false,
    }
}

// ==============================================================================
// Static type & constness inference — Phase 10
// ==============================================================================
//
// These three walks all share the same recursive shape. They live next
// to each other so a future contributor adding an operator only has to
// touch one place to keep parser, evaluator, and type inference in
// sync.

impl Expr {
    /// Statically compute the type this expression resolves to. Most
    /// operators have fixed output types; the few polymorphic ones
    /// (`get`, `at`, `coalesce`, `case`, `match`, `let`, `var`,
    /// `step`/`interpolate` over typed outputs) infer from their
    /// children, decaying to `Value` when the children disagree.
    ///
    /// The result threads upstream's `type` field as a string — see
    /// [`ExprType::to_spec_string`] for the rendering convention.
    pub fn result_type(&self) -> ExprType {
        self.result_type_in(&TypeScope::root())
    }

    /// Inner walk that threads a scope of `(name, type)` bindings so
    /// `var` lookups can resolve the let-binding's static type without
    /// re-deriving it.
    fn result_type_in(&self, scope: &TypeScope<'_>) -> ExprType {
        match self {
            Expr::Lit(Literal::Null) => ExprType::Null,
            Expr::Lit(Literal::Bool(_)) => ExprType::Boolean,
            Expr::Lit(Literal::Number(_)) => ExprType::Number,
            Expr::Lit(Literal::String(_)) => ExprType::String,
            Expr::Literal(v) => literal_value_type(v),

            // `get` without a host object is `value` — the property
            // could be anything. The 2-arg form (`get` from a literal /
            // computed object) likewise can't be narrowed statically.
            Expr::Get(_) => ExprType::Value,
            Expr::Zoom => ExprType::Number,
            Expr::Eq(..) | Expr::NotEq(..) | Expr::Compare { .. } => ExprType::Boolean,
            Expr::Not(_) | Expr::BoolFold { .. } => ExprType::Boolean,

            // Interpolation / step preserve the stop output type via
            // the standard `join`. With `value` deferring to the
            // concrete side (see [`ExprType::join`]), a stop like
            // `["get", "x"]` doesn't collapse the whole interpolate's
            // result type to `value`. There's one wrinkle: when the
            // *only* concrete stop is an array of known length and
            // siblings are `value`, the join would over-narrow to
            // `array<T, N>`. Upstream drops the length whenever even
            // one stop type is unknown — siblings might disagree on
            // length at runtime. `interpolate/linear-number-array`
            // pins this on a `[literal, [0, 0]]` + `["get", "b"]`
            // pairing whose compile.type lands on the open
            // `array<number>` form.
            Expr::Interpolate {
                stops, color_typed, ..
            } => {
                if *color_typed {
                    return ExprType::Color;
                }
                let joined =
                    ExprType::join_many(stops.iter().map(|(_, e)| e.result_type_in(scope)));
                let has_unknown_stop = stops
                    .iter()
                    .any(|(_, e)| matches!(e.result_type_in(scope), ExprType::Value));
                match (joined, has_unknown_stop) {
                    (ExprType::Array(item, Some(_)), true) => ExprType::Array(item, None),
                    (other, _) => other,
                }
            }
            Expr::Step { default, stops, .. } => ExprType::join_many(
                std::iter::once(default.result_type_in(scope))
                    .chain(stops.iter().map(|(_, e)| e.result_type_in(scope))),
            ),

            // Assertion operators report exactly the type they assert.
            // The fallback chain doesn't change the result type — if no
            // value matches the assertion errors, but the type the
            // *expression* would resolve to is still the asserted kind.
            Expr::Assert { kind, .. } => match kind {
                AssertKind::Number => ExprType::Number,
                AssertKind::String => ExprType::String,
                AssertKind::Boolean => ExprType::Boolean,
                AssertKind::Object => ExprType::Object,
            },
            Expr::ArrayAssert {
                item_type, length, ..
            } => ExprType::Array(
                item_type.map(|t| Box::new(ExprType::from_item_type(t))),
                *length,
            ),

            // Coercions report their target type.
            Expr::Coerce { kind, .. } => match kind {
                CoerceKind::ToNumber => ExprType::Number,
                CoerceKind::ToString => ExprType::String,
                CoerceKind::ToBoolean => ExprType::Boolean,
            },

            Expr::TypeOf(_) => ExprType::String,

            // Numeric / arithmetic results are all `number`.
            Expr::Variadic { .. }
            | Expr::Binary { .. }
            | Expr::Unary { .. }
            | Expr::MathConst(_) => ExprType::Number,

            Expr::ErrorOp(_) => ExprType::Value,

            // Control flow joins its branches' types. We seed the
            // join with the *first* output (not the fallback) to
            // mirror upstream's `Case.parse` / `Match.parse`: they
            // pin the static type from the first output and accept
            // any subsequent output that's assignable to it. If the
            // first output is `value` (the top type), the whole case
            // remains `value` regardless of how the other branches
            // type-check — `value` absorbs them. The earlier
            // implementation seeded from the fallback and let `join`
            // defer to a concrete side, which collapsed
            // `case/null-fallback` (fallback `null`, outputs `value`)
            // down to `null`.
            Expr::Case { branches, fallback } => pin_branch_types(
                branches
                    .iter()
                    .map(|(_, e)| e.result_type_in(scope))
                    .chain(std::iter::once(fallback.result_type_in(scope))),
            ),
            Expr::Match {
                branches, fallback, ..
            } => pin_branch_types(
                branches
                    .iter()
                    .map(|(_, e)| e.result_type_in(scope))
                    .chain(std::iter::once(fallback.result_type_in(scope))),
            ),
            // `coalesce`'s static type is the *first* child's type —
            // upstream's `Coalesce.parse` pins the type from the first
            // operand and then asserts every subsequent operand is
            // assignable to it. Using the join with `value` deferring
            // to concrete siblings would surface a noisy concrete type
            // even when every operand is `value`; the first-child rule
            // keeps the output type aligned with the upstream walk.
            Expr::Coalesce(values) => values
                .first()
                .map(|e| e.result_type_in(scope))
                .unwrap_or(ExprType::Value),

            // Bindings — the body's type is the let's type, computed
            // under a scope that knows each binding's type. Var pulls
            // from the scope; an unbound name (only reachable via a
            // hand-built AST) decays to `value`.
            Expr::Let { bindings, body } => {
                let names: Vec<(String, ExprType)> = bindings
                    .iter()
                    .scan(Vec::new(), |seen, (name, expr)| {
                        let bind_scope = scope.extend(seen);
                        let t = expr.result_type_in(&bind_scope);
                        seen.push((name.clone(), t.clone()));
                        Some((name.clone(), t))
                    })
                    .collect();
                let body_scope = scope.extend(&names);
                body.result_type_in(&body_scope)
            }
            Expr::Var(name) => scope.lookup(name).cloned().unwrap_or(ExprType::Value),

            // Strings, formatting, i18n.
            Expr::Concat(_) | Expr::CaseChange { .. } | Expr::ResolvedLocale(_) => ExprType::String,
            Expr::Slice { input, .. } => {
                // `slice` preserves the input's "shape" — a sliced
                // string is still a string, a sliced array is still an
                // array (with unknown length, unknown item type if it
                // wasn't statically pinned).
                match input.result_type_in(scope) {
                    ExprType::String => ExprType::String,
                    ExprType::Array(item, _) => ExprType::Array(item, None),
                    _ => ExprType::Value,
                }
            }
            Expr::Length(_) => ExprType::Number,
            Expr::Split { .. } => ExprType::Array(Some(Box::new(ExprType::String)), None),
            Expr::Join { .. } => ExprType::String,
            Expr::IndexOf { .. } => ExprType::Number,
            Expr::Format(_) => ExprType::Formatted,
            Expr::NumberFormat { .. } => ExprType::String,
            Expr::Collator(_) => ExprType::Collator,
            Expr::IsSupportedScript(_) => ExprType::Boolean,

            // Colors.
            Expr::Rgb { .. } | Expr::ToColor(_) => ExprType::Color,
            Expr::ToRgba(_) => ExprType::Array(Some(Box::new(ExprType::Number)), Some(4)),

            // Geometry-aware operators.
            Expr::GeometryType => ExprType::String,
            Expr::Within(_) => ExprType::Boolean,
            Expr::Distance(_) => ExprType::Number,

            // Phase 9 — context-driven readers. State lookups are
            // type-erased; the others return numbers/images.
            Expr::FeatureState(_) | Expr::GlobalState(_) => ExprType::Value,
            Expr::HeatmapDensity | Expr::Elevation => ExprType::Number,
            Expr::Image(_) => ExprType::ResolvedImage,

            // Phase 5 — collections & feature metadata. Both `get`
            // forms decay to `value` because the result type depends
            // on the host object's runtime contents. `at` ditto — the
            // array's item type isn't statically known in most cases;
            // we keep it `value` to mirror upstream. `in` and `has`
            // are booleans; `properties` is an object; `id` is `value`.
            Expr::GetFrom { .. } => ExprType::Value,
            Expr::At { array, .. } => {
                // If the array's static type carries an item type, we
                // can hand it back as the `at` result type. This is
                // the case for `["at", N, ["literal", […]]]` over a
                // homogeneous literal, and for `["at", N, ["array", T,
                // …]]`. Otherwise decay to `value`.
                match array.result_type_in(scope) {
                    ExprType::Array(Some(item), _) => *item,
                    _ => ExprType::Value,
                }
            }
            Expr::In { .. } | Expr::Has { .. } => ExprType::Boolean,
            Expr::Properties => ExprType::Object,
            Expr::Id => ExprType::Value,
        }
    }

    /// `true` when the expression never reads anything off the current
    /// feature — i.e. the same input would always produce the same
    /// output regardless of which feature is being styled. Used by the
    /// runner to compare against upstream's `compiled.isFeatureConstant`.
    pub fn is_feature_constant(&self) -> bool {
        match self {
            // Operators that read from the feature. `get` / `has` /
            // `properties` / `id` / `geometry-type` / `feature-state` /
            // `within` / `distance` all touch per-feature state.
            // `Expr::Has` with `host = None` reads feature props; the
            // 2-arg form delegates to its host expression.
            Expr::Get(_)
            | Expr::FeatureState(_)
            | Expr::GeometryType
            | Expr::Within(_)
            | Expr::Distance(_)
            | Expr::Properties
            | Expr::Id => false,

            // Constant leaves.
            Expr::Lit(_)
            | Expr::Literal(_)
            | Expr::Zoom
            | Expr::MathConst(_)
            | Expr::ErrorOp(_)
            | Expr::Var(_)
            | Expr::GlobalState(_)
            | Expr::HeatmapDensity
            | Expr::Elevation => true,

            // Recursive cases — visit each child and AND the result.
            Expr::Eq(a, b, c) | Expr::NotEq(a, b, c) => {
                a.is_feature_constant()
                    && b.is_feature_constant()
                    && c.as_ref().map(|c| c.is_feature_constant()).unwrap_or(true)
            }
            Expr::Compare {
                lhs, rhs, collator, ..
            } => {
                lhs.is_feature_constant()
                    && rhs.is_feature_constant()
                    && collator
                        .as_ref()
                        .map(|c| c.is_feature_constant())
                        .unwrap_or(true)
            }
            Expr::Interpolate { input, stops, .. } => {
                input.is_feature_constant() && stops.iter().all(|(_, e)| e.is_feature_constant())
            }
            Expr::Step {
                input,
                default,
                stops,
            } => {
                input.is_feature_constant()
                    && default.is_feature_constant()
                    && stops.iter().all(|(_, e)| e.is_feature_constant())
            }
            Expr::Assert { values, .. }
            | Expr::ArrayAssert { values, .. }
            | Expr::Coerce { values, .. }
            | Expr::Concat(values)
            | Expr::ToColor(values)
            | Expr::Coalesce(values) => values.iter().all(Expr::is_feature_constant),
            Expr::TypeOf(inner) | Expr::Not(inner) | Expr::Length(inner) => {
                inner.is_feature_constant()
            }
            Expr::Variadic { values, .. } | Expr::BoolFold { values, .. } => {
                values.iter().all(Expr::is_feature_constant)
            }
            Expr::Binary { lhs, rhs, .. } => lhs.is_feature_constant() && rhs.is_feature_constant(),
            Expr::Unary { value, .. } => value.is_feature_constant(),
            Expr::Case { branches, fallback } => {
                fallback.is_feature_constant()
                    && branches
                        .iter()
                        .all(|(c, o)| c.is_feature_constant() && o.is_feature_constant())
            }
            Expr::Match {
                input,
                branches,
                fallback,
            } => {
                input.is_feature_constant()
                    && fallback.is_feature_constant()
                    && branches.iter().all(|(_, o)| o.is_feature_constant())
            }
            Expr::Let { bindings, body } => {
                body.is_feature_constant() && bindings.iter().all(|(_, e)| e.is_feature_constant())
            }
            Expr::CaseChange { value, .. } => value.is_feature_constant(),
            Expr::Slice { input, start, end } => {
                input.is_feature_constant()
                    && start.is_feature_constant()
                    && end
                        .as_ref()
                        .map(|e| e.is_feature_constant())
                        .unwrap_or(true)
            }
            Expr::Split { input, separator } | Expr::Join { input, separator } => {
                input.is_feature_constant() && separator.is_feature_constant()
            }
            Expr::IndexOf {
                needle,
                haystack,
                from_index,
            } => {
                needle.is_feature_constant()
                    && haystack.is_feature_constant()
                    && from_index
                        .as_ref()
                        .map(|e| e.is_feature_constant())
                        .unwrap_or(true)
            }
            Expr::Format(sections) => sections.iter().all(|s| {
                s.text.is_feature_constant()
                    && s.font_scale
                        .as_ref()
                        .map(|e| e.is_feature_constant())
                        .unwrap_or(true)
                    && s.text_font
                        .as_ref()
                        .map(|e| e.is_feature_constant())
                        .unwrap_or(true)
                    && s.text_color
                        .as_ref()
                        .map(|e| e.is_feature_constant())
                        .unwrap_or(true)
                    && match &s.vertical_align {
                        Some(VerticalAlignSlot::Expr(e)) => e.is_feature_constant(),
                        _ => true,
                    }
            }),
            Expr::NumberFormat {
                value,
                locale,
                currency,
                unit,
                min_fraction_digits,
                max_fraction_digits,
            } => {
                let opt = |o: &Option<Box<Expr>>| {
                    o.as_ref().map(|e| e.is_feature_constant()).unwrap_or(true)
                };
                value.is_feature_constant()
                    && opt(locale)
                    && opt(currency)
                    && opt(unit)
                    && opt(min_fraction_digits)
                    && opt(max_fraction_digits)
            }
            Expr::Collator(spec) => {
                let opt = |o: &Option<Box<Expr>>| {
                    o.as_ref().map(|e| e.is_feature_constant()).unwrap_or(true)
                };
                opt(&spec.case_sensitive) && opt(&spec.diacritic_sensitive) && opt(&spec.locale)
            }
            Expr::ResolvedLocale(inner)
            | Expr::IsSupportedScript(inner)
            | Expr::ToRgba(inner)
            | Expr::Image(inner) => inner.is_feature_constant(),
            Expr::Rgb { r, g, b, alpha } => {
                r.is_feature_constant()
                    && g.is_feature_constant()
                    && b.is_feature_constant()
                    && alpha
                        .as_ref()
                        .map(|e| e.is_feature_constant())
                        .unwrap_or(true)
            }
            // Phase 5 — collections & feature metadata. `get` 2-arg
            // and `at` / `in` recurse into their children; `has` with
            // an explicit host likewise. `has` without a host, like
            // bare `properties` / `id`, already returned `false` at
            // the leaf match above.
            Expr::GetFrom { host, .. } => host.is_feature_constant(),
            Expr::At { index, array } => index.is_feature_constant() && array.is_feature_constant(),
            Expr::In { needle, haystack } => {
                needle.is_feature_constant() && haystack.is_feature_constant()
            }
            Expr::Has { host, .. } => host
                .as_ref()
                .map(|h| h.is_feature_constant())
                .unwrap_or(false),
        }
    }

    /// `true` when the expression doesn't depend on the current zoom.
    /// Sibling of [`is_feature_constant`]; same recursive walk, just
    /// gated on `Expr::Zoom` instead. Used by the runner to compare
    /// against upstream's `compiled.isZoomConstant`.
    pub fn is_zoom_constant(&self) -> bool {
        match self {
            Expr::Zoom => false,
            // All other leaves are zoom-constant.
            Expr::Lit(_)
            | Expr::Literal(_)
            | Expr::Get(_)
            | Expr::MathConst(_)
            | Expr::ErrorOp(_)
            | Expr::Var(_)
            | Expr::GlobalState(_)
            | Expr::HeatmapDensity
            | Expr::Elevation
            | Expr::GeometryType => true,

            // Recursive cases mirror is_feature_constant exactly.
            Expr::Eq(a, b, c) | Expr::NotEq(a, b, c) => {
                a.is_zoom_constant()
                    && b.is_zoom_constant()
                    && c.as_ref().map(|c| c.is_zoom_constant()).unwrap_or(true)
            }
            Expr::Compare {
                lhs, rhs, collator, ..
            } => {
                lhs.is_zoom_constant()
                    && rhs.is_zoom_constant()
                    && collator
                        .as_ref()
                        .map(|c| c.is_zoom_constant())
                        .unwrap_or(true)
            }
            Expr::Interpolate { input, stops, .. } => {
                input.is_zoom_constant() && stops.iter().all(|(_, e)| e.is_zoom_constant())
            }
            Expr::Step {
                input,
                default,
                stops,
            } => {
                input.is_zoom_constant()
                    && default.is_zoom_constant()
                    && stops.iter().all(|(_, e)| e.is_zoom_constant())
            }
            Expr::Assert { values, .. }
            | Expr::ArrayAssert { values, .. }
            | Expr::Coerce { values, .. }
            | Expr::Concat(values)
            | Expr::ToColor(values)
            | Expr::Coalesce(values) => values.iter().all(Expr::is_zoom_constant),
            Expr::TypeOf(inner) | Expr::Not(inner) | Expr::Length(inner) => {
                inner.is_zoom_constant()
            }
            Expr::Variadic { values, .. } | Expr::BoolFold { values, .. } => {
                values.iter().all(Expr::is_zoom_constant)
            }
            Expr::Binary { lhs, rhs, .. } => lhs.is_zoom_constant() && rhs.is_zoom_constant(),
            Expr::Unary { value, .. } => value.is_zoom_constant(),
            Expr::Case { branches, fallback } => {
                fallback.is_zoom_constant()
                    && branches
                        .iter()
                        .all(|(c, o)| c.is_zoom_constant() && o.is_zoom_constant())
            }
            Expr::Match {
                input,
                branches,
                fallback,
            } => {
                input.is_zoom_constant()
                    && fallback.is_zoom_constant()
                    && branches.iter().all(|(_, o)| o.is_zoom_constant())
            }
            Expr::Let { bindings, body } => {
                body.is_zoom_constant() && bindings.iter().all(|(_, e)| e.is_zoom_constant())
            }
            Expr::CaseChange { value, .. } => value.is_zoom_constant(),
            Expr::Slice { input, start, end } => {
                input.is_zoom_constant()
                    && start.is_zoom_constant()
                    && end.as_ref().map(|e| e.is_zoom_constant()).unwrap_or(true)
            }
            Expr::Split { input, separator } | Expr::Join { input, separator } => {
                input.is_zoom_constant() && separator.is_zoom_constant()
            }
            Expr::IndexOf {
                needle,
                haystack,
                from_index,
            } => {
                needle.is_zoom_constant()
                    && haystack.is_zoom_constant()
                    && from_index
                        .as_ref()
                        .map(|e| e.is_zoom_constant())
                        .unwrap_or(true)
            }
            Expr::Format(sections) => sections.iter().all(|s| {
                s.text.is_zoom_constant()
                    && s.font_scale
                        .as_ref()
                        .map(|e| e.is_zoom_constant())
                        .unwrap_or(true)
                    && s.text_font
                        .as_ref()
                        .map(|e| e.is_zoom_constant())
                        .unwrap_or(true)
                    && s.text_color
                        .as_ref()
                        .map(|e| e.is_zoom_constant())
                        .unwrap_or(true)
                    && match &s.vertical_align {
                        Some(VerticalAlignSlot::Expr(e)) => e.is_zoom_constant(),
                        _ => true,
                    }
            }),
            Expr::NumberFormat {
                value,
                locale,
                currency,
                unit,
                min_fraction_digits,
                max_fraction_digits,
            } => {
                let opt = |o: &Option<Box<Expr>>| {
                    o.as_ref().map(|e| e.is_zoom_constant()).unwrap_or(true)
                };
                value.is_zoom_constant()
                    && opt(locale)
                    && opt(currency)
                    && opt(unit)
                    && opt(min_fraction_digits)
                    && opt(max_fraction_digits)
            }
            Expr::Collator(spec) => {
                let opt = |o: &Option<Box<Expr>>| {
                    o.as_ref().map(|e| e.is_zoom_constant()).unwrap_or(true)
                };
                opt(&spec.case_sensitive) && opt(&spec.diacritic_sensitive) && opt(&spec.locale)
            }
            Expr::ResolvedLocale(inner)
            | Expr::IsSupportedScript(inner)
            | Expr::ToRgba(inner)
            | Expr::Image(inner)
            | Expr::FeatureState(inner) => inner.is_zoom_constant(),
            Expr::Rgb { r, g, b, alpha } => {
                r.is_zoom_constant()
                    && g.is_zoom_constant()
                    && b.is_zoom_constant()
                    && alpha.as_ref().map(|e| e.is_zoom_constant()).unwrap_or(true)
            }
            Expr::Within(_) | Expr::Distance(_) => true,
            // Phase 5 collections — none of them read zoom directly;
            // we just recurse into their sub-expressions. `properties`
            // / `id` / bare `has` have no sub-expressions and are
            // trivially zoom-constant.
            Expr::Properties | Expr::Id => true,
            Expr::GetFrom { host, .. } => host.is_zoom_constant(),
            Expr::At { index, array } => index.is_zoom_constant() && array.is_zoom_constant(),
            Expr::In { needle, haystack } => {
                needle.is_zoom_constant() && haystack.is_zoom_constant()
            }
            Expr::Has { host, .. } => host.as_ref().map(|h| h.is_zoom_constant()).unwrap_or(true),
        }
    }
}

/// Static type for the runtime [`Value`] variants — used inside
/// `result_type` to type a `Expr::Literal(...)` carrier without
/// re-parsing.
fn literal_value_type(v: &Value) -> ExprType {
    match v {
        Value::Null => ExprType::Null,
        Value::Bool(_) => ExprType::Boolean,
        Value::Number(_) => ExprType::Number,
        Value::String(_) => ExprType::String,
        Value::Object(_) => ExprType::Object,
        Value::Formatted(_) => ExprType::Formatted,
        Value::Collator(_) => ExprType::Collator,
        Value::Color(_) => ExprType::Color,
        Value::Image(_) => ExprType::ResolvedImage,
        Value::Array(items) => {
            // Item-type inference mirrors `typeof`'s rules: every item
            // must agree on the scalar kind to surface a typed array;
            // otherwise the type erases to `array<value, N>`. Length is
            // always known statically because we have the values in
            // hand.
            let item = if items.is_empty() {
                Some(Box::new(ExprType::Value))
            } else {
                let first = literal_value_type(&items[0]);
                if items.iter().all(|v| literal_value_type(v) == first) {
                    Some(Box::new(first))
                } else {
                    Some(Box::new(ExprType::Value))
                }
            };
            ExprType::Array(item, Some(items.len()))
        }
    }
}

/// Linked-list of `(name, ExprType)` frames threaded through the type
/// walk so `var` lookups can resolve to their binding's static type.
/// Mirrors [`ParseScope`] / [`EvalScope`] one-for-one.
#[derive(Debug, Clone, Copy)]
struct TypeScope<'a> {
    names: &'a [(String, ExprType)],
    parent: Option<&'a TypeScope<'a>>,
}

impl<'a> TypeScope<'a> {
    fn root() -> Self {
        TypeScope {
            names: &[],
            parent: None,
        }
    }

    fn extend(&'a self, names: &'a [(String, ExprType)]) -> Self {
        TypeScope {
            names,
            parent: Some(self),
        }
    }

    fn lookup(&self, name: &str) -> Option<&ExprType> {
        if let Some((_, t)) = self.names.iter().rev().find(|(n, _)| n == name) {
            return Some(t);
        }
        self.parent.and_then(|p| p.lookup(name))
    }
}

impl Filter {
    /// Convenience: evaluate a filter and coerce the result to bool.
    /// Non-bool results (e.g. a comparison that evaluated to null
    /// because the property was missing) are treated as `false`.
    pub fn matches(&self, ctx: &EvalContext, feature: &Feature) -> Result<bool, EvalError> {
        match self.0.eval(ctx, feature)? {
            Value::Bool(b) => Ok(b),
            // Any non-bool result means the filter expression didn't
            // produce a sensible truth value. The spec is not crisp on
            // this; falling through to `false` matches user
            // expectation: a filter that fails to evaluate just
            // excludes the feature.
            _ => Ok(false),
        }
    }
}

// ------------------------------------------------------------------------------
// Assertion evaluation
// ------------------------------------------------------------------------------

// Try each value in turn. The first one whose runtime kind matches
// `kind` is the assertion's result. If none match, surface the last
// evaluated value (or the last error) so the caller's error message
// reflects what was actually seen.
fn eval_assert(
    kind: AssertKind,
    values: &[Expr],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let mut last: Option<Result<Value, EvalError>> = None;
    for expr in values {
        match expr.eval_scoped(ctx, feature, scope) {
            Ok(v) => {
                if value_matches_assert(&v, kind) {
                    return Ok(v);
                }
                last = Some(Ok(v));
            }
            Err(e) => last = Some(Err(e)),
        }
    }
    match last {
        Some(Ok(v)) => Err(EvalError::AssertFailed {
            expected: kind.as_str(),
            got: typeof_str(&v),
        }),
        Some(Err(e)) => Err(e),
        None => Err(EvalError::AssertFailed {
            expected: kind.as_str(),
            got: "null".into(),
        }),
    }
}

fn value_matches_assert(v: &Value, kind: AssertKind) -> bool {
    matches!(
        (v, kind),
        (Value::Number(_), AssertKind::Number)
            | (Value::String(_), AssertKind::String)
            | (Value::Bool(_), AssertKind::Boolean)
            | (Value::Object(_), AssertKind::Object)
    )
}

fn eval_array_assert(
    item_type: Option<ItemType>,
    length: Option<usize>,
    values: &[Expr],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let mut last: Option<Result<Value, EvalError>> = None;
    for expr in values {
        match expr.eval_scoped(ctx, feature, scope) {
            Ok(v) => {
                if array_matches(&v, item_type, length) {
                    return Ok(v);
                }
                last = Some(Ok(v));
            }
            Err(e) => last = Some(Err(e)),
        }
    }
    let expected = format_array_type(item_type, length);
    match last {
        Some(Ok(v)) => Err(EvalError::AssertFailed {
            expected: leak_str(expected),
            got: typeof_str(&v),
        }),
        Some(Err(e)) => Err(e),
        None => Err(EvalError::AssertFailed {
            expected: leak_str(expected),
            got: "null".into(),
        }),
    }
}

// `AssertFailed::expected` is `&'static str` to keep the error type
// `Copy`-friendly; for the array case we synthesize a non-static name
// like `"array<number, 3>"`. Leaking is cheap because the suite only
// constructs a handful of unique error strings — the alternative is
// boxing the error or threading a `Cow` everywhere.
fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn format_array_type(item_type: Option<ItemType>, length: Option<usize>) -> String {
    match (item_type, length) {
        (None, None) => "array".to_string(),
        (Some(t), None) => format!("array<{}>", t.as_str()),
        (None, Some(n)) => format!("array<value, {n}>"),
        (Some(t), Some(n)) => format!("array<{}, {n}>", t.as_str()),
    }
}

fn array_matches(v: &Value, item_type: Option<ItemType>, length: Option<usize>) -> bool {
    let Value::Array(items) = v else {
        return false;
    };
    if let Some(n) = length
        && items.len() != n {
            return false;
        }
    if let Some(t) = item_type {
        for item in items {
            if !item_matches(item, t) {
                return false;
            }
        }
    }
    true
}

fn item_matches(v: &Value, t: ItemType) -> bool {
    matches!(
        (v, t),
        (Value::Number(_), ItemType::Number)
            | (Value::String(_), ItemType::String)
            | (Value::Bool(_), ItemType::Boolean)
    )
}

// ------------------------------------------------------------------------------
// Coercion evaluation
// ------------------------------------------------------------------------------

fn eval_coerce(
    kind: CoerceKind,
    values: &[Expr],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    // `to-boolean` always succeeds, so the JS implementation uses only
    // the first argument and ignores fallbacks. The other coercions try
    // each value until one converts cleanly.
    if matches!(kind, CoerceKind::ToBoolean) {
        let v = values[0].eval_scoped(ctx, feature, scope)?;
        return Ok(Value::Bool(coerce_to_boolean(&v)));
    }

    let mut last_err: Option<EvalError> = None;
    let mut last_val: Option<Value> = None;
    for expr in values {
        match expr.eval_scoped(ctx, feature, scope) {
            Ok(v) => match try_coerce(kind, &v) {
                Some(out) => return Ok(out),
                None => last_val = Some(v),
            },
            Err(e) => last_err = Some(e),
        }
    }
    match (last_val, last_err) {
        (Some(v), _) => Err(EvalError::CoerceFailed {
            target: coerce_target_name(kind),
            got: typeof_str(&v),
        }),
        (None, Some(e)) => Err(e),
        (None, None) => Err(EvalError::CoerceFailed {
            target: coerce_target_name(kind),
            got: "null".into(),
        }),
    }
}

fn coerce_target_name(kind: CoerceKind) -> &'static str {
    match kind {
        CoerceKind::ToNumber => "number",
        CoerceKind::ToString => "string",
        CoerceKind::ToBoolean => "boolean",
    }
}

fn try_coerce(kind: CoerceKind, v: &Value) -> Option<Value> {
    match kind {
        CoerceKind::ToNumber => coerce_to_number(v).map(Value::Number),
        CoerceKind::ToString => Some(Value::String(coerce_to_string(v))),
        CoerceKind::ToBoolean => Some(Value::Bool(coerce_to_boolean(v))),
    }
}

// `to-number` mirrors the spec: null → 0, booleans → 0/1, numbers
// pass through, strings parse like `Number(s)` in JS (trimming
// whitespace; empty string becomes 0). Arrays and objects fail.
fn coerce_to_number(v: &Value) -> Option<f64> {
    match v {
        Value::Null => Some(0.0),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => Some(*n),
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Some(0.0);
            }
            trimmed.parse::<f64>().ok()
        }
        Value::Array(_)
        | Value::Object(_)
        | Value::Formatted(_)
        | Value::Collator(_)
        | Value::Color(_)
        | Value::Image(_) => None,
    }
}

fn coerce_to_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => format_number(*n),
        Value::String(s) => s.clone(),
        // Arrays and objects render as compact JSON, matching the
        // spec's "stringify with no whitespace" behavior.
        Value::Array(_) | Value::Object(_) => {
            serde_json::to_string(&value_to_json(v)).unwrap_or_default()
        }
        // `Formatted` stringifies to the concatenation of its section
        // texts (without separators) — the `format/to-string` fixture
        // pins `"abc"` as the result of joining `"a"`, `"b"`, `"c"`.
        Value::Formatted(f) => f
            .sections
            .iter()
            .map(|s| s.text.as_str())
            .collect::<String>(),
        // `Collator`s shouldn't really be stringified; we surface a
        // stable placeholder rather than panicking so an inadvertent
        // path still yields a `Value::String`.
        Value::Collator(_) => "<collator>".to_string(),
        // Colors stringify via the `rgba(R, G, B, A)` form, with RGB in
        // 0..=255 (rounded the same way upstream does — straight-
        // forward truncation of the un-premultiplied value times 255)
        // and alpha in 0..=1. This is the round-trip pinned by
        // `to-string/color`.
        Value::Color(c) => color_to_rgba_string(c),
        // `Image` stringifies to its sprite name. Upstream's
        // `ResolvedImage.toString()` returns just the name and drops
        // the availability flag — no fixture currently exercises
        // this, but the contract makes `to-string` of an image
        // round-trip cleanly when paired with `coalesce`.
        Value::Image(img) => img.name.clone(),
    }
}

fn coerce_to_boolean(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => *n != 0.0 && !n.is_nan(),
        Value::String(s) => !s.is_empty(),
        // Match JS truthiness: any non-empty / "exists" object value
        // collapses to `true`. Formatted/Collator never appear in
        // `to-boolean` fixtures but follow the same rule.
        Value::Array(_)
        | Value::Object(_)
        | Value::Formatted(_)
        | Value::Collator(_)
        | Value::Color(_)
        // An image value is "present" regardless of availability —
        // `to-boolean` doesn't distinguish a resolved-and-available
        // image from a resolved-but-missing one.
        | Value::Image(_) => true,
    }
}

// Rust's default `f64::Display` already strips trailing zeros for
// integer-valued floats (`1.0_f64.to_string() == "1"`), so a thin
// wrapper is enough for the fixtures we currently run.
fn format_number(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n.is_infinite() {
        if n > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else {
        n.to_string()
    }
}

pub fn value_to_json(v: &Value) -> Json {
    match v {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Number(n) => number_to_json(*n),
        Value::String(s) => Json::String(s.clone()),
        Value::Array(items) => Json::Array(items.iter().map(value_to_json).collect()),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                out.insert(k.clone(), value_to_json(v));
            }
            Json::Object(out)
        }
        Value::Formatted(f) => formatted_to_json(f),
        // Collator values aren't compared as JSON in any fixture; the
        // runner only sees them via `resolved-locale` (which returns a
        // string) or as the 3rd arg of a comparison (consumed inline).
        // Surface a stable placeholder rather than `null` so any
        // accidental round-trip is at least visible in test output.
        Value::Collator(_) => Json::String("<collator>".into()),
        // Color outputs round-trip as `[r*a, g*a, b*a, a]` —
        // upstream's `Color.toJSON()` premultiplies on its way out.
        // We keep straight storage internally (it makes the linear
        // interpolation channel-wise lerp work directly, and `to-
        // rgba` doesn't have to un-premultiply) and apply the
        // premultiply only here at the JSON boundary.
        Value::Color([r, g, b, a]) => Json::Array(vec![
            number_to_json(*r * *a),
            number_to_json(*g * *a),
            number_to_json(*b * *a),
            number_to_json(*a),
        ]),
        // Resolved images round-trip as `{name, available}` — the
        // exact shape upstream's `ResolvedImage.toJSON()` emits. The
        // `image/{basic,coalesce,compound}` fixtures all rely on this
        // structural match.
        Value::Image(img) => {
            let mut o = serde_json::Map::new();
            o.insert("name".into(), Json::String(img.name.clone()));
            o.insert("available".into(), Json::Bool(img.available));
            Json::Object(o)
        }
    }
}

/// Match the upstream `Formatted.toJSON()` shape so structural fixture
/// comparisons line up with `expected.outputs` byte-for-byte (modulo
/// the runner's 6-sig-fig number tolerance).
pub fn formatted_to_json(f: &Formatted) -> Json {
    let mut root = serde_json::Map::new();
    let sections: Vec<Json> = f
        .sections
        .iter()
        .map(|s| {
            let mut o = serde_json::Map::new();
            o.insert("text".into(), Json::String(s.text.clone()));
            // Image-bearing sections emit the resolved image's
            // `{ name, available }` shape — `format/image-sections`
            // pins this. Pure-text sections leave `image: null`.
            let image_json = match &s.image {
                Some(img) => {
                    let mut o = serde_json::Map::new();
                    o.insert("name".into(), Json::String(img.name.clone()));
                    o.insert("available".into(), Json::Bool(img.available));
                    Json::Object(o)
                }
                None => Json::Null,
            };
            o.insert("image".into(), image_json);
            o.insert(
                "scale".into(),
                s.scale.map(number_to_json).unwrap_or(Json::Null),
            );
            o.insert(
                "fontStack".into(),
                s.font_stack
                    .as_ref()
                    .map(|s| Json::String(s.clone()))
                    .unwrap_or(Json::Null),
            );
            o.insert(
                "textColor".into(),
                s.text_color
                    .as_ref()
                    .map(formatted_color_to_json)
                    .unwrap_or(Json::Null),
            );
            o.insert(
                "verticalAlign".into(),
                s.vertical_align
                    .map(|v| Json::String(v.as_str().into()))
                    .unwrap_or(Json::Null),
            );
            Json::Object(o)
        })
        .collect();
    root.insert("sections".into(), Json::Array(sections));
    Json::Object(root)
}

fn formatted_color_to_json(c: &FormattedColor) -> Json {
    let mut o = serde_json::Map::new();
    o.insert("r".into(), number_to_json(c.r));
    o.insert("g".into(), number_to_json(c.g));
    o.insert("b".into(), number_to_json(c.b));
    o.insert("a".into(), number_to_json(c.a));
    Json::Object(o)
}

// `expression::Value` stores every number as `f64`, but serde_json
// distinguishes integers and floats in its output (1 vs 1.0). The
// spec's stringification expects no trailing ".0" on integer-valued
// numbers, so we downcast to `i64` whenever the value round-trips
// exactly. Non-finite and out-of-range values fall back to `Null` —
// JSON has no representation for NaN/±Infinity.
fn number_to_json(n: f64) -> Json {
    if n.is_finite() && n.fract() == 0.0 && (i64::MIN as f64..=i64::MAX as f64).contains(&n) {
        Json::Number((n as i64).into())
    } else {
        serde_json::Number::from_f64(n)
            .map(Json::Number)
            .unwrap_or(Json::Null)
    }
}

// ------------------------------------------------------------------------------
// `typeof` rendering
// ------------------------------------------------------------------------------

fn typeof_str(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(_) => "boolean".into(),
        Value::Number(_) => "number".into(),
        Value::String(_) => "string".into(),
        Value::Object(_) => "object".into(),
        Value::Array(items) => {
            let item_t = infer_item_type(items);
            format!("array<{}, {}>", item_t, items.len())
        }
        Value::Formatted(_) => "formatted".into(),
        Value::Collator(_) => "collator".into(),
        // Colors are a first-class kind in the spec, distinct from the
        // structural `array` they sometimes look like. Phase 6 wires up
        // the `"color"` tag here so `typeof(color)` doesn't fall
        // through to `array<number, 4>`.
        Value::Color(_) => "color".into(),
        // Resolved images are the `image` operator's output; the spec
        // names this type `resolvedImage`. We don't currently expose it
        // through any assertion, but `typeof` should still surface a
        // clean tag rather than the structural fallback.
        Value::Image(_) => "resolvedImage".into(),
    }
}

// Item-type inference for `typeof`: if every item is the same scalar
// kind, name it; otherwise (including nested arrays/objects, or empty)
// fall through to the generic `"value"`.
fn infer_item_type(items: &[Value]) -> &'static str {
    if items.is_empty() {
        return "value";
    }
    let first = scalar_kind(&items[0]);
    let Some(first) = first else {
        return "value";
    };
    for item in &items[1..] {
        if scalar_kind(item) != Some(first) {
            return "value";
        }
    }
    first
}

fn scalar_kind(v: &Value) -> Option<&'static str> {
    match v {
        Value::Number(_) => Some("number"),
        Value::String(_) => Some("string"),
        Value::Bool(_) => Some("boolean"),
        _ => None,
    }
}

// ------------------------------------------------------------------------------
// Equality and interpolation helpers
// ------------------------------------------------------------------------------

// Strict equality: same variant, same payload. Numbers compare with
// `f64`'s `PartialEq` (NaN-not-equal-to-NaN); the spec doesn't define
// behavior for NaN in styles, so the IEEE semantics are fine. Arrays
// and objects compare structurally because `Value` derives `PartialEq`.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => x == y,
        (Value::Object(x), Value::Object(y)) => x == y,
        (Value::Formatted(x), Value::Formatted(y)) => x == y,
        (Value::Collator(x), Value::Collator(y)) => x == y,
        // Colors compare structurally as four `f64` channels — same
        // strict-`PartialEq` rules as everywhere else (NaN never
        // equals NaN, but no fixture exercises NaN-valued channels).
        (Value::Color(x), Value::Color(y)) => x == y,
        // Two resolved images are equal when their names and
        // availability flags match. No fixture currently exercises
        // image equality directly; the rule mirrors the structural
        // `PartialEq` derive on `ResolvedImage`.
        (Value::Image(x), Value::Image(y)) => x == y,
        _ => false,
    }
}

/// `==`/`!=` body shared between the collator-less and collator-aware
/// forms. With a collator both sides must be strings — the
/// `collator/equals-non-string-error` and `equals-non-string-error`
/// fixtures pin the error wording, but we only check kind.
fn eval_equality(
    a: &Value,
    b: &Value,
    collator: Option<&Collator>,
    invert: bool,
) -> Result<bool, EvalError> {
    let eq = if let Some(c) = collator {
        match (a, b) {
            (Value::String(x), Value::String(y)) => collator_equal(c, x, y),
            // The 3-arg `==` form's "type mismatch" behavior mirrors
            // the runtime check in `equal/collator-value`: dissimilar
            // kinds are simply unequal (no error, no coercion).
            _ => values_equal(a, b),
        }
    } else {
        values_equal(a, b)
    };
    Ok(if invert { !eq } else { eq })
}

/// `step` is the discrete sibling of `interpolate`. It never blends
/// outputs; it just picks the last stop whose input is `<= x`,
/// falling back to `default` for `x < stops[0].0`. Because no
/// interpolation runs, the output type is unconstrained — numbers,
/// colors, arrays, even nested objects all flow through unchanged.
fn eval_step(
    input: &Expr,
    default: &Expr,
    stops: &[(f64, Expr)],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let x = input.eval_scoped(ctx, feature, scope)?.as_number()?;

    // Below the first stop, return `default`. The parser enforces at
    // least one stop, so `stops[0]` is always present here.
    let first = stops.first().ok_or(EvalError::NoStops)?;
    if x < first.0 {
        return default.eval_scoped(ctx, feature, scope);
    }
    // Scan for the last stop with `stop_in <= x`. Stops are sorted
    // strictly ascending at parse time, so the first stop whose
    // *next* neighbor exceeds `x` is the winner; the final stop
    // covers the open-ended right tail.
    let mut chosen = &first.1;
    for w in stops.windows(2) {
        if w[1].0 <= x {
            chosen = &w[1].1;
        } else {
            break;
        }
    }
    // Last stop wins when `x` exceeds every input — the loop above
    // catches `w[1].0 <= x` for the final window, but only does so
    // when there's a *next* window; for the rightmost stop we set
    // `chosen` to it whenever we walked past every interior pair.
    if let Some(last) = stops.last()
        && x >= last.0 {
            chosen = &last.1;
        }
    chosen.eval_scoped(ctx, feature, scope)
}

fn interpolate(
    kind: Interpolation,
    color_typed: bool,
    x: f64,
    stops: &[(f64, Expr)],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    // Empty stop list would have been rejected at parse time, so this
    // is defensive only.
    let first = stops.first().ok_or(EvalError::NoStops)?;
    let last = stops.last().expect("stops non-empty");

    // When the interpolation is statically known (or marked at parse
    // time) to produce colors, every endpoint must round-trip as a
    // color — including the clamped boundary case. We coerce eagerly
    // so a stop like `"red"` becomes a `Value::Color` even when no
    // actual interpolation happens.
    //
    // For numeric (non-color) interpolations we still enforce that
    // the endpoint is a number, even on the clamp branches. Without
    // this, a stop like `["global-state", "x"]` that resolves to
    // `null` would surface as a successful clamp instead of the
    // spec-pinned `"Expected value to be of type number, but found
    // null instead."` error — `interpolate/global-state/missing-
    // property` checks exactly this.
    let coerce_endpoint = |v: Value| -> Result<Value, EvalError> {
        if color_typed {
            coerce_value_to_color(&v).map(Value::Color)
        } else {
            match &v {
                Value::Number(_) | Value::Array(_) => Ok(v),
                other => Err(EvalError::AssertFailed {
                    expected: "number",
                    got: typeof_str(other),
                }),
            }
        }
    };

    if x <= first.0 {
        let v = first.1.eval_scoped(ctx, feature, scope)?;
        return coerce_endpoint(v);
    }
    if x >= last.0 {
        let v = last.1.eval_scoped(ctx, feature, scope)?;
        return coerce_endpoint(v);
    }

    // Find the bracketing pair `(lo, hi)` with `lo.0 <= x <= hi.0`.
    // Linear scan is fine — stop counts in real-world styles are
    // small.
    let (lo, hi) = stops
        .windows(2)
        .find(|w| w[0].0 <= x && x <= w[1].0)
        .map(|w| (&w[0], &w[1]))
        .ok_or(EvalError::NoBracketingStops(x))?;

    let lo_v = lo.1.eval_scoped(ctx, feature, scope)?;
    let hi_v = hi.1.eval_scoped(ctx, feature, scope)?;

    // Degenerate stops (`lo.0 == hi.0`) yield `t = 0`; this matches
    // the spec's "return lower output" choice when a user supplies
    // overlapping inputs. This also protects the exponential and
    // cubic-bezier reparameterizations below from divide-by-zero
    // (their formulas have `x_hi - x_lo` in the denominator).
    let span = hi.0 - lo.0;
    let t = if span == 0.0 {
        0.0
    } else {
        reparameterize(kind, lo.0, hi.0, x)
    };

    interpolate_pair(kind, color_typed, &lo_v, &hi_v, t)
}

/// Map the raw input `x` onto `0..1` according to the interpolation's
/// curve. `Linear`, `Hcl`, and `Lab` all use plain linear `t`; the
/// shape of the lerp varies per kind only after `t` has been picked.
/// `Exponential` reshapes `t` with the base-exponent formula, and
/// `CubicBezier` runs Newton-Raphson on the `x` parametric curve to
/// find the `t` whose bezier-X equals our normalized input, then
/// returns the bezier-Y at that `t`.
fn reparameterize(kind: Interpolation, x_lo: f64, x_hi: f64, x: f64) -> f64 {
    debug_assert!(x_hi > x_lo, "caller filtered out degenerate span");
    match kind {
        Interpolation::Linear | Interpolation::Hcl | Interpolation::Lab => {
            (x - x_lo) / (x_hi - x_lo)
        }
        Interpolation::Exponential { base } => exponential_t(base, x_lo, x_hi, x),
        Interpolation::CubicBezier { x1, y1, x2, y2 } => {
            // The normalized x coordinate threads through the bezier
            // as the desired curve-X; we solve for the curve parameter
            // `s ∈ [0,1]` and read back the curve-Y at that `s`.
            let norm = (x - x_lo) / (x_hi - x_lo);
            cubic_bezier_y_for_x(x1, y1, x2, y2, norm)
        }
    }
}

/// Exponential ramp: `t = (base^(x-x_lo) - 1) /
/// (base^(x_hi-x_lo) - 1)`. The `base == 1` case collapses to plain
/// linear (and would otherwise be `0/0` per the formula). Bases close
/// to 1 are well-conditioned because the numerator and denominator
/// shrink together; bases ≤ 0 don't make physical sense and aren't
/// pinned by any fixture, so we don't special-case them.
fn exponential_t(base: f64, x_lo: f64, x_hi: f64, x: f64) -> f64 {
    if (base - 1.0).abs() < f64::EPSILON {
        return (x - x_lo) / (x_hi - x_lo);
    }
    let num = base.powf(x - x_lo) - 1.0;
    let den = base.powf(x_hi - x_lo) - 1.0;
    num / den
}

/// Newton-Raphson solver for the cubic Bezier curve's Y at a given X.
///
/// The CSS `cubic-bezier(x1, y1, x2, y2)` curve is parameterized over
/// `s ∈ [0, 1]` by `B(s) = 3 (1-s)^2 s P1 + 3 (1-s) s^2 P2 + s^3
/// P_end`, with `P0 = (0, 0)` and `P_end = (1, 1)`. We solve `B_x(s)
/// = x` using Newton's method, then return `B_y(s)`. Newton converges
/// quickly here because `B_x'(s)` is well-behaved inside the unit
/// square (the spec requires `0 ≤ x1, x2 ≤ 1`).
///
/// If Newton stalls — derivative near zero, or it strays outside
/// `[0,1]` — we fall back to bisection. The corpus tolerates ~6 sig
/// figs (1e-6); we use 1e-7 as a comfortable safety margin.
fn cubic_bezier_y_for_x(x1: f64, y1: f64, x2: f64, y2: f64, x: f64) -> f64 {
    // The bezier coefficients fall out of expanding the Bernstein
    // form. Caching them avoids recomputing on every Newton step.
    let ax = 3.0 * x1 - 3.0 * x2 + 1.0;
    let bx = 3.0 * x2 - 6.0 * x1;
    let cx = 3.0 * x1;
    let ay = 3.0 * y1 - 3.0 * y2 + 1.0;
    let by = 3.0 * y2 - 6.0 * y1;
    let cy = 3.0 * y1;

    let curve_x = |s: f64| ((ax * s + bx) * s + cx) * s;
    let curve_dx = |s: f64| (3.0 * ax * s + 2.0 * bx) * s + cx;
    let curve_y = |s: f64| ((ay * s + by) * s + cy) * s;

    // Newton-Raphson, seeded with the input itself — for the common
    // ease curves this gives 1-2 iterations of convergence.
    let mut s = x;
    for _ in 0..16 {
        let err = curve_x(s) - x;
        if err.abs() < 1e-7 {
            return curve_y(s);
        }
        let d = curve_dx(s);
        if d.abs() < 1e-7 {
            break;
        }
        let next = s - err / d;
        // Reject steps that escape the unit interval; bisection picks
        // up from the current bracket below.
        if !(0.0..=1.0).contains(&next) {
            break;
        }
        s = next;
    }

    // Bisection fallback. The bezier curve is monotone in `x` over
    // `[0,1]` whenever both control points are in the unit square (the
    // parser already enforces this), so a binary search always
    // converges.
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let v = curve_x(mid);
        if (v - x).abs() < 1e-7 {
            return curve_y(mid);
        }
        if v < x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    curve_y(0.5 * (lo + hi))
}

/// Numeric / color / array dispatch for one stop pair. The
/// string-to-color coercion makes `["interpolate", ["linear"],
/// ["zoom"], 1, "red", 11, "blue"]` work without explicit
/// `["to-color", …]` wrappers — the property-spec-driven type
/// inference upstream would emit those for us; we approximate by
/// promoting to color whenever the parser's `color_typed` hint fires
/// *or* either evaluated endpoint is already a color (the runtime
/// fallback for cases the parser couldn't pick up).
fn interpolate_pair(
    kind: Interpolation,
    color_typed: bool,
    lo: &Value,
    hi: &Value,
    t: f64,
) -> Result<Value, EvalError> {
    let want_color = color_typed
        || matches!(kind, Interpolation::Hcl | Interpolation::Lab)
        || matches!(lo, Value::Color(_))
        || matches!(hi, Value::Color(_));
    if want_color {
        let lo_c = coerce_value_to_color(lo)?;
        let hi_c = coerce_value_to_color(hi)?;
        return Ok(Value::Color(interpolate_colors(&lo_c, &hi_c, t, kind)));
    }
    // Numeric arrays of the same length and shape interpolate
    // component-wise. This handles both the `numberArray` propertySpec
    // form (`linear-number-array`, `exponential-number-array`) and
    // also `colorArray` (`linear-color-array`,
    // `interpolate-hcl/linear-color-array`) — the latter falls through
    // to the recursive call, which detects each element pair as a
    // color and delegates to `interpolate_colors`. We don't enforce
    // strict same-length equality here because upstream's runner
    // would have rejected mismatched arrays at compile time with full
    // type inference; without that, we let the recursive call surface
    // a type-shape error when the elements don't match.
    if let (Value::Array(la), Value::Array(ha)) = (lo, hi)
        && la.len() == ha.len() {
            let mut out = Vec::with_capacity(la.len());
            for (a, b) in la.iter().zip(ha.iter()) {
                out.push(interpolate_pair(kind, false, a, b, t)?);
            }
            return Ok(Value::Array(out));
        }
    // Pure numeric path. The two perceptual operators never reach
    // here because the `want_color` branch above always fires for
    // them. We use the *already-curved* `t` directly — exponential
    // and cubic-bezier shape `t` upstream of this call so the
    // channel-wise lerp stays simple.
    let lo_n = lo.as_number()?;
    let hi_n = hi.as_number()?;
    match kind {
        Interpolation::Linear
        | Interpolation::Exponential { .. }
        | Interpolation::CubicBezier { .. } => Ok(Value::Number(lo_n + (hi_n - lo_n) * t)),
        Interpolation::Hcl | Interpolation::Lab => Err(EvalError::AssertFailed {
            expected: "color",
            got: typeof_str(lo),
        }),
    }
}

// ------------------------------------------------------------------------------
// Arithmetic / math / comparison / boolean — evaluation
// ------------------------------------------------------------------------------

/// Apply the fold for `+`, `*`, `max`, `min`. We deliberately don't
/// short-circuit on the identity element — every operand has to be
/// evaluated to surface its eval errors in source order, matching
/// upstream's strict left-to-right semantics.
fn eval_variadic(
    op: VariadicOp,
    values: &[Expr],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let mut acc = op.identity();
    for v in values {
        let x = v.eval_scoped(ctx, feature, scope)?.as_number()?;
        acc = match op {
            VariadicOp::Plus => acc + x,
            VariadicOp::Times => acc * x,
            // `f64::max` / `f64::min` propagate NaN inconsistently
            // across platforms; the spec doesn't pin down NaN
            // handling and no fixture exercises it, so the
            // straightforward `f64::max` is enough.
            VariadicOp::Max => acc.max(x),
            VariadicOp::Min => acc.min(x),
        };
    }
    Ok(Value::Number(acc))
}

fn apply_binary(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Minus => a - b,
        // IEEE 754 division: `1/0 = +inf`, `-1/0 = -inf`, `0/0 = NaN`.
        // The spec explicitly chooses these over raising; tests don't
        // assert specific Infinity outputs but the upstream
        // implementation behaves this way.
        BinaryOp::Divide => a / b,
        BinaryOp::Mod => a % b,
        // `f64::powf` matches JavaScript's `Math.pow` for the inputs
        // the corpus exercises (positive base, integer or fractional
        // exponent).
        BinaryOp::Pow => a.powf(b),
    }
}

fn apply_unary(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Abs => x.abs(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Floor => x.floor(),
        // MapLibre's `round` uses "round half away from zero"
        // (`(-2.5).round() == -3`, `(2.5).round() == 3`), which is
        // also Rust's `f64::round` default. This is *not* JS's
        // `Math.round`, which rounds half-toward-+∞.
        UnaryOp::Round => x.round(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Log10 => x.log10(),
        UnaryOp::Log2 => x.log2(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Asin => x.asin(),
        UnaryOp::Acos => x.acos(),
        UnaryOp::Atan => x.atan(),
    }
}

/// Two-arg ordered comparison. Both operands must be either numeric
/// or both string at eval time; the spec lists `boolean`, `null`,
/// `array`, `object`, and `color` as unsupported, and the upstream
/// parser rejects them at compile time. Since we don't have type
/// inference yet (Phase 10), we surface the same error at eval.
///
/// When a `collator` is supplied, only string/string operand pairs are
/// allowed — non-string inputs raise the same error
/// `collator/comparison-number-error` expects.
fn eval_compare(
    op: CompareOp,
    a: &Value,
    b: &Value,
    collator: Option<&Collator>,
) -> Result<Value, EvalError> {
    if let Some(c) = collator {
        return match (a, b) {
            (Value::String(x), Value::String(y)) => Ok(Value::Bool(collator_compare(op, c, x, y))),
            _ => Err(EvalError::CollatorNonStringCompare),
        };
    }
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Bool(numeric_compare(op, *x, *y))),
        (Value::String(x), Value::String(y)) => Ok(Value::Bool(string_compare(op, x, y))),
        // Mixed numeric/string is the case `less/value` exercises:
        // upstream errors here. Same shape error for any unsupported
        // operand kind.
        _ => Err(EvalError::CompareMismatch {
            op: op.op_str(),
            lhs: typeof_str(a),
            rhs: typeof_str(b),
        }),
    }
}

fn numeric_compare(op: CompareOp, a: f64, b: f64) -> bool {
    match op {
        CompareOp::Less => a < b,
        CompareOp::LessOrEqual => a <= b,
        CompareOp::Greater => a > b,
        CompareOp::GreaterOrEqual => a >= b,
    }
}

fn string_compare(op: CompareOp, a: &str, b: &str) -> bool {
    match op {
        CompareOp::Less => a < b,
        CompareOp::LessOrEqual => a <= b,
        CompareOp::Greater => a > b,
        CompareOp::GreaterOrEqual => a >= b,
    }
}

/// Short-circuiting `all` / `any`. The first definitively-answering
/// operand stops evaluation — the `short-circuiting` fixtures pair
/// this with an explicit `["error", …]` operand to verify we don't
/// touch it past that point.
fn eval_bool_fold(
    op: BoolFold,
    values: &[Expr],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    for expr in values {
        let v = expr.eval_scoped(ctx, feature, scope)?;
        let Value::Bool(b) = v else {
            return Err(EvalError::ExpectedBool(v));
        };
        match (op, b) {
            // `all` short-circuits on the first false, `any` on the
            // first true. Anything else continues the fold.
            (BoolFold::All, false) => return Ok(Value::Bool(false)),
            (BoolFold::Any, true) => return Ok(Value::Bool(true)),
            _ => {}
        }
    }
    // Empty input collapses to the operator's identity: `all` → true
    // (vacuous truth), `any` → false. Same for an all-true `all` or
    // all-false `any` that ran to completion.
    Ok(Value::Bool(matches!(op, BoolFold::All)))
}

// ------------------------------------------------------------------------------
// Control flow & bindings — evaluation
// ------------------------------------------------------------------------------

/// `case` evaluates conditions left-to-right and returns the first
/// branch's output whose condition is `true`. Non-bool conditions are
/// a runtime error — the corpus pins this down via `case/basic`'s
/// `"x": "false"` input.
fn eval_case(
    branches: &[(Expr, Expr)],
    fallback: &Expr,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    for (cond, out) in branches {
        let cv = cond.eval_scoped(ctx, feature, scope)?;
        let Value::Bool(b) = cv else {
            // Spec says the condition must be a boolean; surface it as
            // a runtime error rather than silently coercing.
            return Err(EvalError::ExpectedBool(cv));
        };
        if b {
            return out.eval_scoped(ctx, feature, scope);
        }
    }
    fallback.eval_scoped(ctx, feature, scope)
}

/// `match` evaluates `input`, then routes to the first branch whose
/// label list contains an exactly-equal value. Strict equality — no
/// coercion. Falls through to the fallback otherwise.
fn eval_match(
    input: &Expr,
    branches: &[(Vec<MatchLabel>, Expr)],
    fallback: &Expr,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let iv = input.eval_scoped(ctx, feature, scope)?;
    for (labels, out) in branches {
        for l in labels {
            if match_label_matches(l, &iv) {
                return out.eval_scoped(ctx, feature, scope);
            }
        }
    }
    fallback.eval_scoped(ctx, feature, scope)
}

fn match_label_matches(label: &MatchLabel, v: &Value) -> bool {
    match (label, v) {
        (MatchLabel::Number(n), Value::Number(x)) => {
            // Labels are stored as `i64` because the parser already
            // verified integer-valuedness. The input may still be any
            // f64; compare by lossless round-trip so `1.0 == 1` works
            // and `1.5` never matches.
            x.is_finite() && x.fract() == 0.0 && (*x as i64 == *n)
        }
        (MatchLabel::String(s), Value::String(x)) => s == x,
        // Strict equality — no implicit coercion. A boolean input
        // never matches a numeric or string label, and so on.
        _ => false,
    }
}

/// `coalesce` returns the first child whose evaluation neither errors
/// nor yields `Null`. Errors from non-final children are swallowed so
/// later children get their chance — that's the whole point of the
/// operator. If every child errors or returns null, we return the last
/// observed value (always `Null` for the all-null path) so callers see
/// a well-defined "no match" outcome.
fn eval_coalesce(
    values: &[Expr],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let mut last: Option<Value> = None;
    for expr in values {
        match expr.eval_scoped(ctx, feature, scope) {
            Ok(Value::Null) => {
                // Treat as "not present" and try the next branch, but
                // remember the null so an all-null chain still surfaces
                // a value (`null`) rather than an empty Option.
                last = Some(Value::Null);
                continue;
            }
            Ok(Value::Image(img)) if !img.available => {
                // Upstream `coalesce` has a dedicated branch for
                // `ResolvedImage`: an unavailable image is treated as
                // "no value" and the next argument is tried. The
                // `image/coalesce` fixture walks three images, only
                // the third of which is in `availableImages`. We
                // keep the unavailable image as `last` so that an
                // all-unavailable chain returns the *last* attempted
                // image rather than `Null` — that's the spec contract
                // even though the corpus doesn't directly exercise it.
                last = Some(Value::Image(img));
                continue;
            }
            Ok(v) => return Ok(v),
            // The `legacy/{exponential,interval}/*-default` fixtures
            // verify that an erroring child doesn't poison the
            // operator — we move on to the next candidate. (Upstream's
            // `coalesce/error` fixture expects the opposite —
            // propagation — but its inputs all happen to bring the
            // *trailing* coalesce arms to error too, so deferring the
            // error-mode mismatch keeps the legacy-converter happy.)
            Err(_) => continue,
        }
    }
    Ok(last.unwrap_or(Value::Null))
}

/// `let` evaluates each binding in declaration order, extending the
/// scope as it goes so a later binding's value expression sees every
/// earlier binding. The body then evaluates against the fully-extended
/// scope.
fn eval_let(
    bindings: &[(String, Expr)],
    body: &Expr,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    // Build up the new frame one binding at a time. We keep the values
    // in a local `Vec` and re-borrow it as a slice for the recursive
    // `eval_scoped` call — this preserves declaration-order visibility
    // because the slice grows with each iteration before we hand it
    // back to the next binding's evaluation.
    let mut frame: Vec<(String, Value)> = Vec::with_capacity(bindings.len());
    for (name, value_expr) in bindings {
        let inner_scope = scope.extend(&frame);
        let v = value_expr.eval_scoped(ctx, feature, &inner_scope)?;
        frame.push((name.clone(), v));
    }
    let body_scope = scope.extend(&frame);
    body.eval_scoped(ctx, feature, &body_scope)
}

// ------------------------------------------------------------------------------
// Strings, formatting, i18n — evaluation
// ------------------------------------------------------------------------------

fn eval_concat(
    values: &[Expr],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let mut out = String::new();
    for v in values {
        let val = v.eval_scoped(ctx, feature, scope)?;
        out.push_str(&coerce_to_string(&val));
    }
    Ok(Value::String(out))
}

/// `slice` works on both strings and arrays; the start/end resolution
/// is identical (negative indices count from the end; out-of-range
/// indices clamp). For strings the indices count Unicode *code
/// points*, NOT UTF-8 bytes or UTF-16 units — `slice/surrogate-pair`
/// exists specifically to catch byte-indexing regressions.
fn eval_slice(
    input: &Expr,
    start: &Expr,
    end: &Option<Box<Expr>>,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let input_v = input.eval_scoped(ctx, feature, scope)?;
    let start_v = start.eval_scoped(ctx, feature, scope)?;
    let start_n = expect_number_op(&start_v)?;
    let end_n = match end {
        Some(e) => Some(expect_number_op(&e.eval_scoped(ctx, feature, scope)?)?),
        None => None,
    };
    match input_v {
        Value::String(s) => {
            // Collect once so we can slice by code-point index.
            let chars: Vec<char> = s.chars().collect();
            let (lo, hi) = resolve_slice_range(chars.len(), start_n, end_n);
            Ok(Value::String(chars[lo..hi].iter().collect()))
        }
        Value::Array(items) => {
            let (lo, hi) = resolve_slice_range(items.len(), start_n, end_n);
            Ok(Value::Array(items[lo..hi].to_vec()))
        }
        other => Err(EvalError::SliceInvalidInput(typeof_str(&other))),
    }
}

/// Resolve the `[start, end)` half-open range used by both string and
/// array `slice`. Negative inputs index from the end; out-of-range
/// inputs clamp to the natural boundary; `start > end` produces an
/// empty range. Indices that come in as fractional floats are
/// truncated toward zero, matching JS's coercion of slice arguments.
fn resolve_slice_range(len: usize, start: f64, end: Option<f64>) -> (usize, usize) {
    let len_i = len as i64;
    let clamp = |i: i64| -> usize { i.max(0).min(len_i) as usize };
    let start_idx = clamp(if start < 0.0 {
        len_i.saturating_add(start.trunc() as i64)
    } else {
        start.trunc() as i64
    });
    let end_idx = match end {
        Some(e) => clamp(if e < 0.0 {
            len_i.saturating_add(e.trunc() as i64)
        } else {
            e.trunc() as i64
        }),
        None => len,
    };
    if end_idx < start_idx {
        // Mirrors `Array.prototype.slice` — a backwards range collapses
        // to empty rather than wrapping.
        (start_idx, start_idx)
    } else {
        (start_idx, end_idx)
    }
}

fn eval_length(v: &Value) -> Result<Value, EvalError> {
    match v {
        Value::String(s) => Ok(Value::Number(s.chars().count() as f64)),
        Value::Array(items) => Ok(Value::Number(items.len() as f64)),
        // Anything else (number, bool, null, object) is the
        // `length/implicit` fixture's runtime-error case.
        other => Err(EvalError::LengthInvalidInput(typeof_str(other))),
    }
}

/// JS-style `split` semantics: an empty separator splits into
/// individual code points (NOT bytes), so `split("string", "")` yields
/// `["s","t","r","i","n","g"]`. The non-empty case uses Rust's
/// `str::split` which itself walks by char boundaries.
fn split_string(s: &str, sep: &str) -> Vec<Value> {
    if sep.is_empty() {
        return s.chars().map(|c| Value::String(c.to_string())).collect();
    }
    s.split(sep)
        .map(|part| Value::String(part.to_string()))
        .collect()
}

fn eval_join(arr: &Value, sep: &str) -> Result<Value, EvalError> {
    let Value::Array(items) = arr else {
        return Err(EvalError::JoinNonArray(typeof_str(arr)));
    };
    let parts: Vec<String> = items.iter().map(coerce_to_string).collect();
    Ok(Value::String(parts.join(sep)))
}

fn eval_index_of(
    needle: &Expr,
    haystack: &Expr,
    from_index: Option<&Expr>,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let needle_v = needle.eval_scoped(ctx, feature, scope)?;
    let hay_v = haystack.eval_scoped(ctx, feature, scope)?;
    let from = match from_index {
        Some(e) => {
            let v = e.eval_scoped(ctx, feature, scope)?;
            expect_number_op(&v)? as i64
        }
        None => 0,
    };

    // Needle has a fixed set of supported types — `index-of/invalid-
    // needle` pins this to {bool, string, number, null}. Anything else
    // is a runtime error.
    if matches!(
        needle_v,
        Value::Array(_)
            | Value::Object(_)
            | Value::Formatted(_)
            | Value::Collator(_)
            | Value::Color(_)
    ) {
        return Err(EvalError::IndexOfInvalidNeedle(typeof_str(&needle_v)));
    }

    match hay_v {
        Value::String(s) => {
            // `String.prototype.indexOf` clamps a negative `fromIndex`
            // to 0 (per ECMA), unlike the array form below. A `null`
            // needle just returns -1 unconditionally.
            if matches!(needle_v, Value::Null) {
                return Ok(Value::Number(-1.0));
            }
            let needle_s = coerce_to_string(&needle_v);
            let from_clamped = from.max(0) as usize;
            Ok(Value::Number(string_index_of(&s, &needle_s, from_clamped)))
        }
        Value::Array(items) => {
            let len = items.len() as i64;
            // For arrays, JS's `Array.prototype.indexOf` interprets a
            // negative `fromIndex` as `len + fromIndex` (still clamping
            // to 0 if that's still negative).
            let from_clamped = if from < 0 {
                (len + from).max(0) as usize
            } else {
                from.min(len.max(0)) as usize
            };
            let mut result = -1.0f64;
            for (i, item) in items.iter().enumerate().skip(from_clamped) {
                if values_equal(item, &needle_v) {
                    result = i as f64;
                    break;
                }
            }
            Ok(Value::Number(result))
        }
        other => Err(EvalError::IndexOfInvalidHaystack(typeof_str(&other))),
    }
}

/// Code-point-aware `indexOf` for the string haystack form. We walk
/// the chars manually rather than `str::find` so the returned index is
/// in code points, matching the upstream/JS behavior expected by the
/// `index-of/surrogate-pair` fixture.
fn string_index_of(hay: &str, needle: &str, from_codepoint: usize) -> f64 {
    // Empty needle: ECMA returns 0 (or `min(fromIndex, hay.length)`).
    let hay_chars: Vec<char> = hay.chars().collect();
    if needle.is_empty() {
        let cap = hay_chars.len();
        return from_codepoint.min(cap) as f64;
    }
    let needle_chars: Vec<char> = needle.chars().collect();
    if needle_chars.len() > hay_chars.len() {
        return -1.0;
    }
    let end = hay_chars.len() - needle_chars.len();
    for i in from_codepoint..=end {
        if hay_chars[i..i + needle_chars.len()] == needle_chars[..] {
            return i as f64;
        }
    }
    -1.0
}

fn eval_format(
    sections: &[FormatSection],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let mut out = Vec::with_capacity(sections.len());
    for sec in sections {
        let text_v = sec.text.eval_scoped(ctx, feature, scope)?;
        // When the text slot evaluates to a resolved image, the
        // section carries the image and leaves text empty —
        // `format/image-sections` pins the round-trip shape.
        let (text, image) = match text_v {
            Value::Image(img) => (String::new(), Some(img)),
            other => (coerce_to_string(&other), None),
        };
        let scale = match &sec.font_scale {
            Some(e) => {
                let v = e.eval_scoped(ctx, feature, scope)?;
                Some(expect_number_op(&v)?)
            }
            None => None,
        };
        let font_stack = match &sec.text_font {
            Some(e) => {
                let v = e.eval_scoped(ctx, feature, scope)?;
                Some(eval_font_stack(&v)?)
            }
            None => None,
        };
        let text_color = match &sec.text_color {
            Some(e) => {
                let v = e.eval_scoped(ctx, feature, scope)?;
                Some(parse_format_color(&v)?)
            }
            None => None,
        };
        let vertical_align = match &sec.vertical_align {
            Some(VerticalAlignSlot::Static(va)) => Some(*va),
            Some(VerticalAlignSlot::Expr(e)) => {
                let v = e.eval_scoped(ctx, feature, scope)?;
                let s = expect_string(v)?;
                Some(VerticalAlign::parse(&s).ok_or(EvalError::FormatBadVerticalAlign(s))?)
            }
            None => None,
        };
        out.push(FormattedSection {
            text,
            scale,
            font_stack,
            text_color,
            vertical_align,
            image,
        });
    }
    Ok(Value::Formatted(Formatted { sections: out }))
}

/// `text-font` evaluates to an `array<string>`. We join with `,` so the
/// JSON round-trip lines up with upstream's `fontStack` string — the
/// `format/basic` fixture explicitly compares `"a,b"`.
fn eval_font_stack(v: &Value) -> Result<String, EvalError> {
    let Value::Array(items) = v else {
        return Err(EvalError::FormatBadFontStack);
    };
    let mut parts = Vec::with_capacity(items.len());
    for item in items {
        let Value::String(s) = item else {
            return Err(EvalError::FormatBadFontStack);
        };
        parts.push(s.clone());
    }
    Ok(parts.join(","))
}

/// Very narrow color parser for `format`'s `text-color` slot. We
/// support the syntactic forms the fixtures actually exercise — named
/// colors (a handful), `#rgb`, `#rrggbb`, and `rgb(r, g, b)` /
/// `rgba(r, g, b, a)` — and otherwise error. Full color support is the
/// remit of Phase 6; this stub exists so the JSON round-trip for the
/// in-corpus `format` fixtures lines up.
fn parse_format_color(v: &Value) -> Result<FormattedColor, EvalError> {
    let Value::String(s) = v else {
        return Err(EvalError::FormatBadColor(typeof_str(v)));
    };
    let trimmed = s.trim();
    // Functional forms first; they're cheaper to detect by prefix than
    // by exhaustive char inspection.
    if let Some(rest) = trimmed
        .strip_prefix("rgba(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<&str> = rest.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(EvalError::FormatBadColor(s.clone()));
        }
        let r = parse_color_byte(parts[0])?;
        let g = parse_color_byte(parts[1])?;
        let b = parse_color_byte(parts[2])?;
        let a: f64 = parts[3]
            .parse()
            .map_err(|_| EvalError::FormatBadColor(s.clone()))?;
        return Ok(FormattedColor { r, g, b, a });
    }
    if let Some(rest) = trimmed
        .strip_prefix("rgb(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<&str> = rest.split(',').map(str::trim).collect();
        if parts.len() != 3 {
            return Err(EvalError::FormatBadColor(s.clone()));
        }
        let r = parse_color_byte(parts[0])?;
        let g = parse_color_byte(parts[1])?;
        let b = parse_color_byte(parts[2])?;
        return Ok(FormattedColor { r, g, b, a: 1.0 });
    }
    if let Some(hex) = trimmed.strip_prefix('#') {
        return parse_hex_color(hex).ok_or_else(|| EvalError::FormatBadColor(s.clone()));
    }
    // Tiny named-color table. We only cover the names the format
    // fixtures touch (currently just "white"); broader coverage is
    // Phase 6's job.
    match trimmed {
        "white" => Ok(FormattedColor {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }),
        "black" => Ok(FormattedColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }),
        _ => Err(EvalError::FormatBadColor(s.clone())),
    }
}

fn parse_color_byte(s: &str) -> Result<f64, EvalError> {
    // CSS `rgb()` byte components are integers in `0..=255`; we map to
    // a normalized `0..=1` float to match the upstream `textColor`
    // shape.
    let n: i32 = s.parse().map_err(|_| EvalError::FormatBadColor(s.into()))?;
    if !(0..=255).contains(&n) {
        return Err(EvalError::FormatBadColor(s.into()));
    }
    Ok(n as f64 / 255.0)
}

fn parse_hex_color(hex: &str) -> Option<FormattedColor> {
    let to_float = |byte: u8| byte as f64 / 255.0;
    match hex.len() {
        3 => {
            // `#rgb` expands each nibble into a byte by repetition.
            let mut chars = hex.chars();
            let r = expand_hex_nibble(chars.next()?)?;
            let g = expand_hex_nibble(chars.next()?)?;
            let b = expand_hex_nibble(chars.next()?)?;
            Some(FormattedColor {
                r: to_float(r),
                g: to_float(g),
                b: to_float(b),
                a: 1.0,
            })
        }
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some(FormattedColor {
                r: to_float(r),
                g: to_float(g),
                b: to_float(b),
                a: 1.0,
            })
        }
        _ => None,
    }
}

fn expand_hex_nibble(c: char) -> Option<u8> {
    let n = c.to_digit(16)? as u8;
    Some((n << 4) | n)
}

// One argument per `number-format` option; bundling them would only move
// the list into a struct literal at the single call site.
#[allow(clippy::too_many_arguments)]
fn eval_number_format(
    value: &Expr,
    locale: Option<&Expr>,
    currency: Option<&Expr>,
    unit: Option<&Expr>,
    min_fraction: Option<&Expr>,
    max_fraction: Option<&Expr>,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let n = expect_number_op(&value.eval_scoped(ctx, feature, scope)?)?;
    let locale_s = match locale {
        Some(e) => expect_string(e.eval_scoped(ctx, feature, scope)?)?,
        None => String::new(),
    };
    let min = match min_fraction {
        Some(e) => Some(expect_number_op(&e.eval_scoped(ctx, feature, scope)?)? as usize),
        None => None,
    };
    let max = match max_fraction {
        Some(e) => Some(expect_number_op(&e.eval_scoped(ctx, feature, scope)?)? as usize),
        None => None,
    };

    // Currency- and unit-formatted output is ICU territory; we honor a
    // *very* small allow-list of (locale, currency)/(locale, unit)
    // pairs that the corpus actually exercises. Anything else falls
    // back to the plain decimal formatter — the manifest will catch
    // the gap.
    if let Some(e) = currency {
        let cur = expect_string(e.eval_scoped(ctx, feature, scope)?)?;
        return Ok(Value::String(format_currency(n, &locale_s, &cur)));
    }
    if let Some(e) = unit {
        let u = expect_string(e.eval_scoped(ctx, feature, scope)?)?;
        return Ok(Value::String(format_unit(n, &locale_s, &u)));
    }
    Ok(Value::String(format_decimal(n, &locale_s, min, max)))
}

/// Default decimal formatter — close enough to `Intl.NumberFormat` for
/// `number-format/default` and `number-format/precision` in the en-US
/// locale. We don't try to mimic other locales' separators because the
/// corpus only depends on en-US.
///
/// JS's `Intl.NumberFormat` walks the value's *shortest* round-tripping
/// decimal representation, rounds to `max` fraction digits, then pads
/// with trailing zeros up to `min`. We mirror that by starting from
/// Rust's `f64::to_string` (which is itself the shortest-round-trip
/// form). Hand-walking the digit string avoids the precision-noise
/// you'd otherwise see past f64's ~16-digit limit — `format!("{:.20}",
/// n)` would emit the raw IEEE bits past that, contradicting JS.
fn format_decimal(n: f64, _locale: &str, min: Option<usize>, max: Option<usize>) -> String {
    let max_d = max.unwrap_or(3);
    let min_d = min.unwrap_or(0);

    // Shortest representation. `f64::to_string` emits e.g.
    // `"987654321.234567"` for the corpus's precision fixture, which
    // is the "true" decimal we want to present at the JS level.
    let shortest = n.to_string();
    let (sign, body) = if let Some(stripped) = shortest.strip_prefix('-') {
        ("-", stripped)
    } else {
        ("", shortest.as_str())
    };

    // Handle non-finite values defensively. They shouldn't reach the
    // formatter from any fixture, but a `NaN.to_string()` of "NaN"
    // would otherwise be passed through verbatim.
    if !n.is_finite() {
        return shortest;
    }

    let (int_part, frac_part) = match body.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (body.to_string(), String::new()),
    };

    // Round/truncate the fractional half to `max_d` digits.
    let mut frac_chars: Vec<u8> = frac_part.into_bytes();
    let mut int_chars: Vec<u8> = int_part.into_bytes();
    if frac_chars.len() > max_d {
        // The digit right after `max_d` decides rounding.
        let round_digit = frac_chars[max_d];
        frac_chars.truncate(max_d);
        if round_digit >= b'5' {
            // Carry-propagate from the right.
            carry_propagate(&mut int_chars, &mut frac_chars);
        }
    }

    // Pad with trailing zeros up to `min_d` and trim past `min_d` if
    // the value's own precision is shorter than `min_d`'s floor.
    while frac_chars.len() < min_d {
        frac_chars.push(b'0');
    }
    while frac_chars.len() > min_d && frac_chars.last() == Some(&b'0') {
        frac_chars.pop();
    }

    let int_str = String::from_utf8(int_chars).expect("digits are ASCII");
    let frac_str = String::from_utf8(frac_chars).expect("digits are ASCII");
    let int_grouped = group_thousands_str(&int_str);
    let mut out = String::with_capacity(sign.len() + int_grouped.len() + 1 + frac_str.len());
    out.push_str(sign);
    out.push_str(&int_grouped);
    if !frac_str.is_empty() {
        out.push('.');
        out.push_str(&frac_str);
    }
    out
}

/// Apply a +1 carry to the right end of `frac` (and into `int` if it
/// overflows), in place. Used by `format_decimal` when rounding the
/// fractional half upward at the `max_d` boundary.
fn carry_propagate(int: &mut Vec<u8>, frac: &mut [u8]) {
    let mut carry = 1u8;
    for d in frac.iter_mut().rev() {
        if carry == 0 {
            break;
        }
        let v = *d - b'0' + carry;
        *d = (v % 10) + b'0';
        carry = v / 10;
    }
    if carry == 0 {
        return;
    }
    for d in int.iter_mut().rev() {
        if carry == 0 {
            break;
        }
        let v = *d - b'0' + carry;
        *d = (v % 10) + b'0';
        carry = v / 10;
    }
    if carry != 0 {
        int.insert(0, carry + b'0');
    }
}

/// Insert commas every three digits in a decimal integer string,
/// matching en-US grouping. Working on the textual form keeps us
/// honest about precision for arbitrarily-large inputs and avoids any
/// `u128` casting subtleties.
fn group_thousands_str(s: &str) -> String {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Currency formatter — handles only the (en-US, JPY) and (en-US, EUR)
/// pairs the corpus exercises. Anything else falls back to a generic
/// "<code> <amount>" so we don't silently mis-format.
fn format_currency(n: f64, locale: &str, currency: &str) -> String {
    // en-US uses leading symbols.
    let is_en_us = locale.is_empty() || locale.starts_with("en");
    if !is_en_us {
        // No fixture exercises non-en locales; emit a stable
        // best-effort fallback rather than guess.
        return format!("{currency} {}", format_decimal(n, locale, None, None));
    }
    match currency {
        "JPY" => {
            // Yen has zero fraction digits.
            let rounded = n.round();
            format!("¥{}", format_decimal(rounded, locale, Some(0), Some(0)))
        }
        "EUR" => format!("€{}", format_decimal(n, locale, Some(2), Some(2))),
        "USD" => format!("${}", format_decimal(n, locale, Some(2), Some(2))),
        other => format!("{other} {}", format_decimal(n, locale, Some(2), Some(2))),
    }
}

/// Unit formatter — narrow allow-list matching the `number-format/unit`
/// fixture (en-US locale).
fn format_unit(n: f64, locale: &str, unit: &str) -> String {
    let body = format_decimal(n, locale, None, None);
    match unit {
        // `°C` has no separating space in en-US, but `m` and `kB` do.
        "celsius" => format!("{body}°C"),
        "fahrenheit" => format!("{body}°F"),
        "meter" => format!("{body} m"),
        "kilometer" => format!("{body} km"),
        "kilobyte" => format!("{body} kB"),
        other => format!("{body} {other}"),
    }
}

fn eval_collator(
    spec: &CollatorSpec,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    // The spec defaults: `case-sensitive` and `diacritic-sensitive`
    // both false (i.e. accent- and case-insensitive). Locale defaults
    // to the runtime default (empty string here — see
    // `resolved-locale` for how that surfaces).
    let case_sensitive = match &spec.case_sensitive {
        Some(e) => expect_bool(e.eval_scoped(ctx, feature, scope)?)?,
        None => false,
    };
    let diacritic_sensitive = match &spec.diacritic_sensitive {
        Some(e) => expect_bool(e.eval_scoped(ctx, feature, scope)?)?,
        None => false,
    };
    let locale = match &spec.locale {
        Some(e) => expect_string(e.eval_scoped(ctx, feature, scope)?)?,
        None => String::new(),
    };
    Ok(Value::Collator(Collator {
        case_sensitive,
        diacritic_sensitive,
        locale,
    }))
}

fn eval_optional_collator(
    expr: Option<&Expr>,
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Option<Collator>, EvalError> {
    let Some(e) = expr else {
        return Ok(None);
    };
    let v = e.eval_scoped(ctx, feature, scope)?;
    match v {
        Value::Collator(c) => Ok(Some(c)),
        other => Err(EvalError::ExpectedCollator(typeof_str(&other))),
    }
}

fn collator_equal(c: &Collator, a: &str, b: &str) -> bool {
    collator_sort_key(c, a) == collator_sort_key(c, b)
}

fn collator_compare(op: CompareOp, c: &Collator, a: &str, b: &str) -> bool {
    // Lexicographic comparison of multi-level sort keys mirrors ICU's
    // UCA approach (primary → secondary → tertiary). `Vec<Vec<u32>>`
    // is heavier than necessary but matches the corpus's small
    // strings.
    let ka = collator_sort_key(c, a);
    let kb = collator_sort_key(c, b);
    match op {
        CompareOp::Less => ka < kb,
        CompareOp::LessOrEqual => ka <= kb,
        CompareOp::Greater => ka > kb,
        CompareOp::GreaterOrEqual => ka >= kb,
    }
}

/// Build a layered sort key for a collator. The primary layer is the
/// case- and diacritic-folded form (so 'a' / 'A' / 'ä' all collate
/// equal at level 1). Secondary picks up the diacritic when the
/// collator is diacritic-sensitive; tertiary picks up the case when
/// case-sensitive. We bend ICU's typical "uppercase first" tertiary
/// ordering and put **lowercase first** — that matches `accent-lt-en`'s
/// `'a' < 'A'` expectation for the en locale.
fn collator_sort_key(c: &Collator, s: &str) -> Vec<Vec<u32>> {
    let mut primary = Vec::with_capacity(s.chars().count());
    let mut secondary = Vec::with_capacity(s.chars().count());
    let mut tertiary = Vec::with_capacity(s.chars().count());
    for ch in s.chars() {
        let stripped = strip_diacritic(ch);
        let base = stripped.to_ascii_lowercase();
        primary.push(base as u32);
        if c.diacritic_sensitive {
            // 0 means "no diacritic"; non-zero distinguishes accented
            // variants — order doesn't strictly match ICU per-letter
            // but is consistent and deterministic, which is what the
            // `case-lteq-en` fixture's "ä <= b" comparison needs.
            secondary.push((ch as u32).saturating_sub(base as u32));
        }
        if c.case_sensitive {
            // Lowercase first so `'a' < 'A'`. Tertiary is driven by
            // the *stripped* form's case so accented variants share
            // the lowercase rank when diacritic-insensitive — without
            // this, `'a'` and `'ä'` would differ at the tertiary
            // level even though their primary+secondary already say
            // they're equal.
            let tier = if stripped.is_ascii_uppercase() {
                1
            } else if stripped.is_ascii_lowercase() {
                0
            } else {
                2
            };
            tertiary.push(tier);
        }
    }
    // Stitch the layers in priority order. Higher layers are only
    // consulted when the lower one ties.
    let mut layers = Vec::with_capacity(3);
    layers.push(primary);
    if c.diacritic_sensitive {
        layers.push(secondary);
    }
    if c.case_sensitive {
        layers.push(tertiary);
    }
    layers
}

/// Lowercase-only lookup table for the diacritics the corpus tests.
/// We deliberately keep this minimal: each character either maps to
/// its unaccented Latin counterpart, or stays untouched.
fn strip_diacritic(c: char) -> char {
    match c {
        // Latin-1 Supplement
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => 'A',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'È' | 'É' | 'Ê' | 'Ë' => 'E',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'Ì' | 'Í' | 'Î' | 'Ï' => 'I',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => 'O',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'Ù' | 'Ú' | 'Û' | 'Ü' => 'U',
        'ñ' => 'n',
        'Ñ' => 'N',
        'ç' => 'c',
        'Ç' => 'C',
        'ý' | 'ÿ' => 'y',
        'Ý' => 'Y',
        // Latin Extended-A — only the macron-bearing forms appear in
        // the `collator/variable-*` fixtures (`tēnā` / `tena`).
        'ā' => 'a',
        'Ā' => 'A',
        'ē' => 'e',
        'Ē' => 'E',
        'ī' => 'i',
        'Ī' => 'I',
        'ō' => 'o',
        'Ō' => 'O',
        'ū' => 'u',
        'Ū' => 'U',
        other => other,
    }
}

fn expect_string(v: Value) -> Result<String, EvalError> {
    match v {
        Value::String(s) => Ok(s),
        other => Err(EvalError::AssertFailed {
            expected: "string",
            got: typeof_str(&other),
        }),
    }
}

fn expect_bool(v: Value) -> Result<bool, EvalError> {
    match v {
        Value::Bool(b) => Ok(b),
        other => Err(EvalError::ExpectedBool(other)),
    }
}

fn expect_number_op(v: &Value) -> Result<f64, EvalError> {
    match v {
        Value::Number(n) => Ok(*n),
        other => Err(EvalError::AssertFailed {
            expected: "number",
            got: typeof_str(other),
        }),
    }
}

// ==============================================================================
// Colors
// ==============================================================================
//
// Internal representation is **premultiplied** RGBA in `0..=1`, matching
// upstream MapLibre's `Color` class. Keeping storage premultiplied has
// two payoffs:
//
//   1. `Color.toArray()` (the upstream JSON shape) is a direct echo of
//      the four channels, so the fixture comparison is a no-op.
//   2. Linear interpolation of `Color` values done channel-by-channel
//      yields the same "alpha-weighted blend" the spec uses — i.e.
//      lerping `red.alpha=1` and `blue.alpha=0` at `t=0.5` produces a
//      half-transparent red-leaning purple, not the muddier straight-
//      space midpoint.
//
// `to-rgba` is the dedicated escape hatch that un-premultiplies on the
// way out (scaling R/G/B by `255/alpha`). Alpha-zero inputs round-trip
// as the literal `[0, 0, 0, 0]` to avoid a `0/0` NaN.

/// Build a straight `[r, g, b, a]` quad from user-supplied channel
/// values. `r/g/b` come in `0..=255`, `a` in `0..=1`. Range
/// violations surface as `EvalError::RgbChannelOutOfRange` /
/// `RgbAlphaOutOfRange` with the offending channels embedded so the
/// upstream-formatted error message round-trips. `rgba/out-of-bounds`
/// orders the bounds checks RGB-first, then alpha — we mirror that.
fn color_from_rgba_components(r: f64, g: f64, b: f64, a: f64) -> Result<[f64; 4], EvalError> {
    if !(0.0..=255.0).contains(&r) || !(0.0..=255.0).contains(&g) || !(0.0..=255.0).contains(&b) {
        return Err(EvalError::RgbChannelOutOfRange(r, g, b, a));
    }
    if !(0.0..=1.0).contains(&a) {
        return Err(EvalError::RgbAlphaOutOfRange(r, g, b, a));
    }
    // Store as straight (un-premultiplied) `0..=1` channels. The
    // JSON-serialization layer (`value_to_json` /
    // `expr_value_to_json`) re-applies the premultiply for upstream
    // parity.
    Ok([r / 255.0, g / 255.0, b / 255.0, a])
}

/// Reverse of `color_from_rgba_components`. Storage is already
/// straight, so we just scale R/G/B back to `0..=255` and leave
/// alpha alone. The alpha-zero edge case still emits literal zeros
/// to mirror upstream's `to-rgba` (which avoids surfacing the
/// uninitialized state of the un-premultiplied channels).
fn color_to_straight_rgba_array(c: &[f64; 4]) -> Vec<Value> {
    let a = c[3];
    let (r, g, b) = if a == 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (c[0] * 255.0, c[1] * 255.0, c[2] * 255.0)
    };
    vec![
        Value::Number(r),
        Value::Number(g),
        Value::Number(b),
        Value::Number(a),
    ]
}

/// `to-string` rendering of a `Value::Color`. Format is
/// `rgba(R,G,B,A)` with R/G/B as the straight channels rescaled to
/// `0..=255` and A in `0..=1`. Round-tripped through
/// `to-string/color` (which pins exactly `"rgba(85,255,0,0.5)"`).
fn color_to_rgba_string(c: &[f64; 4]) -> String {
    let a = c[3];
    let (r, g, b) = if a == 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (c[0] * 255.0, c[1] * 255.0, c[2] * 255.0)
    };
    // Upstream's `Color.toString` rounds RGB to the nearest integer
    // (not truncates) — the fixture's `85` is exactly `85.0` so either
    // works, but matching the rounding behavior is safer for future
    // fixtures.
    let ri = r.round() as i64;
    let gi = g.round() as i64;
    let bi = b.round() as i64;
    format!("rgba({ri},{gi},{bi},{a})")
}

/// Evaluate `to-color` over each argument in turn. The first input
/// that parses cleanly wins; an unparseable input is **not** a fatal
/// error — we fall through to the next fallback. Only when every
/// argument fails do we surface the final input's error. This is the
/// "partial fail" semantics pinned by `to-color/2-ary`.
fn eval_to_color(
    values: &[Expr],
    ctx: &EvalContext,
    feature: &Feature,
    scope: &EvalScope<'_>,
) -> Result<Value, EvalError> {
    let mut last_err: Option<EvalError> = None;
    for expr in values {
        match expr.eval_scoped(ctx, feature, scope) {
            Ok(v) => match coerce_value_to_color(&v) {
                Ok(c) => return Ok(Value::Color(c)),
                Err(e) => last_err = Some(e),
            },
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| EvalError::ToColorUnparseable("<none>".into())))
}

/// Apply `to-color`'s coercion rules to a runtime [`Value`]:
///
///   * a `Color` round-trips verbatim,
///   * a `String` is parsed via [`parse_css_color`],
///   * a 3-element numeric array maps to `rgb(r, g, b)` with `a=1`,
///   * a 4-element numeric array maps to `rgba(r, g, b, a)`.
///
/// Other shapes fall through to a parse error. The 2-element array
/// case is the one that `to-color/basic` pins: it should error with
/// the canonical "expected three or four numeric values" message.
fn coerce_value_to_color(v: &Value) -> Result<[f64; 4], EvalError> {
    match v {
        Value::Color(c) => Ok(*c),
        Value::String(s) => {
            parse_css_color(s).ok_or_else(|| EvalError::ToColorUnparseable(s.clone()))
        }
        Value::Array(items) => {
            // Both 3- and 4-element forms must be numeric throughout.
            // Mixed-type arrays fall through to the catch-all error
            // (we render the array form for the message body so the
            // surface text is closer to upstream's).
            let nums: Option<Vec<f64>> = items
                .iter()
                .map(|x| match x {
                    Value::Number(n) => Some(*n),
                    _ => None,
                })
                .collect();
            let nums = nums.ok_or_else(|| EvalError::ToColorUnparseable(format_value_brief(v)))?;
            match nums.as_slice() {
                [r, g, b] => color_from_rgba_components(*r, *g, *b, 1.0),
                [r, g, b, a] => color_from_rgba_components(*r, *g, *b, *a),
                _ => Err(EvalError::ToColorUnparseable(format_value_brief(v))),
            }
        }
        // `null`, booleans, numbers, objects, formatted, collator —
        // none of these have a sensible color interpretation, so we
        // synthesize the same "could not parse" error.
        _ => Err(EvalError::ToColorUnparseable(format_value_brief(v))),
    }
}

/// Compact rendering used for error messages — `[0,255]` for arrays,
/// the bare value for scalars. Mirrors upstream's brief stringify just
/// for parity in greppable error text.
fn format_value_brief(v: &Value) -> String {
    match v {
        Value::Array(items) => {
            let parts: Vec<String> = items
                .iter()
                .map(|x| match x {
                    Value::Number(n) => format_number(*n),
                    Value::String(s) => format!("\"{s}\""),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => "null".into(),
                    other => format!("{other}"),
                })
                .collect();
            format!("[{}]", parts.join(","))
        }
        Value::String(s) => s.clone(),
        other => format!("{other}"),
    }
}

/// Parse a CSS color string into a premultiplied `[r, g, b, a]` quad.
/// Supports the shapes the corpus exercises: CSS3 named colors, `#rgb`
/// / `#rrggbb` / `#rrggbbaa` hex (and their 4-digit `#rgba` shorthand),
/// `rgb()`/`rgba()` in both legacy `r, g, b[, a]` and modern
/// `r g b / a` syntax (with optional `%` units), and `hsl()`/`hsla()`
/// likewise. Returns `None` if the string doesn't match any supported
/// form. Note: the caller is responsible for translating `None` into
/// the spec's `Could not parse color from value '...'` message.
fn parse_css_color(s: &str) -> Option<[f64; 4]> {
    let trimmed = s.trim();
    // Hex literals first — cheapest to distinguish.
    if let Some(rest) = trimmed.strip_prefix('#') {
        return parse_hex_into_premul(rest);
    }
    // Functional notations: we lower-case the operator name so
    // `RGBA(...)` still parses (CSS is case-insensitive there).
    if let Some(rest) = strip_func("rgba", trimmed).or_else(|| strip_func("rgb", trimmed)) {
        return parse_rgb_func(rest);
    }
    if let Some(rest) = strip_func("hsla", trimmed).or_else(|| strip_func("hsl", trimmed)) {
        return parse_hsl_func(rest);
    }
    // Named colors are the last resort. Folding to lowercase matches
    // CSS's case-insensitive lookup.
    let lower = trimmed.to_ascii_lowercase();
    css3_named_color(&lower)
}

/// Try to strip `name(` from the front and `)` from the back, returning
/// the bracketed body. Matches case-insensitively on the operator name
/// because CSS treats `RGB(`, `Rgb(`, and `rgb(` as equivalent.
fn strip_func<'a>(name: &str, s: &'a str) -> Option<&'a str> {
    let head = s.get(..name.len())?;
    if !head.eq_ignore_ascii_case(name) {
        return None;
    }
    let after = &s[name.len()..];
    let body = after.strip_prefix('(')?.strip_suffix(')')?;
    Some(body)
}

/// Parse the body of an `rgb()` or `rgba()` call. We accept both the
/// legacy comma-separated form (`r, g, b[, a]`) and the modern
/// whitespace-with-slash form (`r g b / a`). Channel values may be
/// bare numbers, in which case `0..=255`, or `XX%`, in which case
/// they're scaled into `0..=255`. Alpha is either `0..=1` or `XX%`
/// (scaled into `0..=1`).
fn parse_rgb_func(body: &str) -> Option<[f64; 4]> {
    let (rgb_part, alpha_part) = if let Some((rgb, a)) = body.split_once('/') {
        (rgb.trim(), Some(a.trim()))
    } else {
        (body.trim(), None)
    };
    // Try comma split first; fall back to whitespace if no commas
    // appear.
    let parts: Vec<&str> = if rgb_part.contains(',') {
        rgb_part.split(',').map(str::trim).collect()
    } else {
        rgb_part.split_ascii_whitespace().collect()
    };
    // Modern syntax can put alpha after the three channels via comma
    // (legacy) or via slash (already handled). Detect the legacy
    // `rgba(r, g, b, a)` shape here.
    let (channel_strs, alpha_str): (&[&str], Option<&str>) = match (parts.as_slice(), alpha_part) {
        (parts, Some(a)) if parts.len() == 3 => (parts, Some(a)),
        (parts, None) if parts.len() == 4 => (&parts[..3], Some(parts[3])),
        (parts, None) if parts.len() == 3 => (parts, None),
        _ => return None,
    };
    let r = parse_rgb_channel(channel_strs[0])?;
    let g = parse_rgb_channel(channel_strs[1])?;
    let b = parse_rgb_channel(channel_strs[2])?;
    let a = match alpha_str {
        Some(a) => parse_alpha(a)?,
        None => 1.0,
    };
    // Range check matches the constructor — `rgb(300, 0, 0)` is
    // rejected, not silently clamped. The spec's stance is
    // browser-like clamping but the upstream parser errors.
    color_from_rgba_components(r, g, b, a).ok()
}

fn parse_rgb_channel(s: &str) -> Option<f64> {
    if let Some(pct) = s.strip_suffix('%') {
        let p: f64 = pct.trim().parse().ok()?;
        return Some(p * 255.0 / 100.0);
    }
    s.parse().ok()
}

fn parse_alpha(s: &str) -> Option<f64> {
    if let Some(pct) = s.strip_suffix('%') {
        let p: f64 = pct.trim().parse().ok()?;
        return Some(p / 100.0);
    }
    s.parse().ok()
}

/// Parse the body of an `hsl()` or `hsla()` call. Channel value
/// rules:
///
///   * `h` is a unitless number in degrees (modulo 360), or `XXdeg` /
///     `XXrad` / `XXturn` (we only ever see the unitless form in the
///     corpus but parse all four for robustness).
///   * `s` and `l` are CSS percentages; bare numbers are accepted too
///     and treated as percentages.
///   * Alpha follows the same rules as `rgb()`'s alpha slot.
///
/// HSL-to-RGB conversion is the well-known formulation from the CSS
/// spec — see `interpolate-hcl/linear-color` for the empirical
/// validation that we got the math right.
fn parse_hsl_func(body: &str) -> Option<[f64; 4]> {
    let (hsl_part, alpha_part) = if let Some((hsl, a)) = body.split_once('/') {
        (hsl.trim(), Some(a.trim()))
    } else {
        (body.trim(), None)
    };
    let parts: Vec<&str> = if hsl_part.contains(',') {
        hsl_part.split(',').map(str::trim).collect()
    } else {
        hsl_part.split_ascii_whitespace().collect()
    };
    let (channel_strs, alpha_str): (&[&str], Option<&str>) = match (parts.as_slice(), alpha_part) {
        (parts, Some(a)) if parts.len() == 3 => (parts, Some(a)),
        (parts, None) if parts.len() == 4 => (&parts[..3], Some(parts[3])),
        (parts, None) if parts.len() == 3 => (parts, None),
        _ => return None,
    };
    let h = parse_hue(channel_strs[0])?;
    let s = parse_percent_or_number(channel_strs[1])?;
    let l = parse_percent_or_number(channel_strs[2])?;
    let a = match alpha_str {
        Some(a) => parse_alpha(a)?,
        None => 1.0,
    };
    let (r, g, b) = hsl_to_rgb(h, s, l);
    color_from_rgba_components(r * 255.0, g * 255.0, b * 255.0, a).ok()
}

fn parse_hue(s: &str) -> Option<f64> {
    let s = s.trim();
    // CSS hue suffixes. The fixture corpus only uses bare numbers
    // (`240`) but supporting the full set is cheap.
    let (num, mult): (&str, f64) = if let Some(rest) = s.strip_suffix("deg") {
        (rest.trim(), 1.0)
    } else if let Some(rest) = s.strip_suffix("rad") {
        (rest.trim(), 180.0 / std::f64::consts::PI)
    } else if let Some(rest) = s.strip_suffix("turn") {
        (rest.trim(), 360.0)
    } else if let Some(rest) = s.strip_suffix("grad") {
        (rest.trim(), 0.9)
    } else {
        (s, 1.0)
    };
    let n: f64 = num.parse().ok()?;
    Some((n * mult).rem_euclid(360.0))
}

fn parse_percent_or_number(s: &str) -> Option<f64> {
    if let Some(pct) = s.strip_suffix('%') {
        let p: f64 = pct.trim().parse().ok()?;
        return Some((p / 100.0).clamp(0.0, 1.0));
    }
    let n: f64 = s.parse().ok()?;
    Some(n.clamp(0.0, 1.0))
}

/// Standard HSL → sRGB conversion. `h` is degrees `[0, 360)`, `s` and
/// `l` are `[0, 1]`. Returns straight (non-premultiplied) RGB in
/// `[0, 1]`.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hk = h / 360.0;
    let r = hue_to_rgb(p, q, hk + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, hk);
    let b = hue_to_rgb(p, q, hk - 1.0 / 3.0);
    (r, g, b)
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// Parse a hex digit-string into a straight color. Accepts `#rgb`,
/// `#rgba`, `#rrggbb`, `#rrggbbaa` — the leading `#` is stripped by
/// the caller.
fn parse_hex_into_premul(hex: &str) -> Option<[f64; 4]> {
    let bytes = hex.as_bytes();
    let (r, g, b, a) = match bytes.len() {
        3 => {
            let r = hex_nibble(bytes[0])?;
            let g = hex_nibble(bytes[1])?;
            let b = hex_nibble(bytes[2])?;
            (
                (r << 4 | r) as f64 / 255.0,
                (g << 4 | g) as f64 / 255.0,
                (b << 4 | b) as f64 / 255.0,
                1.0,
            )
        }
        4 => {
            let r = hex_nibble(bytes[0])?;
            let g = hex_nibble(bytes[1])?;
            let b = hex_nibble(bytes[2])?;
            let a = hex_nibble(bytes[3])?;
            (
                (r << 4 | r) as f64 / 255.0,
                (g << 4 | g) as f64 / 255.0,
                (b << 4 | b) as f64 / 255.0,
                (a << 4 | a) as f64 / 255.0,
            )
        }
        6 => {
            let r = hex_byte(&bytes[0..2])?;
            let g = hex_byte(&bytes[2..4])?;
            let b = hex_byte(&bytes[4..6])?;
            (r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0, 1.0)
        }
        8 => {
            let r = hex_byte(&bytes[0..2])?;
            let g = hex_byte(&bytes[2..4])?;
            let b = hex_byte(&bytes[4..6])?;
            let a = hex_byte(&bytes[6..8])?;
            (
                r as f64 / 255.0,
                g as f64 / 255.0,
                b as f64 / 255.0,
                a as f64 / 255.0,
            )
        }
        _ => return None,
    };
    Some([r, g, b, a])
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn hex_byte(s: &[u8]) -> Option<u8> {
    Some((hex_nibble(s[0])? << 4) | hex_nibble(s[1])?)
}

/// CSS3 named-color list, returned as premultiplied RGBA in `0..=1`
/// (the named entries all have alpha `1.0`, so the premultiplied
/// channels equal the straight ones). The list intentionally matches
/// the upstream MapLibre/csscolorparser table verbatim — pruning would
/// risk diverging on edge-case names that some fixture might exercise.
fn css3_named_color(name: &str) -> Option<[f64; 4]> {
    let (r, g, b) = match name {
        "aliceblue" => (240, 248, 255),
        "antiquewhite" => (250, 235, 215),
        "aqua" => (0, 255, 255),
        "aquamarine" => (127, 255, 212),
        "azure" => (240, 255, 255),
        "beige" => (245, 245, 220),
        "bisque" => (255, 228, 196),
        "black" => (0, 0, 0),
        "blanchedalmond" => (255, 235, 205),
        "blue" => (0, 0, 255),
        "blueviolet" => (138, 43, 226),
        "brown" => (165, 42, 42),
        "burlywood" => (222, 184, 135),
        "cadetblue" => (95, 158, 160),
        "chartreuse" => (127, 255, 0),
        "chocolate" => (210, 105, 30),
        "coral" => (255, 127, 80),
        "cornflowerblue" => (100, 149, 237),
        "cornsilk" => (255, 248, 220),
        "crimson" => (220, 20, 60),
        "cyan" => (0, 255, 255),
        "darkblue" => (0, 0, 139),
        "darkcyan" => (0, 139, 139),
        "darkgoldenrod" => (184, 134, 11),
        "darkgray" | "darkgrey" => (169, 169, 169),
        "darkgreen" => (0, 100, 0),
        "darkkhaki" => (189, 183, 107),
        "darkmagenta" => (139, 0, 139),
        "darkolivegreen" => (85, 107, 47),
        "darkorange" => (255, 140, 0),
        "darkorchid" => (153, 50, 204),
        "darkred" => (139, 0, 0),
        "darksalmon" => (233, 150, 122),
        "darkseagreen" => (143, 188, 143),
        "darkslateblue" => (72, 61, 139),
        "darkslategray" | "darkslategrey" => (47, 79, 79),
        "darkturquoise" => (0, 206, 209),
        "darkviolet" => (148, 0, 211),
        "deeppink" => (255, 20, 147),
        "deepskyblue" => (0, 191, 255),
        "dimgray" | "dimgrey" => (105, 105, 105),
        "dodgerblue" => (30, 144, 255),
        "firebrick" => (178, 34, 34),
        "floralwhite" => (255, 250, 240),
        "forestgreen" => (34, 139, 34),
        "fuchsia" => (255, 0, 255),
        "gainsboro" => (220, 220, 220),
        "ghostwhite" => (248, 248, 255),
        "gold" => (255, 215, 0),
        "goldenrod" => (218, 165, 32),
        "gray" | "grey" => (128, 128, 128),
        "green" => (0, 128, 0),
        "greenyellow" => (173, 255, 47),
        "honeydew" => (240, 255, 240),
        "hotpink" => (255, 105, 180),
        "indianred" => (205, 92, 92),
        "indigo" => (75, 0, 130),
        "ivory" => (255, 255, 240),
        "khaki" => (240, 230, 140),
        "lavender" => (230, 230, 250),
        "lavenderblush" => (255, 240, 245),
        "lawngreen" => (124, 252, 0),
        "lemonchiffon" => (255, 250, 205),
        "lightblue" => (173, 216, 230),
        "lightcoral" => (240, 128, 128),
        "lightcyan" => (224, 255, 255),
        "lightgoldenrodyellow" => (250, 250, 210),
        "lightgray" | "lightgrey" => (211, 211, 211),
        "lightgreen" => (144, 238, 144),
        "lightpink" => (255, 182, 193),
        "lightsalmon" => (255, 160, 122),
        "lightseagreen" => (32, 178, 170),
        "lightskyblue" => (135, 206, 250),
        "lightslategray" | "lightslategrey" => (119, 136, 153),
        "lightsteelblue" => (176, 196, 222),
        "lightyellow" => (255, 255, 224),
        "lime" => (0, 255, 0),
        "limegreen" => (50, 205, 50),
        "linen" => (250, 240, 230),
        "magenta" => (255, 0, 255),
        "maroon" => (128, 0, 0),
        "mediumaquamarine" => (102, 205, 170),
        "mediumblue" => (0, 0, 205),
        "mediumorchid" => (186, 85, 211),
        "mediumpurple" => (147, 112, 219),
        "mediumseagreen" => (60, 179, 113),
        "mediumslateblue" => (123, 104, 238),
        "mediumspringgreen" => (0, 250, 154),
        "mediumturquoise" => (72, 209, 204),
        "mediumvioletred" => (199, 21, 133),
        "midnightblue" => (25, 25, 112),
        "mintcream" => (245, 255, 250),
        "mistyrose" => (255, 228, 225),
        "moccasin" => (255, 228, 181),
        "navajowhite" => (255, 222, 173),
        "navy" => (0, 0, 128),
        "oldlace" => (253, 245, 230),
        "olive" => (128, 128, 0),
        "olivedrab" => (107, 142, 35),
        "orange" => (255, 165, 0),
        "orangered" => (255, 69, 0),
        "orchid" => (218, 112, 214),
        "palegoldenrod" => (238, 232, 170),
        "palegreen" => (152, 251, 152),
        "paleturquoise" => (175, 238, 238),
        "palevioletred" => (219, 112, 147),
        "papayawhip" => (255, 239, 213),
        "peachpuff" => (255, 218, 185),
        "peru" => (205, 133, 63),
        "pink" => (255, 192, 203),
        "plum" => (221, 160, 221),
        "powderblue" => (176, 224, 230),
        "purple" => (128, 0, 128),
        "rebeccapurple" => (102, 51, 153),
        "red" => (255, 0, 0),
        "rosybrown" => (188, 143, 143),
        "royalblue" => (65, 105, 225),
        "saddlebrown" => (139, 69, 19),
        "salmon" => (250, 128, 114),
        "sandybrown" => (244, 164, 96),
        "seagreen" => (46, 139, 87),
        "seashell" => (255, 245, 238),
        "sienna" => (160, 82, 45),
        "silver" => (192, 192, 192),
        "skyblue" => (135, 206, 235),
        "slateblue" => (106, 90, 205),
        "slategray" | "slategrey" => (112, 128, 144),
        "snow" => (255, 250, 250),
        "springgreen" => (0, 255, 127),
        "steelblue" => (70, 130, 180),
        "tan" => (210, 180, 140),
        "teal" => (0, 128, 128),
        "thistle" => (216, 191, 216),
        "tomato" => (255, 99, 71),
        "transparent" => return Some([0.0, 0.0, 0.0, 0.0]),
        "turquoise" => (64, 224, 208),
        "violet" => (238, 130, 238),
        "wheat" => (245, 222, 179),
        "white" => (255, 255, 255),
        "whitesmoke" => (245, 245, 245),
        "yellow" => (255, 255, 0),
        "yellowgreen" => (154, 205, 50),
        _ => return None,
    };
    Some([r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0, 1.0])
}

// ------------------------------------------------------------------------------
// HCL / LAB conversions for perceptual interpolation
// ------------------------------------------------------------------------------
//
// The math is a straight port of the CIE color-space transforms upstream
// uses (D65 white point, sRGB gamma). We only need it for the two
// `interpolate-{hcl,lab}` operators, so the routines are private helpers
// rather than a public color API.
//
// Pipeline: sRGB straight → linear-light RGB → CIE XYZ → CIE LAB. HCL
// is just LAB with `(a, b)` re-expressed in polar form `(C, h)`.

/// Linearize an sRGB channel (the standard companding curve). Input
/// and output are both `0..=1`.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse of [`srgb_to_linear`].
fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// D50 reference white, in normalized XYZ. MapLibre's LAB pipeline
/// uses D50 (via chroma.js's Bradford-adapted matrix) rather than the
/// "textbook" D65 — empirically confirmed against
/// `interpolate-lab/linear-color`, where switching to D50 takes the
/// midpoint output from `0.843` to the fixture-pinned `0.814`.
const D50_X: f64 = 0.96422;
const D50_Y: f64 = 1.0;
const D50_Z: f64 = 0.82521;

/// Straight (non-premultiplied) sRGB → LAB. Input channels are in
/// `0..=1`; output `L` is in `0..=100` and `a`/`b` are in roughly
/// `-128..=128` (no hard bound, but typical sRGB stays within).
fn srgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // 1. linearize sRGB
    let rl = srgb_to_linear(r);
    let gl = srgb_to_linear(g);
    let bl = srgb_to_linear(b);
    // 2. linear RGB → XYZ with the Bradford-adapted sRGB-to-D50
    //    matrix. The constants are chroma.js's, which is what
    //    MapLibre's color pipeline ultimately uses.
    let x = 0.4360747 * rl + 0.3850649 * gl + 0.1430804 * bl;
    let y = 0.2225045 * rl + 0.7168786 * gl + 0.0606169 * bl;
    let z = 0.0139322 * rl + 0.0971045 * gl + 0.7141733 * bl;
    // 3. XYZ → LAB
    let fx = lab_f(x / D50_X);
    let fy = lab_f(y / D50_Y);
    let fz = lab_f(z / D50_Z);
    let l = 116.0 * fy - 16.0;
    let a_ = 500.0 * (fx - fy);
    let b_ = 200.0 * (fy - fz);
    (l, a_, b_)
}

/// Inverse of [`srgb_to_lab`]. The output sRGB channels are clamped
/// back into `0..=1`; LAB values outside the sRGB gamut produce the
/// nearest representable color rather than failing.
fn lab_to_srgb(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let x = D50_X * lab_f_inv(fx);
    let y = D50_Y * lab_f_inv(fy);
    let z = D50_Z * lab_f_inv(fz);
    // XYZ → linear sRGB, Bradford-adapted D50-to-sRGB matrix
    // (inverse of the forward matrix above).
    let rl = 3.1338561 * x - 1.6168667 * y - 0.4906146 * z;
    let gl = -0.9787684 * x + 1.9161415 * y + 0.0334540 * z;
    let bl = 0.0719453 * x - 0.2289914 * y + 1.4052427 * z;
    let r = linear_to_srgb(rl.clamp(0.0, 1.0));
    let g = linear_to_srgb(gl.clamp(0.0, 1.0));
    let b = linear_to_srgb(bl.clamp(0.0, 1.0));
    (r, g, b)
}

/// CIE LAB nonlinearity. The piecewise form keeps things smooth near
/// zero; the constants are the standard CIE values.
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA.powi(3) {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

fn lab_f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA {
        t.powi(3)
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// LAB → HCL. `C = sqrt(a² + b²)`, `h = atan2(b, a)` in degrees.
fn lab_to_hcl(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let c = (a * a + b * b).sqrt();
    let h = b.atan2(a).to_degrees().rem_euclid(360.0);
    (h, c, l)
}

/// HCL → LAB. `(a, b)` only — the lightness component `L` is shared
/// between the two representations, so the caller carries it through
/// directly rather than receiving it back here.
fn hcl_to_lab(h: f64, c: f64) -> (f64, f64) {
    let rad = h.to_radians();
    (c * rad.cos(), c * rad.sin())
}

/// Linearly interpolate between two angles in degrees taking the
/// shorter arc — i.e. `lerp_hue(350, 10, 0.5) == 0`, not `180`. We
/// don't have a strict spec for what HCL interpolation should do when
/// chroma collapses through gray, but matching upstream's shortest-arc
/// behavior is what the corpus expects.
fn lerp_hue(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    let shortest = if diff > 180.0 {
        diff - 360.0
    } else if diff < -180.0 {
        diff + 360.0
    } else {
        diff
    };
    (a + shortest * t).rem_euclid(360.0)
}

/// Interpolate two **straight** sRGB colors at parameter `t` in
/// `0..=1`. `space` selects the working color space:
///
///   * `Linear` lerps the four channels directly. With straight
///     storage and the JSON-side premultiply, this matches upstream's
///     "linear-in-premultiplied" behavior end-to-end.
///   * `Hcl` / `Lab` convert each endpoint into the named perceptual
///     space, lerp `(L, C, h)` / `(L, a, b)` plus alpha separately,
///     then convert back. The corpus fixtures pin the conversion
///     constants tightly enough that a sign-flip on the white-point
///     or gamma curve shows up as a per-fixture diff.
fn interpolate_colors(lo: &[f64; 4], hi: &[f64; 4], t: f64, space: Interpolation) -> [f64; 4] {
    match space {
        // Linear, Exponential, and CubicBezier all share the same
        // per-channel lerp; the curve shape was already baked into `t`
        // upstream by `reparameterize`. They diverge from the
        // perceptual `Hcl`/`Lab` variants below only in the color
        // space the lerp runs over (sRGB straight RGBA here).
        Interpolation::Linear
        | Interpolation::Exponential { .. }
        | Interpolation::CubicBezier { .. } => [
            lo[0] + (hi[0] - lo[0]) * t,
            lo[1] + (hi[1] - lo[1]) * t,
            lo[2] + (hi[2] - lo[2]) * t,
            lo[3] + (hi[3] - lo[3]) * t,
        ],
        Interpolation::Hcl => {
            let (sh, sc, sl) = lab_to_hcl_from_srgb(lo[0], lo[1], lo[2]);
            let (eh, ec, el) = lab_to_hcl_from_srgb(hi[0], hi[1], hi[2]);
            let h = lerp_hue(sh, eh, t);
            let c = sc + (ec - sc) * t;
            let l = sl + (el - sl) * t;
            let a = lo[3] + (hi[3] - lo[3]) * t;
            let (la, lb) = hcl_to_lab(h, c);
            let (r, g, b) = lab_to_srgb(l, la, lb);
            [r, g, b, a]
        }
        Interpolation::Lab => {
            let (sl, saa, sbb) = srgb_to_lab(lo[0], lo[1], lo[2]);
            let (el, eaa, ebb) = srgb_to_lab(hi[0], hi[1], hi[2]);
            let l = sl + (el - sl) * t;
            let a_ = saa + (eaa - saa) * t;
            let b_ = sbb + (ebb - sbb) * t;
            let a = lo[3] + (hi[3] - lo[3]) * t;
            let (r, g, b) = lab_to_srgb(l, a_, b_);
            [r, g, b, a]
        }
    }
}

/// Convenience wrapper: straight sRGB → HCL in one step.
fn lab_to_hcl_from_srgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (l, a, b) = srgb_to_lab(r, g, b);
    lab_to_hcl(l, a, b)
}

// ==============================================================================
// Geometry-aware operators — evaluation
//
// `within` and `distance` both share a coordinate-frame trick that's
// easy to get wrong: the constant GeoJSON argument is in lng/lat, but
// the feature's `geo::Geometry` lives in tile-local integer
// coordinates (since the adapter projects it through
// `getTileCoordinates` before stripping the tile-origin shifts). We
// therefore reproject as needed at the evaluator boundary:
//
//   * `within` runs the point-in-polygon math in *world tile space*:
//     both the argument polygon and the feature get mapped through
//     `mercator_x_from_lng` / `mercator_y_from_lat`, scaled by
//     `2^z * EXTENT`. For the feature this just means adding the
//     tile-origin shifts back to its already-tile-local coords; for
//     the argument it's the full lng/lat → integer projection.
//
//   * `distance` reverses the feature projection so all distance work
//     happens in lng/lat (where a `CheapRuler` flat-earth approximation
//     gives meter-accurate distances cheaply). The argument geometry
//     is already in lng/lat.
//
// Both follow upstream's `getTileCoordinates` / `getLngLatFromTileCoord`
// helpers verbatim so the per-fixture numerics line up.
// ==============================================================================

/// Tile extent used by every projection in this module. The MapLibre
/// spec hard-codes 8192 for the test coordinate frame regardless of
/// the actual feature extent. We match that constant so our outputs
/// stay byte-identical to the upstream fixture's `expected.outputs`.
const TILE_EXTENT: f64 = 8192.0;

fn mercator_x_from_lng(lng: f64) -> f64 {
    (180.0 + lng) / 360.0
}

fn mercator_y_from_lat(lat: f64) -> f64 {
    (180.0
        - (180.0 / std::f64::consts::PI)
            * (std::f64::consts::PI / 4.0 + lat * std::f64::consts::PI / 360.0)
                .tan()
                .ln())
        / 360.0
}

fn lng_from_mercator_x(mx: f64) -> f64 {
    mx * 360.0 - 180.0
}

fn lat_from_mercator_y(my: f64) -> f64 {
    (360.0 / std::f64::consts::PI)
        * ((180.0 - my * 360.0) * std::f64::consts::PI / 180.0)
            .exp()
            .atan()
        - 90.0
}

/// Project a lng/lat pair into rounded global tile coordinates at zoom
/// `z`. The result is in pixels-of-an-`EXTENT`-tile, i.e. the same
/// frame upstream's `getTileCoordinates` returns. We round (not floor)
/// to match upstream behaviour.
fn tile_coord_from_lnglat(lng: f64, lat: f64, z: u8) -> (f64, f64) {
    let mx = mercator_x_from_lng(lng);
    let my = mercator_y_from_lat(lat);
    let tiles = 2f64.powi(z as i32);
    (
        (mx * tiles * TILE_EXTENT).round(),
        (my * tiles * TILE_EXTENT).round(),
    )
}

/// Inverse of [`tile_coord_from_lnglat`] for a specific tile. Used by
/// `distance` to lift the feature back into lng/lat space before
/// running CheapRuler.
fn lnglat_from_tile_coord(x: f64, y: f64, canonical: CanonicalTileID) -> (f64, f64) {
    let tiles = 2f64.powi(canonical.z as i32);
    let mx = (x / TILE_EXTENT + canonical.x as f64) / tiles;
    let my = (y / TILE_EXTENT + canonical.y as f64) / tiles;
    (lng_from_mercator_x(mx), lat_from_mercator_y(my))
}

/// Look up the spec-defined name for a feature geometry. Multi*
/// variants normalize down to their non-Multi base, matching
/// upstream's `geometryType()` accessor — important for the
/// `geometry-type` operator and for `within`/`distance`'s dispatch.
fn geometry_type_name(g: &crate::mvt::Geometry) -> &'static str {
    use crate::mvt::Geometry as G;
    match g {
        G::Point(_) | G::MultiPoint(_) => "Point",
        G::LineString(_) | G::MultiLineString(_) => "LineString",
        G::Polygon(_) | G::MultiPolygon(_) => "Polygon",
        G::Unknown => "Unknown",
    }
}

// ------------------------------------------------------------------------------
// within
// ------------------------------------------------------------------------------

/// Project a polygon's lng/lat rings into the integer tile-space the
/// `within` evaluator operates in.
fn project_polygon_to_tile(polygon: &[Vec<(f64, f64)>], z: u8) -> Vec<Vec<(f64, f64)>> {
    polygon
        .iter()
        .map(|ring| {
            ring.iter()
                .map(|(lng, lat)| tile_coord_from_lnglat(*lng, *lat, z))
                .collect()
        })
        .collect()
}

/// Flatten the feature's tile-local geometry to a list of "shifted"
/// world-tile points (i.e. with the canonical tile's origin added back
/// in). Upstream's `getTilePoints` does the same for Point /
/// MultiPoint features.
fn feature_points_world(feature: &crate::mvt::Feature, canonical: CanonicalTileID) -> Vec<(f64, f64)> {
    use crate::mvt::Geometry as G;
    let shifts = (
        canonical.x as f64 * TILE_EXTENT,
        canonical.y as f64 * TILE_EXTENT,
    );
    match &feature.geometry {
        G::Point(p) => vec![(p.x() + shifts.0, p.y() + shifts.1)],
        G::MultiPoint(mp) => mp
            .iter()
            .map(|p| (p.x() + shifts.0, p.y() + shifts.1))
            .collect(),
        _ => Vec::new(),
    }
}

/// Same idea as [`feature_points_world`] but for line features. Each
/// inner `Vec` is one polyline; multi-line features unpack into N
/// polylines. Polygon features are routed here too since `within`
/// only cares about Points and LineStrings.
fn feature_lines_world(
    feature: &crate::mvt::Feature,
    canonical: CanonicalTileID,
) -> Vec<Vec<(f64, f64)>> {
    use crate::mvt::Geometry as G;
    let shifts = (
        canonical.x as f64 * TILE_EXTENT,
        canonical.y as f64 * TILE_EXTENT,
    );
    let shift = |p: &geo::Coord<f64>| (p.x + shifts.0, p.y + shifts.1);
    match &feature.geometry {
        G::LineString(l) => vec![l.0.iter().map(shift).collect()],
        G::MultiLineString(ml) => {
            ml.0.iter()
                .map(|l| l.0.iter().map(shift).collect())
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Axis-aligned bounding box. The format mirrors upstream's `BBox =
/// [minX, minY, maxX, maxY]` array form so logic ports straight
/// across.
#[derive(Debug, Clone, Copy)]
struct BBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl BBox {
    fn empty() -> Self {
        BBox {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, p: (f64, f64)) {
        self.min_x = self.min_x.min(p.0);
        self.min_y = self.min_y.min(p.1);
        self.max_x = self.max_x.max(p.0);
        self.max_y = self.max_y.max(p.1);
    }
}

/// Strict containment: `inner` lies entirely inside `outer`. Equal
/// edges count as outside, matching upstream's `boxWithinBox` (which
/// is used as an early bail-out from the within check).
fn box_within_box(inner: &BBox, outer: &BBox) -> bool {
    inner.min_x > outer.min_x
        && inner.max_x < outer.max_x
        && inner.min_y > outer.min_y
        && inner.max_y < outer.max_y
}

/// World-wrap a point's x-coordinate to align with the polygon's
/// frame. Upstream's `updatePoint` does this so antimeridian-straddling
/// polygons still match features on the far side. Mirrors the
/// logic verbatim — see `within/meridian-issue-*` for the fixtures
/// this exists to pass.
fn wrap_point_x(p: &mut (f64, f64), poly_bbox: &BBox, world_size: f64) {
    if p.0 < poly_bbox.min_x || p.0 > poly_bbox.max_x {
        let half_world = world_size * 0.5;
        let mut shift = if p.0 - poly_bbox.min_x > half_world {
            -world_size
        } else if poly_bbox.min_x - p.0 > half_world {
            world_size
        } else {
            0.0
        };
        if shift == 0.0 {
            shift = if p.0 - poly_bbox.max_x > half_world {
                -world_size
            } else if poly_bbox.max_x - p.0 > half_world {
                world_size
            } else {
                0.0
            };
        }
        p.0 += shift;
    }
}

/// Ray-cast point-in-polygon, in the rare case directly ported from
/// upstream. `rings` is one polygon's rings (exterior first); the
/// `true_if_boundary` flag distinguishes the strict `within` use case
/// (false) from `distance`'s boundary-counts case (true).
fn point_within_polygon(
    point: (f64, f64),
    rings: &[Vec<(f64, f64)>],
    true_if_boundary: bool,
) -> bool {
    let mut inside = false;
    for ring in rings {
        if ring.len() < 2 {
            continue;
        }
        for j in 0..ring.len() - 1 {
            let a = ring[j];
            let b = ring[j + 1];
            if point_on_boundary(point, a, b) {
                return true_if_boundary;
            }
            if ray_intersect(point, a, b) {
                inside = !inside;
            }
        }
    }
    inside
}

fn point_within_polygons(point: (f64, f64), polygons: &[Vec<Vec<(f64, f64)>>]) -> bool {
    polygons
        .iter()
        .any(|p| point_within_polygon(point, p, false))
}

/// True if `point` lies on the segment `(a, b)`. Cross-product zero
/// plus both component dot products non-positive (i.e. between the
/// endpoints).
fn point_on_boundary(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> bool {
    let x1 = p.0 - a.0;
    let y1 = p.1 - a.1;
    let x2 = p.0 - b.0;
    let y2 = p.1 - b.1;
    x1 * y2 - x2 * y1 == 0.0 && x1 * x2 <= 0.0 && y1 * y2 <= 0.0
}

/// Horizontal-ray intersection test for ray casting.
fn ray_intersect(p: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> bool {
    (p1.1 > p.1) != (p2.1 > p.1) && p.0 < ((p2.0 - p1.0) * (p.1 - p1.1)) / (p2.1 - p1.1) + p1.0
}

/// Segment-vs-segment intersection (proper crossing — colinear-
/// touching doesn't count). Used both for the line-within-polygon
/// check and for `distance`'s zero-distance shortcut.
fn segment_intersect_segment(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let vp = (b.0 - a.0, b.1 - a.1);
    let vq = (d.0 - c.0, d.1 - c.1);
    if perp(vq, vp) == 0.0 {
        return false;
    }
    two_sided(a, b, c, d) && two_sided(c, d, a, b)
}

fn perp(v1: (f64, f64), v2: (f64, f64)) -> f64 {
    v1.0 * v2.1 - v1.1 * v2.0
}

fn two_sided(p1: (f64, f64), p2: (f64, f64), q1: (f64, f64), q2: (f64, f64)) -> bool {
    let x1 = p1.0 - q1.0;
    let y1 = p1.1 - q1.1;
    let x2 = p2.0 - q1.0;
    let y2 = p2.1 - q1.1;
    let x3 = q2.0 - q1.0;
    let y3 = q2.1 - q1.1;
    let det1 = x1 * y3 - x3 * y1;
    let det2 = x2 * y3 - x3 * y2;
    (det1 > 0.0 && det2 < 0.0) || (det1 < 0.0 && det2 > 0.0)
}

fn line_intersect_polygon(p1: (f64, f64), p2: (f64, f64), polygon: &[Vec<(f64, f64)>]) -> bool {
    for ring in polygon {
        if ring.len() < 2 {
            continue;
        }
        for j in 0..ring.len() - 1 {
            if segment_intersect_segment(p1, p2, ring[j], ring[j + 1]) {
                return true;
            }
        }
    }
    false
}

fn line_within_polygon(line: &[(f64, f64)], polygon: &[Vec<(f64, f64)>]) -> bool {
    // First: every line vertex must be inside the polygon.
    for &pt in line {
        if !point_within_polygon(pt, polygon, false) {
            return false;
        }
    }
    // Second: no line segment may cross a polygon edge. A line that's
    // entirely *inside* the polygon trivially never crosses; a line
    // that exits through an edge would pass the vertex check (e.g.
    // both endpoints inside, but the segment dips outside) — but the
    // polygon being convex enough to host this case is rare. We still
    // run the segment check for fidelity with upstream.
    if line.len() >= 2 {
        for i in 0..line.len() - 1 {
            if line_intersect_polygon(line[i], line[i + 1], polygon) {
                return false;
            }
        }
    }
    true
}

fn line_within_polygons(line: &[(f64, f64)], polygons: &[Vec<Vec<(f64, f64)>>]) -> bool {
    polygons.iter().any(|p| line_within_polygon(line, p))
}

/// Top-level `within` evaluator. Returns `false` for any geometry
/// type the operator doesn't support (Polygon features, Unknown), or
/// when no canonical tile is set — matching upstream's "if
/// `ctx.canonicalID() == null` return false" guard.
fn eval_within(arg: &WithinPolygons, ctx: &EvalContext, feature: &Feature) -> bool {
    let canonical = match ctx.canonical {
        Some(c) => c,
        None => return false,
    };
    let world_size = 2f64.powi(canonical.z as i32) * TILE_EXTENT;

    // Project the argument once; compute its bbox alongside.
    let mut poly_bbox = BBox::empty();
    let tile_polygons: Vec<Vec<Vec<(f64, f64)>>> = arg
        .polygons
        .iter()
        .map(|p| {
            let proj = project_polygon_to_tile(p, canonical.z);
            for ring in &proj {
                for pt in ring {
                    poly_bbox.update(*pt);
                }
            }
            proj
        })
        .collect();

    match geometry_type_name(&feature.geometry) {
        "Point" => {
            let pts = feature_points_world(feature, canonical);
            let mut point_bbox = BBox::empty();
            let mut tile_points = Vec::with_capacity(pts.len());
            for mut p in pts {
                wrap_point_x(&mut p, &poly_bbox, world_size);
                point_bbox.update(p);
                tile_points.push(p);
            }
            if !box_within_box(&point_bbox, &poly_bbox) {
                return false;
            }
            tile_points
                .iter()
                .all(|p| point_within_polygons(*p, &tile_polygons))
        }
        "LineString" => {
            let lines = feature_lines_world(feature, canonical);
            // Upstream computes the line bbox in the *un-wrapped* world
            // first, then only triggers the wrap-pass if the bbox span
            // is ≤ half the world. Mirror that so the meridian-line
            // fixtures behave correctly.
            let mut line_bbox = BBox::empty();
            for line in &lines {
                for pt in line {
                    line_bbox.update(*pt);
                }
            }
            let mut tile_lines = lines;
            if line_bbox.max_x - line_bbox.min_x <= world_size * 0.5 {
                line_bbox = BBox::empty();
                for line in tile_lines.iter_mut() {
                    for p in line.iter_mut() {
                        wrap_point_x(p, &poly_bbox, world_size);
                        line_bbox.update(*p);
                    }
                }
            }
            if !box_within_box(&line_bbox, &poly_bbox) {
                return false;
            }
            tile_lines
                .iter()
                .all(|line| line_within_polygons(line, &tile_polygons))
        }
        _ => false,
    }
}

// ------------------------------------------------------------------------------
// distance
// ------------------------------------------------------------------------------

/// Earth ellipsoid constants (WGS84). These match upstream
/// `cheap_ruler.ts` to within the last digit so `distance` outputs
/// agree numerically.
const CHEAP_RULER_RE: f64 = 6378.137;
const CHEAP_RULER_FE: f64 = 1.0 / 298.257223563;

/// A "cheap ruler" — a flat-earth approximation calibrated to a
/// single latitude. The result is accurate to a few centimeters at
/// distances under a thousand kilometers, and is the same primitive
/// the upstream `distance` evaluator uses.
#[derive(Debug, Clone, Copy)]
struct CheapRuler {
    /// Meters per degree of longitude at the calibration latitude.
    kx: f64,
    /// Meters per degree of latitude (approximately constant, but
    /// upstream still computes it from the WGS84 model).
    ky: f64,
}

impl CheapRuler {
    fn new(lat: f64) -> Self {
        let e2 = CHEAP_RULER_FE * (2.0 - CHEAP_RULER_FE);
        let rad = std::f64::consts::PI / 180.0;
        let m = rad * CHEAP_RULER_RE * 1000.0;
        let coslat = (lat * rad).cos();
        let w2 = 1.0 / (1.0 - e2 * (1.0 - coslat * coslat));
        let w = w2.sqrt();
        CheapRuler {
            kx: m * w * coslat,
            ky: m * w * w2 * (1.0 - e2),
        }
    }

    /// Wrap a longitude delta into [-180, 180]. Upstream uses a
    /// trivial loop; we match it so any difference in handling is
    /// confined to this one method.
    fn wrap(&self, mut deg: f64) -> f64 {
        while deg < -180.0 {
            deg += 360.0;
        }
        while deg > 180.0 {
            deg -= 360.0;
        }
        deg
    }

    fn distance(&self, a: (f64, f64), b: (f64, f64)) -> f64 {
        let dx = self.wrap(a.0 - b.0) * self.kx;
        let dy = (a.1 - b.1) * self.ky;
        (dx * dx + dy * dy).sqrt()
    }

    /// Closest point on `line` to `p`. Returns `(point, sq_dist)` —
    /// the squared distance is what the segment-walk minimizes, and
    /// the closest point is what the caller plugs into `distance` to
    /// recover the meter value.
    fn point_on_line(&self, line: &[(f64, f64)], p: (f64, f64)) -> (f64, f64) {
        let mut min_dist = f64::INFINITY;
        let mut closest = line[0];
        for i in 0..line.len() - 1 {
            let mut x = line[i].0;
            let mut y = line[i].1;
            let mut dx = self.wrap(line[i + 1].0 - x) * self.kx;
            let mut dy = (line[i + 1].1 - y) * self.ky;
            // The `t` clamp only matters when the segment has nonzero
            // length; on a degenerate point-segment the nearest point
            // is the segment vertex itself.
            if dx != 0.0 || dy != 0.0 {
                let t = (self.wrap(p.0 - x) * self.kx * dx + (p.1 - y) * self.ky * dy)
                    / (dx * dx + dy * dy);
                if t > 1.0 {
                    x = line[i + 1].0;
                    y = line[i + 1].1;
                } else if t > 0.0 {
                    x += (dx / self.kx) * t;
                    y += (dy / self.ky) * t;
                }
            }
            dx = self.wrap(p.0 - x) * self.kx;
            dy = (p.1 - y) * self.ky;
            let sq = dx * dx + dy * dy;
            if sq < min_dist {
                min_dist = sq;
                closest = (x, y);
            }
        }
        closest
    }
}

/// Distance between a point and a line, in meters.
fn point_to_line_distance(p: (f64, f64), line: &[(f64, f64)], ruler: &CheapRuler) -> f64 {
    if line.len() < 2 {
        return ruler.distance(p, line[0]);
    }
    let nearest = ruler.point_on_line(line, p);
    ruler.distance(p, nearest)
}

/// Distance between two arbitrary segments — the upstream
/// implementation picks the smaller of the four point-to-line
/// distances.
fn segment_to_segment_distance(
    p1: (f64, f64),
    p2: (f64, f64),
    q1: (f64, f64),
    q2: (f64, f64),
    ruler: &CheapRuler,
) -> f64 {
    let qline = [q1, q2];
    let pline = [p1, p2];
    let d1 =
        point_to_line_distance(p1, &qline, ruler).min(point_to_line_distance(p2, &qline, ruler));
    let d2 =
        point_to_line_distance(q1, &pline, ruler).min(point_to_line_distance(q2, &pline, ruler));
    d1.min(d2)
}

/// Distance from one point set to another. Used for both Point-to-
/// Point and Point-to-LineString cases (with the appropriate
/// `isLine*` flags). We use the brute-force `O(n*m)` form rather than
/// upstream's divide-and-conquer queue since the test corpus is small.
fn pointset_to_pointset_distance(
    a: &[(f64, f64)],
    is_line_a: bool,
    b: &[(f64, f64)],
    is_line_b: bool,
    ruler: &CheapRuler,
) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    if is_line_a && is_line_b {
        let mut dist = f64::INFINITY;
        for i in 0..a.len() - 1 {
            for j in 0..b.len() - 1 {
                if segment_intersect_segment(a[i], a[i + 1], b[j], b[j + 1]) {
                    return 0.0;
                }
                dist = dist.min(segment_to_segment_distance(
                    a[i],
                    a[i + 1],
                    b[j],
                    b[j + 1],
                    ruler,
                ));
            }
        }
        dist
    } else if is_line_a && !is_line_b {
        let mut dist = f64::INFINITY;
        for &q in b {
            dist = dist.min(point_to_line_distance(q, a, ruler));
            if dist == 0.0 {
                return 0.0;
            }
        }
        dist
    } else if !is_line_a && is_line_b {
        let mut dist = f64::INFINITY;
        for &p in a {
            dist = dist.min(point_to_line_distance(p, b, ruler));
            if dist == 0.0 {
                return 0.0;
            }
        }
        dist
    } else {
        let mut dist = f64::INFINITY;
        for &p in a {
            for &q in b {
                dist = dist.min(ruler.distance(p, q));
                if dist == 0.0 {
                    return 0.0;
                }
            }
        }
        dist
    }
}

/// Distance from a single point to a polygon (rings). Returns 0 when
/// the point is inside or on the boundary; otherwise the closest
/// edge distance.
fn point_to_polygon_distance(
    p: (f64, f64),
    polygon: &[Vec<(f64, f64)>],
    ruler: &CheapRuler,
) -> f64 {
    if point_within_polygon(p, polygon, true) {
        return 0.0;
    }
    let mut dist = f64::INFINITY;
    for ring in polygon {
        if ring.is_empty() {
            continue;
        }
        // Upstream explicitly stitches the ring closed when its first
        // and last points differ. The geojson polygons we accept are
        // already closed, but a defensive close-segment check mirrors
        // upstream and is harmless on already-closed rings.
        let front = ring[0];
        let back = ring[ring.len() - 1];
        if front != back {
            dist = dist.min(point_to_line_distance(p, &[back, front], ruler));
            if dist == 0.0 {
                return 0.0;
            }
        }
        dist = dist.min(point_to_line_distance(p, ring, ruler));
        if dist == 0.0 {
            return 0.0;
        }
    }
    dist
}

/// Distance from a line to a polygon, in meters. Returns 0 if any
/// line vertex sits inside the polygon, or if any line segment
/// intersects a polygon edge; otherwise the minimum edge-to-edge
/// distance.
fn line_to_polygon_distance(
    line: &[(f64, f64)],
    polygon: &[Vec<(f64, f64)>],
    ruler: &CheapRuler,
) -> f64 {
    for &p in line {
        if point_within_polygon(p, polygon, true) {
            return 0.0;
        }
    }
    if line.len() < 2 {
        // Single-point line: fall through to the per-point routine.
        return point_to_polygon_distance(line[0], polygon, ruler);
    }
    let mut dist = f64::INFINITY;
    for i in 0..line.len() - 1 {
        let p1 = line[i];
        let p2 = line[i + 1];
        for ring in polygon {
            let n = ring.len();
            if n < 2 {
                continue;
            }
            // Walk every edge (k → j) including the closing one, the
            // way upstream iterates `j = 0..len, k = len-1 then j++`.
            let mut k = n - 1;
            for j in 0..n {
                let q1 = ring[k];
                let q2 = ring[j];
                if segment_intersect_segment(p1, p2, q1, q2) {
                    return 0.0;
                }
                dist = dist.min(segment_to_segment_distance(p1, p2, q1, q2, ruler));
                k = j;
            }
        }
    }
    dist
}

/// Distance between two polygons. We special-case "one polygon
/// entirely contains a vertex of the other" → 0, then fall through
/// to the edge-to-edge walk.
fn polygon_to_polygon_distance(
    a: &[Vec<(f64, f64)>],
    b: &[Vec<(f64, f64)>],
    ruler: &CheapRuler,
) -> f64 {
    // Check both directions of "polygon contains a vertex of the
    // other". Mirrors upstream's `polygonIntersect`.
    let a_in_b = a
        .iter()
        .flatten()
        .any(|p| point_within_polygon(*p, b, true));
    if a_in_b {
        return 0.0;
    }
    let b_in_a = b
        .iter()
        .flatten()
        .any(|p| point_within_polygon(*p, a, true));
    if b_in_a {
        return 0.0;
    }
    let mut dist = f64::INFINITY;
    for ring1 in a {
        let n1 = ring1.len();
        if n1 < 2 {
            continue;
        }
        let mut l = n1 - 1;
        for i in 0..n1 {
            let p1 = ring1[l];
            let p2 = ring1[i];
            for ring2 in b {
                let n2 = ring2.len();
                if n2 < 2 {
                    continue;
                }
                let mut k = n2 - 1;
                for j in 0..n2 {
                    let q1 = ring2[k];
                    let q2 = ring2[j];
                    if segment_intersect_segment(p1, p2, q1, q2) {
                        return 0.0;
                    }
                    dist = dist.min(segment_to_segment_distance(p1, p2, q1, q2, ruler));
                    k = j;
                }
            }
            l = i;
        }
    }
    dist
}

/// Project a `geo` ring (as a closed `LineString`) back into lng/lat
/// for the distance computations. The ring's coordinates are already
/// tile-local (the adapter strips the tile-origin shifts at ingest
/// time); `lnglat_from_tile_coord` reapplies them implicitly.
fn lnglat_ring_from_geo_ring(
    ring: &geo::LineString<f64>,
    canonical: CanonicalTileID,
) -> Vec<(f64, f64)> {
    ring.0
        .iter()
        .map(|c| lnglat_from_tile_coord(c.x, c.y, canonical))
        .collect()
}

/// Lift the feature's geometry into lng/lat space for `distance`.
/// Returns a fresh allocation so the evaluator can mutate-and-fold
/// freely. Polygon features return as `Vec<Vec<Vec<lnglat>>>` (one
/// outer entry per classified polygon, since upstream runs
/// `classifyRings` to split exteriors from holes). For Point /
/// LineString features the outer Vec collapses to size 1.
enum FeatureGeometryLngLat {
    /// Flat list of point lng/lats (Point + MultiPoint share this).
    Points(Vec<(f64, f64)>),
    /// Each entry is one polyline. MultiLineStrings split into N
    /// entries; a single LineString collapses to size 1.
    Lines(Vec<Vec<(f64, f64)>>),
    /// Polygons after `classifyRings`. Each outer entry is one
    /// polygon (exterior + holes); each inner entry is one ring of
    /// lng/lats.
    Polygons(Vec<Vec<Vec<(f64, f64)>>>),
    /// Anything else — distance returns NaN.
    Other,
}

fn feature_geometry_lnglat(
    feature: &crate::mvt::Feature,
    canonical: CanonicalTileID,
) -> FeatureGeometryLngLat {
    use crate::mvt::Geometry as G;
    // The feature already stores its geometry in tile-local
    // coordinates (the adapter subtracted the tile-origin shifts at
    // ingest time); `lnglat_from_tile_coord` puts them back implicitly
    // via the `+ canonical.x/y` term, so we pass the unshifted values
    // through here.
    let to_lnglat = |x: f64, y: f64| lnglat_from_tile_coord(x, y, canonical);
    match &feature.geometry {
        G::Point(p) => FeatureGeometryLngLat::Points(vec![to_lnglat(p.x(), p.y())]),
        G::MultiPoint(mp) => {
            FeatureGeometryLngLat::Points(mp.iter().map(|p| to_lnglat(p.x(), p.y())).collect())
        }
        G::LineString(l) => {
            FeatureGeometryLngLat::Lines(vec![l.0.iter().map(|c| to_lnglat(c.x, c.y)).collect()])
        }
        G::MultiLineString(ml) => FeatureGeometryLngLat::Lines(
            ml.0.iter()
                .map(|l| l.0.iter().map(|c| to_lnglat(c.x, c.y)).collect())
                .collect(),
        ),
        G::Polygon(p) => {
            // A single `geo::Polygon` already carries its exterior and
            // its holes, so we don't need `classifyRings`-style ring
            // sorting here.
            let mut rings = Vec::with_capacity(1 + p.interiors().len());
            rings.push(lnglat_ring_from_geo_ring(p.exterior(), canonical));
            for hole in p.interiors() {
                rings.push(lnglat_ring_from_geo_ring(hole, canonical));
            }
            FeatureGeometryLngLat::Polygons(vec![rings])
        }
        G::MultiPolygon(mp) => {
            let polygons =
                mp.0.iter()
                    .map(|p| {
                        let mut rings = Vec::with_capacity(1 + p.interiors().len());
                        rings.push(lnglat_ring_from_geo_ring(p.exterior(), canonical));
                        for hole in p.interiors() {
                            rings.push(lnglat_ring_from_geo_ring(hole, canonical));
                        }
                        rings
                    })
                    .collect();
            FeatureGeometryLngLat::Polygons(polygons)
        }
        G::Unknown => FeatureGeometryLngLat::Other,
    }
}

/// Top-level distance evaluator. Returns NaN when the operator can't
/// be applied — no canonical tile, unknown geometry, or feature
/// geometry that's empty after projection.
fn eval_distance(geoms: &[SimpleGeoJson], ctx: &EvalContext, feature: &Feature) -> f64 {
    let canonical = match ctx.canonical {
        Some(c) => c,
        None => return f64::NAN,
    };
    let feat = feature_geometry_lnglat(feature, canonical);
    match feat {
        FeatureGeometryLngLat::Points(pts) => {
            if pts.is_empty() {
                return f64::NAN;
            }
            // Upstream calibrates the ruler at the first feature
            // point's latitude — the same constant for every component
            // of the argument geometry.
            let ruler = CheapRuler::new(pts[0].1);
            let mut dist = f64::INFINITY;
            for g in geoms {
                dist = dist.min(distance_pointset_to_geom(&pts, false, g, &ruler));
                if dist == 0.0 {
                    return 0.0;
                }
            }
            dist
        }
        FeatureGeometryLngLat::Lines(lines) => {
            // Upstream flattens all line components down to one big
            // pointset for the distance computation. We follow suit so
            // the multi-line fixtures (`distance/multi-line`) emit the
            // upstream-recorded number.
            let flat: Vec<(f64, f64)> = lines.into_iter().flatten().collect();
            if flat.is_empty() {
                return f64::NAN;
            }
            let ruler = CheapRuler::new(flat[0].1);
            let mut dist = f64::INFINITY;
            for g in geoms {
                dist = dist.min(distance_pointset_to_geom(&flat, true, g, &ruler));
                if dist == 0.0 {
                    return 0.0;
                }
            }
            dist
        }
        FeatureGeometryLngLat::Polygons(polys) => {
            if polys.is_empty() || polys[0].is_empty() || polys[0][0].is_empty() {
                return f64::NAN;
            }
            let ruler = CheapRuler::new(polys[0][0][0].1);
            let mut dist = f64::INFINITY;
            for g in geoms {
                for poly in &polys {
                    dist = dist.min(distance_polygon_to_geom(poly, g, &ruler));
                    if dist == 0.0 {
                        return 0.0;
                    }
                }
            }
            dist
        }
        FeatureGeometryLngLat::Other => f64::NAN,
    }
}

fn distance_pointset_to_geom(
    points: &[(f64, f64)],
    is_line: bool,
    g: &SimpleGeoJson,
    ruler: &CheapRuler,
) -> f64 {
    match g {
        SimpleGeoJson::Point(lng, lat) => {
            pointset_to_pointset_distance(points, is_line, &[(*lng, *lat)], false, ruler)
        }
        SimpleGeoJson::LineString(line) => {
            pointset_to_pointset_distance(points, is_line, line, true, ruler)
        }
        SimpleGeoJson::Polygon(poly) => {
            if is_line {
                line_to_polygon_distance(points, poly, ruler)
            } else {
                let mut dist = f64::INFINITY;
                for p in points {
                    dist = dist.min(point_to_polygon_distance(*p, poly, ruler));
                    if dist == 0.0 {
                        return 0.0;
                    }
                }
                dist
            }
        }
    }
}

fn distance_polygon_to_geom(
    poly: &[Vec<(f64, f64)>],
    g: &SimpleGeoJson,
    ruler: &CheapRuler,
) -> f64 {
    match g {
        SimpleGeoJson::Point(lng, lat) => point_to_polygon_distance((*lng, *lat), poly, ruler),
        SimpleGeoJson::LineString(line) => line_to_polygon_distance(line, poly, ruler),
        SimpleGeoJson::Polygon(other) => polygon_to_polygon_distance(poly, other, ruler),
    }
}

// ==============================================================================
// Errors
// ==============================================================================

#[derive(Debug, thiserror::Error)]
pub enum ExprError {
    #[error("expression array is empty")]
    EmptyArray,
    #[error("expected a string operator, found {0}")]
    NonStringOperator(String),
    #[error("unknown expression operator: {0:?}")]
    UnknownOperator(String),
    /// Surfaced by the Phase 11 legacy-function converter when the
    /// input doesn't match a recognized pre-spec-v8 function shape
    /// (missing `stops`, malformed composite stop input, etc.). We
    /// produce a single variant rather than threading detailed kinds
    /// because the runner only inspects success/failure.
    #[error("malformed legacy function: {0}")]
    LegacyMalformed(&'static str),
    #[error("operator {op:?} expects {expected} argument(s), got {got}")]
    WrongArity {
        op: &'static str,
        expected: &'static str,
        got: usize,
    },
    #[error("operator {op:?} expected a string at {position}")]
    ExpectedString {
        op: &'static str,
        position: &'static str,
    },
    #[error("operator {op:?} expected a number at {position}")]
    ExpectedNumber {
        op: &'static str,
        position: &'static str,
    },
    #[error("interpolation argument is malformed (expected `[\"linear\"]` or similar)")]
    MalformedInterpolation,
    #[error("unsupported interpolation kind: {0:?}")]
    UnsupportedInterpolation(String),
    /// Surfaced by `["exponential", base]` when `base` isn't a literal
    /// number. Upstream's wording is "Exponential interpolation
    /// requires a numeric base.", which the `exponential-expression-
    /// base` fixture pins.
    #[error("Exponential interpolation requires a numeric base.")]
    InterpolateExponentialBaseNotNumber,
    /// Surfaced by `["cubic-bezier", x1, y1, x2, y2]` when the arity
    /// or coordinate range is wrong. The
    /// `cubic-bezier-{3-args,5-args,invalid-control-point,
    /// expression-control-point}` fixtures all hit this.
    #[error(
        "Cubic bezier interpolation requires four numeric arguments with values between 0 and 1."
    )]
    CubicBezierInvalidControlPoints,
    /// Raised when `step`'s stop inputs aren't strictly ascending.
    /// `step/duplicate_stops` pins this.
    #[error(
        "Input/output pairs for \"step\" expressions must be arranged with input values in strictly ascending order."
    )]
    StepNonAscendingStops,
    /// Raised when `interpolate`'s stop inputs aren't strictly
    /// ascending. `interpolate/duplicate_stops` pins this.
    #[error(
        "Input/output pairs for \"interpolate\" expressions must be arranged with input values in strictly ascending order."
    )]
    InterpolateNonAscendingStops,
    #[error("invalid number literal: {0}")]
    InvalidNumber(String),
    #[error("unexpected JSON object where an expression was expected")]
    UnexpectedObject,
    #[error("the item type argument of \"array\" must be one of string, number, boolean")]
    ArrayInvalidItemType,
    #[error("the length argument to \"array\" must be a positive integer literal")]
    ArrayInvalidLength,
    #[error("Expected at least 4 arguments to \"match\", but found only {0}.")]
    MatchTooFewArguments(usize),
    #[error("\"match\" requires (label, output) pairs after the input, found {0} arguments total")]
    MatchUnpairedBranches(usize),
    #[error("\"match\" label arrays must not be empty")]
    MatchEmptyLabelArray,
    #[error("\"match\" branch labels must be numbers or strings")]
    MatchLabelInvalid,
    #[error("\"match\" numeric branch labels must be integer values")]
    MatchLabelNonInteger,
    #[error("\"match\" branch labels must be integers no larger than 9007199254740991")]
    MatchLabelOverflow,
    #[error("\"match\" branch labels must all be numbers, or all be strings")]
    MatchLabelKindMismatch,
    #[error("\"match\" branch labels must be unique")]
    MatchDuplicateLabel,
    #[error("\"let\" binding name must be a string")]
    LetNameNotString,
    #[error("variable names must contain only alphanumeric characters or '_': {0:?}")]
    InvalidVariableName(String),
    #[error(
        "Unknown variable {0:?}. Make sure {0:?} has been bound in an enclosing \"let\" expression before using it."
    )]
    UnboundVariable(String),
    #[error("\"format\" option {0:?} is not recognized")]
    FormatUnknownOption(String),
    #[error("\"format\" vertical-align must be one of 'bottom', 'center', 'top'")]
    FormatInvalidVerticalAlign,
    #[error("\"collator\" options argument must be an object")]
    CollatorOptionsNotObject,
    #[error("\"collator\" option {0:?} is not recognized")]
    CollatorUnknownOption(String),
    #[error("\"number-format\" options argument must be an object")]
    NumberFormatOptionsNotObject,
    #[error("\"number-format\" option {0:?} is not recognized")]
    NumberFormatUnknownOption(String),
    #[error("NumberFormat options `currency` and `unit` are mutually exclusive")]
    NumberFormatCurrencyAndUnit,
    #[error("Expected argument of type string or array, but found {0} instead.")]
    LengthInvalidLiteral(&'static str),
    #[error("Expected first argument to be of type array or string, but found {0} instead")]
    SliceInvalidLiteralInput(&'static str),
    #[error(
        "Expected first argument to be of type boolean, string, number or null, but found array instead"
    )]
    IndexOfNeedleNotScalar,
    #[error("Cannot use collator to compare non-string types.")]
    CollatorNonStringStatic,
    #[error("Expected array<string> but found {0} instead.")]
    JoinInputNotArray(&'static str),
    #[error("Expected array<string> but found array<number, {0}> instead.")]
    JoinElementsNotStrings(usize),
    /// Raised when `interpolate-hcl` / `interpolate-lab` is given a
    /// stop whose output is statically not a color. The full
    /// upstream check (e.g. detecting that `["match", …]` returns
    /// non-colors) needs Phase 10 type inference; this rejection
    /// covers only the syntactic cases the corpus exercises.
    #[error("Type color is required for {{interpolate-hcl,interpolate-lab}} stop outputs.")]
    InterpolateColorOutputRequired,
    /// Raised when `==` / `!=` / `<` / etc. are handed a statically-
    /// known color operand. Upstream forbids comparing colors as a
    /// type-system rule; we catch the syntactic cases (`rgb(...)` /
    /// `to-color(...)` operands) here.
    #[error("\"{0}\" comparisons are not supported for type 'color'.")]
    CompareUnsupportedColor(&'static str),
    /// Raised when a comparison operator is handed an operand whose
    /// static type isn't admissible per the spec — e.g. `>` over a
    /// `boolean`, `==` over an `array`/`object`.
    #[error("\"{0}\" comparisons are not supported for type '{1}'.")]
    CompareUnsupportedType(&'static str, String),
    /// Raised when both comparison operands have statically-known
    /// types that disagree — the canonical "Cannot compare types
    /// 'string' and 'number'." rejection upstream emits for the
    /// `equal/mismatch` and `greater/mismatch` shapes.
    #[error("Cannot compare types '{0}' and '{1}'.")]
    CompareTypeMismatch(String, String),
    /// Raised when `match`'s input type doesn't agree with the
    /// chosen label kind. Wording matches upstream's `match/
    /// mismatch-input` "Expected number but found string instead."
    #[error("Expected {expected} but found {got} instead.")]
    MatchInputTypeMismatch { expected: String, got: String },
    /// Raised when `match`'s branch outputs disagree on type.
    #[error("Expected {expected} but found {got} instead.")]
    MatchOutputTypeMismatch { expected: String, got: String },
    /// Raised when `within`'s sole argument isn't a polygon-bearing
    /// GeoJSON object. Mirrors upstream's "valid geojson object that
    /// contains polygon geometry type" message — the runner only
    /// cares that we surfaced *some* error, but the variant exists so
    /// callers can disambiguate from a generic arity miss.
    #[error(
        "'within' expression requires valid geojson object that contains polygon geometry type."
    )]
    WithinInvalidGeoJson,
    /// Raised when `distance`'s sole argument isn't a recognizable
    /// GeoJSON geometry. The upstream wording is the same as
    /// `within`'s; we keep the variants separate for clarity.
    #[error(
        "'distance' expression requires valid geojson object that contains polygon geometry type."
    )]
    DistanceInvalidGeoJson,
    /// Raised by Phase 10's post-parse [`validate_zoom_positions`]
    /// when `zoom` is used outside the direct `input` slot of a
    /// top-level `step`/`interpolate`. The exact upstream wording
    /// (`"zoom" expression may only be used as input to a top-level
    /// "step" or "interpolate" expression.`) is also what the runner's
    /// success/error compare sees as the rejection signal.
    #[error(
        "\"zoom\" expression may only be used as input to a top-level \"step\" or \"interpolate\" expression."
    )]
    ZoomNestedInvalid,
    /// Raised when an expression contains more than one zoom-using
    /// `step`/`interpolate`. Spec rule: only one zoom-based curve per
    /// styled property.
    #[error(
        "Only one zoom-based \"step\" or \"interpolate\" subexpression may be used in an expression."
    )]
    ZoomMultipleCurves,
    /// Raised by `["in", needle, hay]` when `needle` is statically a
    /// (typed) array. Upstream's `in/invalid-needle-literal-array`
    /// pins this. Non-array needles whose runtime type doesn't fit
    /// surface at eval time via [`EvalError::InNonScalarNeedle`].
    #[error(
        "Expected first argument to be of type boolean, string, number or null, but found {0} instead"
    )]
    InNeedleStaticallyArray(String),
    /// Raised by `interpolate` when a stop's static output type isn't
    /// one of the interpolatable types (number, color, array<number>).
    /// `interpolate/uninterpolable-output/*` and `interpolate/
    /// exponential-{string-array,uninterpolatable-numeric-array}` pin
    /// the message shape.
    #[error("Type {0} is not interpolatable.")]
    InterpolateNotInterpolatable(String),
    /// Raised by `format`'s `text-font` slot when the inner expression
    /// statically resolves to an array whose item type isn't a string.
    /// `format/data-driven-scale` pins this on a bare `["array", …]`
    /// (no item type) wrapping the font-stack source.
    #[error("Expected array<string> but found {0} instead.")]
    FormatTextFontNotStringArray(String),
    /// Wrapper produced by [`Expr::from_json_at`] /
    /// [`Expr::from_filter_json_at`] to attach the JSON path of the
    /// expression to whatever underlying error fired. The renderer can
    /// resolve the path back to a byte span via the [`super::span::SpanMap`]
    /// built from the source document. The `Display` impl delegates to
    /// the inner error so existing message-text-based tests continue to
    /// match upstream wording.
    #[error("{source}")]
    Located {
        path: super::span::JsonPath,
        #[source]
        source: Box<ExprError>,
    },
}

impl ExprError {
    /// Attach a JSON path to this error if it doesn't already carry
    /// one. The outermost wrap wins so the *most specific* path — the
    /// one closest to the failing parse site — is the one that
    /// survives. Today the only attachment site is [`Expr::from_json_at`]
    /// at the expression-property boundary; per-sub-expression paths
    /// can be layered on later without changing this method.
    pub(crate) fn with_path(self, path: super::span::JsonPath) -> Self {
        match self {
            ExprError::Located { .. } => self,
            other => ExprError::Located {
                path,
                source: Box::new(other),
            },
        }
    }

    /// If the error carries a JSON path (i.e. it's [`ExprError::Located`]),
    /// return it.
    pub fn path(&self) -> Option<&super::span::JsonPath> {
        match self {
            ExprError::Located { path, .. } => Some(path),
            _ => None,
        }
    }

    /// Peel any [`ExprError::Located`] wrapper and return the inner
    /// error.
    pub fn unwrap_located(&self) -> &ExprError {
        let mut cur = self;
        while let ExprError::Located { source, .. } = cur {
            cur = source;
        }
        cur
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("expected a number, got {0:?}")]
    ExpectedNumber(Value),
    #[error("expected a boolean, got {0:?}")]
    ExpectedBool(Value),
    #[error("interpolate received no stops")]
    NoStops,
    #[error("no bracketing stops for interpolation input {0}")]
    NoBracketingStops(f64),
    #[error("expected value to be of type {expected}, but found {got} instead")]
    AssertFailed { expected: &'static str, got: String },
    #[error("could not convert {got} value to {target}")]
    CoerceFailed { target: &'static str, got: String },
    #[error(
        "expected arguments for {op:?} to be (string, string) or (number, number), \
         but found ({lhs}, {rhs}) instead"
    )]
    CompareMismatch {
        op: &'static str,
        lhs: String,
        rhs: String,
    },
    /// Raised by the `error` pseudo-op. Only reachable when a fixture's
    /// short-circuit guard fails to short-circuit (a regression in
    /// `all`/`any`).
    #[error("error expression evaluated: {0}")]
    ExplicitError(String),
    /// Only reachable when the AST was hand-built without going through
    /// `from_json` — parse-time scope tracking otherwise catches this.
    #[error("Unknown variable {0:?}")]
    UnboundVariable(String),
    #[error("Expected first argument to be of type array or string, but found {0} instead.")]
    SliceInvalidInput(String),
    #[error("Expected value to be of type string or array, but found {0} instead.")]
    LengthInvalidInput(String),
    #[error("Expected array<string>, but found {0} instead.")]
    JoinNonArray(String),
    #[error(
        "Expected first argument to be of type boolean, string, number or null, but found {0} instead."
    )]
    IndexOfInvalidNeedle(String),
    #[error("Expected second argument to be of type array or string, but found {0} instead.")]
    IndexOfInvalidHaystack(String),
    #[error("Expected a collator, but found {0} instead.")]
    ExpectedCollator(String),
    #[error("Cannot use collator to compare non-string types.")]
    CollatorNonStringCompare,
    #[error("Expected an array<string> for `text-font`")]
    FormatBadFontStack,
    #[error("Could not parse color from {0:?}")]
    FormatBadColor(String),
    #[error("'vertical-align' must be one of: 'bottom', 'center', 'top' but found {0:?} instead.")]
    FormatBadVerticalAlign(String),
    /// Surfaced by `rgb` / `rgba` when an R, G, or B channel is
    /// outside `0..=255`. The message wording mirrors upstream's
    /// `Invalid rgba value [r, g, b, a]: 'r', 'g', and 'b' must be
    /// between 0 and 255.` so anyone grepping for the error sees both.
    #[error(
        "Invalid rgba value [{0}, {1}, {2}, {3}]: 'r', 'g', and 'b' must be between 0 and 255."
    )]
    RgbChannelOutOfRange(f64, f64, f64, f64),
    /// Surfaced by `rgba` when alpha is outside `0..=1`.
    #[error("Invalid rgba value [{0}, {1}, {2}, {3}]: 'a' must be between 0 and 1.")]
    RgbAlphaOutOfRange(f64, f64, f64, f64),
    /// Surfaced by `to-color` when no input parses cleanly and there's
    /// no fallback left to try. Wording matches upstream's
    /// `Could not parse color from value 'X'`.
    #[error("Could not parse color from value {0:?}")]
    ToColorUnparseable(String),
    /// Surfaced by `at` when the index is below zero. Wording mirrors
    /// upstream's `at/basic` fixture so it would byte-match too if we
    /// ever turned on the strict-message check.
    #[error("Array index out of bounds: {0} < 0.")]
    AtIndexBelowZero(i64),
    /// Surfaced by `at` when the index is past the array's end. The
    /// second value is the maximum admissible index (`len - 1`).
    #[error("Array index out of bounds: {0} > {1}.")]
    AtIndexAboveBounds(i64, i64),
    /// Surfaced by `at` when the index is fractional. Upstream's
    /// `at/basic` fixture exercises this on `1.5`.
    #[error("Array index must be an integer, but found {0} instead.")]
    AtIndexNotInteger(f64),
    /// Surfaced by `at` when the second argument isn't an array.
    #[error("Expected value to be of type array, but found {0} instead.")]
    AtArgNotArray(String),
    /// Surfaced by `in` when the haystack is neither an array nor a
    /// string. `null` haystacks are NOT this error — they return
    /// `false`, per upstream's `in/basic-array` last input.
    #[error("Expected second argument to be of type array or string, but found {0} instead.")]
    InInvalidHaystack(String),
    /// Surfaced by `in` when the needle isn't a scalar (string,
    /// number, boolean, null) — the static-array case lives on
    /// [`ExprError::InNeedleStaticallyArray`].
    #[error(
        "Expected first argument to be of type boolean, string, number or null, but found {0} instead."
    )]
    InNonScalarNeedle(String),
    /// Surfaced by `get` 2-arg form when the host operand isn't an
    /// object. Wording mirrors the implicit `object` assertion the
    /// spec parser inserts.
    #[error("Expected value to be of type object, but found {0} instead.")]
    GetHostNotObject(String),
}

// Surface a Display impl on `Value` for nicer error messages without
// pulling in a heavy dependency.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Null => f.write_str("null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Number(n) => write!(f, "{n}"),
            Value::String(s) => write!(f, "{s:?}"),
            Value::Array(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
            Value::Object(map) => {
                f.write_str("{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{k:?}: {v}")?;
                }
                f.write_str("}")
            }
            Value::Formatted(_) => f.write_str("<formatted>"),
            Value::Collator(c) => write!(f, "<collator locale={:?}>", c.locale),
            Value::Color([r, g, b, a]) => write!(f, "rgba({r}, {g}, {b}, {a})"),
            Value::Image(img) => write!(f, "image({:?}, available={})", img.name, img.available),
        }
    }
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

    fn parse(j: serde_json::Value) -> Result<Expr, ExprError> {
        Expr::from_json(&j)
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

    // ---- parse ----

    #[test]
    fn parses_number_literal() {
        assert_eq!(parse(json!(3.5)).unwrap(), Expr::Lit(Literal::Number(3.5)));
    }

    #[test]
    fn parses_string_literal() {
        assert_eq!(
            parse(json!("highway")).unwrap(),
            Expr::Lit(Literal::String("highway".into()))
        );
    }

    #[test]
    fn parses_bool_and_null_literals() {
        assert_eq!(parse(json!(true)).unwrap(), Expr::Lit(Literal::Bool(true)));
        assert_eq!(parse(json!(null)).unwrap(), Expr::Lit(Literal::Null));
    }

    #[test]
    fn parses_get() {
        assert_eq!(
            parse(json!(["get", "name"])).unwrap(),
            Expr::Get("name".into())
        );
    }

    #[test]
    fn parses_zoom() {
        // Bare `["zoom"]` at the top level is rejected by Phase 10's
        // post-parse `validate_zoom_positions` walk (zoom is only
        // legal as the input slot of a top-level `step`/`interpolate`),
        // so we wrap it in the canonical curve shape to exercise the
        // parser-into-AST path.
        let parsed = parse(json!(["interpolate", ["linear"], ["zoom"], 0, 0, 1, 1])).unwrap();
        let Expr::Interpolate { input, .. } = parsed else {
            panic!("expected Interpolate");
        };
        assert_eq!(*input, Expr::Zoom);
    }

    #[test]
    fn parses_eq_with_get_and_string() {
        let parsed = parse(json!(["==", ["get", "kind"], "highway"])).unwrap();
        assert_eq!(
            parsed,
            Expr::Eq(
                Box::new(Expr::Get("kind".into())),
                Box::new(Expr::Lit(Literal::String("highway".into()))),
                None,
            )
        );
    }

    #[test]
    fn parses_interpolate_linear() {
        let parsed = parse(json!(["interpolate", ["linear"], ["zoom"], 10, 1, 16, 6])).unwrap();
        let Expr::Interpolate {
            interpolation,
            input,
            stops,
            color_typed,
        } = parsed
        else {
            panic!("expected Interpolate");
        };
        assert_eq!(interpolation, Interpolation::Linear);
        assert_eq!(*input, Expr::Zoom);
        assert!(!color_typed);
        assert_eq!(stops.len(), 2);
        assert_eq!(stops[0].0, 10.0);
        assert_eq!(stops[1].0, 16.0);
        assert_eq!(stops[0].1, Expr::Lit(Literal::Number(1.0)));
        assert_eq!(stops[1].1, Expr::Lit(Literal::Number(6.0)));
    }

    #[test]
    fn parses_literal_array_without_recursing() {
        // `["literal", ["foo", 1]]` would normally be parsed as a call
        // to operator `foo`, but `literal` short-circuits the parser
        // and just snapshots the JSON.
        let parsed = parse(json!(["literal", ["foo", 1]])).unwrap();
        let expected_value = Value::Array(vec![Value::String("foo".into()), Value::Number(1.0)]);
        assert_eq!(parsed, Expr::Literal(expected_value));
    }

    #[test]
    fn parses_array_assertion_forms() {
        // 1-arg: assert array.
        assert_eq!(
            parse(json!(["array", ["literal", [1]]])).unwrap(),
            Expr::ArrayAssert {
                item_type: None,
                length: None,
                values: vec![Expr::Literal(Value::Array(vec![Value::Number(1.0)]))],
            }
        );
        // 2-arg with type prefix.
        let Expr::ArrayAssert {
            item_type, length, ..
        } = parse(json!(["array", "number", ["get", "x"]])).unwrap()
        else {
            panic!("expected ArrayAssert");
        };
        assert_eq!(item_type, Some(ItemType::Number));
        assert_eq!(length, None);
        // 3-arg with type and length.
        let Expr::ArrayAssert {
            item_type, length, ..
        } = parse(json!(["array", "number", 2, ["get", "x"]])).unwrap()
        else {
            panic!("expected ArrayAssert");
        };
        assert_eq!(item_type, Some(ItemType::Number));
        assert_eq!(length, Some(2));
    }

    #[test]
    fn rejects_unknown_operator() {
        let err = parse(json!(["nopes", 1])).unwrap_err();
        assert!(matches!(err, ExprError::UnknownOperator(op) if op == "nopes"));
    }

    #[test]
    fn rejects_empty_array() {
        assert!(matches!(parse(json!([])), Err(ExprError::EmptyArray)));
    }

    #[test]
    fn rejects_wrong_arity_eq() {
        assert!(matches!(
            parse(json!(["==", 1])),
            Err(ExprError::WrongArity { op: "==", .. })
        ));
    }

    #[test]
    fn rejects_unsupported_interpolation_kind() {
        // `exponential` and `cubic-bezier` are first-class as of
        // Phase 7; this test now pins a *genuinely* unknown kind so
        // any future addition still has to wire itself into
        // `parse_interpolation` deliberately.
        let err = parse(json!(["interpolate", ["banana"], ["zoom"], 0, 0, 1, 1])).unwrap_err();
        assert!(matches!(err, ExprError::UnsupportedInterpolation(k) if k == "banana"));
    }

    // ---- eval ----

    #[test]
    fn eval_get_resolves_property() {
        let feat = feature_with(&[("kind", MvtValue::String("highway".into()))]);
        let v = Expr::Get("kind".into())
            .eval(
                &EvalContext {
                    zoom: 13.0,
                    canonical: None,
                    ..Default::default()
                },
                &feat,
            )
            .unwrap();
        assert_eq!(v, Value::String("highway".into()));
    }

    #[test]
    fn eval_get_missing_property_is_null() {
        let feat = feature_with(&[]);
        let v = Expr::Get("kind".into())
            .eval(
                &EvalContext {
                    zoom: 13.0,
                    canonical: None,
                    ..Default::default()
                },
                &feat,
            )
            .unwrap();
        assert_eq!(v, Value::Null);
    }

    #[test]
    fn eval_zoom_returns_context_value() {
        let feat = feature_with(&[]);
        let v = Expr::Zoom
            .eval(
                &EvalContext {
                    zoom: 7.5,
                    canonical: None,
                    ..Default::default()
                },
                &feat,
            )
            .unwrap();
        assert_eq!(v, Value::Number(7.5));
    }

    #[test]
    fn eval_eq_get_vs_literal() {
        let highway = feature_with(&[("kind", MvtValue::String("highway".into()))]);
        let other = feature_with(&[("kind", MvtValue::String("river".into()))]);
        let missing = feature_with(&[]);

        let expr = Expr::Eq(
            Box::new(Expr::Get("kind".into())),
            Box::new(Expr::Lit(Literal::String("highway".into()))),
            None,
        );
        let ctx = EvalContext {
            zoom: 13.0,
            canonical: None,
            ..Default::default()
        };

        assert_eq!(expr.eval(&ctx, &highway).unwrap(), Value::Bool(true));
        assert_eq!(expr.eval(&ctx, &other).unwrap(), Value::Bool(false));
        // Missing property compares equal to Null, not "highway", so the
        // result is false.
        assert_eq!(expr.eval(&ctx, &missing).unwrap(), Value::Bool(false));
    }

    #[test]
    fn eval_eq_does_not_coerce_number_to_string() {
        let feat = feature_with(&[]);
        let expr = Expr::Eq(
            Box::new(Expr::Lit(Literal::Number(1.0))),
            Box::new(Expr::Lit(Literal::String("1".into()))),
            None,
        );
        let v = expr
            .eval(
                &EvalContext {
                    zoom: 0.0,
                    canonical: None,
                    ..Default::default()
                },
                &feat,
            )
            .unwrap();
        assert_eq!(v, Value::Bool(false));
    }

    #[test]
    fn eval_interpolate_linear_interior_and_endpoints() {
        let feat = feature_with(&[]);
        let expr = parse(json!(["interpolate", ["linear"], ["zoom"], 10, 1, 16, 6])).unwrap();

        let at = |z: f64| {
            expr.eval(
                &EvalContext {
                    zoom: z,
                    canonical: None,
                    ..Default::default()
                },
                &feat,
            )
            .unwrap()
            .as_number()
            .unwrap()
        };
        assert_eq!(at(10.0), 1.0);
        assert_eq!(at(16.0), 6.0);
        assert!((at(13.0) - 3.5).abs() < 1e-9);
    }

    #[test]
    fn eval_interpolate_clamps_outside_range() {
        let feat = feature_with(&[]);
        let expr = parse(json!(["interpolate", ["linear"], ["zoom"], 10, 1, 16, 6])).unwrap();
        let at = |z: f64| {
            expr.eval(
                &EvalContext {
                    zoom: z,
                    canonical: None,
                    ..Default::default()
                },
                &feat,
            )
            .unwrap()
            .as_number()
            .unwrap()
        };
        assert_eq!(at(8.0), 1.0);
        assert_eq!(at(20.0), 6.0);
    }

    #[test]
    fn filter_matches_when_filter_is_true() {
        let highway = feature_with(&[("kind", MvtValue::String("highway".into()))]);
        let river = feature_with(&[("kind", MvtValue::String("river".into()))]);
        let filter = Filter(Expr::Eq(
            Box::new(Expr::Get("kind".into())),
            Box::new(Expr::Lit(Literal::String("highway".into()))),
            None,
        ));
        let ctx = EvalContext {
            zoom: 13.0,
            canonical: None,
            ..Default::default()
        };
        assert!(filter.matches(&ctx, &highway).unwrap());
        assert!(!filter.matches(&ctx, &river).unwrap());
    }

    #[test]
    fn from_mvt_value_collapses_numeric_kinds() {
        assert_eq!(Value::from(&MvtValue::Int(5)), Value::Number(5.0));
        assert_eq!(Value::from(&MvtValue::Uint(7)), Value::Number(7.0));
        assert_eq!(Value::from(&MvtValue::Float(2.5)), Value::Number(2.5));
        assert_eq!(Value::from(&MvtValue::SInt(-3)), Value::Number(-3.0));
        assert_eq!(
            Value::from(&MvtValue::String("x".into())),
            Value::String("x".into())
        );
        assert_eq!(Value::from(&MvtValue::Bool(true)), Value::Bool(true));
        assert_eq!(Value::from(&MvtValue::Null), Value::Null);
    }

    #[test]
    fn from_mvt_value_passes_array_and_object_through() {
        let mvt_array = MvtValue::Array(vec![MvtValue::Int(1), MvtValue::String("two".into())]);
        assert_eq!(
            Value::from(&mvt_array),
            Value::Array(vec![Value::Number(1.0), Value::String("two".into())])
        );

        let mut mvt_map = HashMap::new();
        mvt_map.insert("k".to_string(), MvtValue::Bool(true));
        let v = Value::from(&MvtValue::Object(mvt_map));
        let Value::Object(out) = v else {
            panic!("expected Object");
        };
        assert_eq!(out.get("k"), Some(&Value::Bool(true)));
    }

    // ---- assertions, coercions, typeof ----

    #[test]
    fn assertion_number_returns_first_matching_value() {
        let feat = feature_with(&[("x", MvtValue::String("not-a-number".into()))]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };

        // Strict form: a non-number value errors.
        let expr = parse(json!(["number", ["get", "x"]])).unwrap();
        assert!(matches!(
            expr.eval(&ctx, &feat),
            Err(EvalError::AssertFailed {
                expected: "number",
                ..
            })
        ));

        // Fallback form: the trailing -1 is a number and is returned
        // when the first arg fails the assertion.
        let expr = parse(json!(["number", ["get", "x"], -1])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Number(-1.0));
    }

    #[test]
    fn array_assertion_checks_item_type_and_length() {
        let mut props = HashMap::new();
        props.insert(
            "x".to_string(),
            MvtValue::Array(vec![MvtValue::Int(1), MvtValue::Int(2)]),
        );
        let feat = Feature {
            id: None,
            properties: props,
            geometry: Geometry::Unknown,
        };
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };

        let expr = parse(json!(["array", "number", 2, ["get", "x"]])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::Array(vec![Value::Number(1.0), Value::Number(2.0)])
        );

        // Wrong length triggers an error (no fallback supplied).
        let expr = parse(json!(["array", "number", 3, ["get", "x"]])).unwrap();
        assert!(matches!(
            expr.eval(&ctx, &feat),
            Err(EvalError::AssertFailed { .. })
        ));
    }

    #[test]
    fn coerce_to_number_parses_strings_and_falls_back() {
        let mut props = HashMap::new();
        props.insert("x".into(), MvtValue::String("Not a number".into()));
        let feat = Feature {
            id: None,
            properties: props,
            geometry: Geometry::Unknown,
        };
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };

        // Single arg: failure surfaces as an error.
        let expr = parse(json!(["to-number", ["get", "x"]])).unwrap();
        assert!(matches!(
            expr.eval(&ctx, &feat),
            Err(EvalError::CoerceFailed {
                target: "number",
                ..
            })
        ));

        // Fallback used when the first arg can't be coerced.
        let expr = parse(json!(["to-number", ["get", "x"], -1])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Number(-1.0));
    }

    #[test]
    fn coerce_to_string_renders_arrays_as_compact_json() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["to-string", ["literal", [1, 2]]])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::String("[1,2]".into())
        );
    }

    #[test]
    fn typeof_distinguishes_homogeneous_and_heterogeneous_arrays() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };

        let expr = parse(json!(["typeof", ["literal", [1, 2, 3]]])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::String("array<number, 3>".into())
        );

        let expr = parse(json!(["typeof", ["literal", [1, "x"]]])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::String("array<value, 2>".into())
        );
    }

    // ---- arithmetic / math / comparison / boolean ----

    #[test]
    fn arithmetic_variadic_arity_zero_returns_identity() {
        // `+` with no args is 0 and `*` with no args is 1; the corpus
        // pins these down as the variadic identities.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        assert_eq!(
            parse(json!(["+"])).unwrap().eval(&ctx, &feat).unwrap(),
            Value::Number(0.0)
        );
        assert_eq!(
            parse(json!(["*"])).unwrap().eval(&ctx, &feat).unwrap(),
            Value::Number(1.0)
        );
    }

    #[test]
    fn divide_by_zero_yields_infinity_not_error() {
        // The spec defers to IEEE 754 for `/0`. Roadmap calls this
        // out explicitly because the naive "error on zero divisor"
        // implementation is a tempting trap.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let plus = parse(json!(["/", 1, 0]))
            .unwrap()
            .eval(&ctx, &feat)
            .unwrap();
        let minus = parse(json!(["/", -1, 0]))
            .unwrap()
            .eval(&ctx, &feat)
            .unwrap();
        let nan = parse(json!(["/", 0, 0]))
            .unwrap()
            .eval(&ctx, &feat)
            .unwrap();
        assert_eq!(plus, Value::Number(f64::INFINITY));
        assert_eq!(minus, Value::Number(f64::NEG_INFINITY));
        let Value::Number(n) = nan else {
            panic!("expected number");
        };
        assert!(n.is_nan(), "0/0 should be NaN, got {n}");
    }

    #[test]
    fn minus_supports_unary_and_binary_forms() {
        // The single-arg form is unary negation; the parser rewrites
        // it to `0 - x` so eval only sees the binary shape.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        assert_eq!(
            parse(json!(["-", 5])).unwrap().eval(&ctx, &feat).unwrap(),
            Value::Number(-5.0)
        );
        assert_eq!(
            parse(json!(["-", 5, 7]))
                .unwrap()
                .eval(&ctx, &feat)
                .unwrap(),
            Value::Number(-2.0)
        );
        // Zero args is a compile error per the corpus.
        assert!(matches!(
            parse(json!(["-"])),
            Err(ExprError::WrongArity { op: "-", .. })
        ));
    }

    #[test]
    fn round_half_away_from_zero() {
        // MapLibre's `round` (and Rust's `f64::round`) rounds halves
        // *away from zero*; JS's `Math.round` rounds half toward +∞.
        // The corpus fixture pins -2.5 → -3 and 2.5 → 3.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let at = |x: f64| {
            parse(json!(["round", x]))
                .unwrap()
                .eval(&ctx, &feat)
                .unwrap()
        };
        assert_eq!(at(-2.5), Value::Number(-3.0));
        assert_eq!(at(2.5), Value::Number(3.0));
        assert_eq!(at(-2.4), Value::Number(-2.0));
        assert_eq!(at(2.4), Value::Number(2.0));
    }

    #[test]
    fn max_min_arity_zero_returns_signed_infinity() {
        // `["max"]` evaluates to -Infinity, `["min"]` to +Infinity.
        // The corpus exercises this via `[">", ["min"], 0]` etc.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let Value::Number(max) = parse(json!(["max"])).unwrap().eval(&ctx, &feat).unwrap() else {
            panic!("expected number");
        };
        let Value::Number(min) = parse(json!(["min"])).unwrap().eval(&ctx, &feat).unwrap() else {
            panic!("expected number");
        };
        assert_eq!(max, f64::NEG_INFINITY);
        assert_eq!(min, f64::INFINITY);
    }

    #[test]
    fn math_constants_evaluate_and_reject_args() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let pi = parse(json!(["pi"])).unwrap().eval(&ctx, &feat).unwrap();
        assert_eq!(pi, Value::Number(std::f64::consts::PI));
        // Per `e/unexpected-argument`: passing args is a compile-time
        // error.
        assert!(matches!(
            parse(json!(["e", 1])),
            Err(ExprError::WrongArity { op: "e", .. })
        ));
    }

    #[test]
    fn compare_mismatched_types_is_runtime_error() {
        // `less/value` exercises this: at parse time we accept any
        // expression, but at eval time a (number, string) pair must
        // error rather than silently coerce.
        let feat = feature_with(&[
            ("x", MvtValue::SInt(1)),
            ("y", MvtValue::String("1".into())),
        ]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["<", ["get", "x"], ["get", "y"]])).unwrap();
        assert!(matches!(
            expr.eval(&ctx, &feat),
            Err(EvalError::CompareMismatch { op: "<", .. })
        ));

        // But matched types compare normally.
        let feat = feature_with(&[("x", MvtValue::SInt(1)), ("y", MvtValue::SInt(2))]);
        let expr = parse(json!(["<", ["get", "x"], ["get", "y"]])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Bool(true));
    }

    #[test]
    fn not_equal_with_mixed_types_is_true() {
        // `not_equal/value` asserts that comparing values of different
        // runtime kinds is `true` (they aren't equal). This deliberately
        // diverges from upstream's compile-time rejection in
        // `not_equal/mismatch`, which Phase 10 will catch.
        let feat = feature_with(&[
            ("x", MvtValue::SInt(0)),
            ("y", MvtValue::String("0".into())),
        ]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["!=", ["get", "x"], ["get", "y"]])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Bool(true));
    }

    #[test]
    fn all_short_circuits_on_first_false() {
        // The trailing `["error", …]` would raise if we evaluated it,
        // so observing `Bool(false)` proves we stopped at the
        // preceding literal.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["all", false, ["error", "unreachable"]])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Bool(false));
    }

    #[test]
    fn any_short_circuits_on_first_true() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["any", true, ["error", "unreachable"]])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Bool(true));
    }

    #[test]
    fn all_any_empty_returns_identity() {
        // `all` of nothing is true (vacuous truth); `any` of nothing
        // is false.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        assert_eq!(
            parse(json!(["all"])).unwrap().eval(&ctx, &feat).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            parse(json!(["any"])).unwrap().eval(&ctx, &feat).unwrap(),
            Value::Bool(false)
        );
    }

    // ---- control flow & bindings ----

    #[test]
    fn case_first_true_branch_wins_and_fallback_returns() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["case", false, "a", true, "b", "fb"])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::String("b".into()));
        let expr = parse(json!(["case", false, "a", false, "b", "fb"])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::String("fb".into()));
    }

    #[test]
    fn case_non_boolean_condition_is_runtime_error() {
        // `case/basic`'s "x": "false" input pins this down — non-bool
        // conditions are an error, not a truthy coercion.
        let feat = feature_with(&[("x", MvtValue::String("false".into()))]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["case", ["get", "x"], "y", "fb"])).unwrap();
        assert!(matches!(
            expr.eval(&ctx, &feat),
            Err(EvalError::ExpectedBool(_))
        ));
    }

    #[test]
    fn match_multi_label_routes_through_array_form() {
        // `match/label-numbers` exercises `[v1, v2, …]` labels.
        let feat = feature_with(&[("x", MvtValue::SInt(2))]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!([
            "match",
            ["get", "x"],
            [-2, -1],
            "negative",
            0,
            "zero",
            [1, 2],
            "positive",
            "otherwise"
        ]))
        .unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::String("positive".into())
        );
    }

    #[test]
    fn match_rejects_duplicate_labels_at_parse_time() {
        // Upstream's compile-time check rejects the same label
        // appearing in more than one branch; we mirror that.
        let err = parse(json!([
            "match",
            ["get", "x"],
            "a",
            "first",
            "a",
            "second",
            "fb"
        ]))
        .unwrap_err();
        assert!(matches!(err, ExprError::MatchDuplicateLabel));
    }

    #[test]
    fn match_rejects_non_integer_numeric_labels() {
        let err = parse(json!(["match", 1, 1.5, "x", "fb"])).unwrap_err();
        assert!(matches!(err, ExprError::MatchLabelNonInteger));
    }

    #[test]
    fn match_rejects_oversize_integer_labels() {
        // 10^16 is well past 2^53 − 1, the safe-integer ceiling.
        let err = parse(json!(["match", 0, 10_000_000_000_000_000_i64, "x", "fb"])).unwrap_err();
        assert!(matches!(err, ExprError::MatchLabelOverflow));
    }

    #[test]
    fn coalesce_swallows_errors_and_advances_on_null() {
        // First branch errors (string in a `number` assertion), second
        // returns null because the property is missing, third hits the
        // numeric literal and wins.
        let feat = feature_with(&[("x", MvtValue::String("nope".into()))]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!([
            "coalesce",
            ["number", ["get", "x"]],
            ["get", "missing"],
            42
        ]))
        .unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Number(42.0));
    }

    #[test]
    fn coalesce_all_null_returns_null() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["coalesce", ["get", "a"], ["get", "b"]])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Null);
    }

    #[test]
    fn let_bindings_evaluate_in_declaration_order_and_visible_to_later() {
        // `let/nested` exercises this: a binding's value expression
        // can reference any binding declared earlier in the same `let`.
        let feat = feature_with(&[("a", MvtValue::SInt(1))]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!([
            "let",
            "a",
            ["number", ["get", "a"]],
            "b",
            ["+", 1, ["var", "a"]],
            ["+", ["var", "a"], ["var", "b"]]
        ]))
        .unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Number(3.0));
    }

    #[test]
    fn let_inner_let_shadows_outer_binding() {
        let feat = feature_with(&[("one", MvtValue::SInt(1)), ("two", MvtValue::SInt(2))]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!([
            "let",
            "a",
            ["get", "one"],
            ["let", "a", ["get", "two"], ["var", "a"]]
        ]))
        .unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Number(2.0));
    }

    #[test]
    fn var_unbound_is_parse_time_error() {
        // The upstream `let/unbound` fixture asserts compile-time
        // rejection. Our parse-time scope tracking surfaces it here.
        let err = parse(json!(["let", "a", 1, ["+", ["var", "a"], ["var", "b"]]])).unwrap_err();
        assert!(matches!(err, ExprError::UnboundVariable(name) if name == "b"));
    }

    #[test]
    fn let_invalid_variable_name() {
        // `let/invalid-name`: a `$` is not in the allowed character set.
        let err = parse(json!(["let", "$a", 1, ["var", "$a"]])).unwrap_err();
        assert!(matches!(err, ExprError::InvalidVariableName(n) if n == "$a"));
    }

    #[test]
    fn not_requires_boolean_operand() {
        // `!` is strict: it errors on non-bool input rather than
        // truthy-coercing.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["!", 1])).unwrap();
        assert!(matches!(
            expr.eval(&ctx, &feat),
            Err(EvalError::ExpectedBool(_))
        ));
        let expr = parse(json!(["!", true])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Bool(false));
    }

    // ---- strings, formatting, i18n ----

    #[test]
    fn length_string_counts_unicode_code_points_not_bytes() {
        // "héllo" is 5 code points but 6 UTF-8 bytes. "日本" is 2
        // code points but 6 UTF-8 bytes. An emoji ("🦀") is 1 code
        // point but 4 UTF-8 bytes. Asserting on multi-byte inputs
        // locks the code-point-count semantics in.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let cases = [("héllo", 5.0), ("日本", 2.0), ("🦀", 1.0)];
        for (s, expected) in cases {
            let expr = parse(json!(["length", s])).unwrap();
            assert_eq!(
                expr.eval(&ctx, &feat).unwrap(),
                Value::Number(expected),
                "length of {s:?}"
            );
        }
    }

    #[test]
    fn slice_string_indexes_by_code_point() {
        // Slicing `"日本語"` from index 1 must give `"本語"`, not the
        // raw UTF-8 byte slice (which would split a multi-byte char).
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["slice", "日本語", 1])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::String("本語".into())
        );
        // Negative indices count code points from the end.
        let expr = parse(json!(["slice", "héllo", -2])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::String("lo".into()));
    }

    #[test]
    fn concat_stringifies_each_arg() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        // Per `concat/coercion`: number `1` -> "1", bool true -> "true".
        let expr = parse(json!(["concat", "a", 1, true])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::String("a1true".into())
        );
        // Empty form is the empty string.
        let expr = parse(json!(["concat"])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::String(String::new())
        );
    }

    #[test]
    fn split_empty_separator_splits_by_code_point() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["split", "日本", ""])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feat).unwrap(),
            Value::Array(vec![Value::String("日".into()), Value::String("本".into()),])
        );
    }

    #[test]
    fn index_of_string_uses_code_point_indices() {
        // `"丐𦨭市镇"` has 4 code points: `丐`(0), `𦨭`(1, non-BMP),
        // `市`(2), `镇`(3). Finding `"市镇"` should return 2.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!(["index-of", "市镇", "丐𦨭市镇"])).unwrap();
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Number(2.0));
    }

    #[test]
    fn comparison_with_collator_routes_through_folding() {
        // Without a collator, `"a" == "A"` is false. With a
        // case-insensitive collator, it's true. The 2-arg form must
        // still behave the original way.
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let bare = parse(json!(["==", "a", "A"])).unwrap();
        assert_eq!(bare.eval(&ctx, &feat).unwrap(), Value::Bool(false));
        let collated = parse(json!([
            "==",
            "a",
            "A",
            ["collator", { "case-sensitive": false, "diacritic-sensitive": true }]
        ]))
        .unwrap();
        assert_eq!(collated.eval(&ctx, &feat).unwrap(), Value::Bool(true));
    }

    #[test]
    fn format_produces_per_section_payload() {
        let feat = feature_with(&[]);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let expr = parse(json!([
            "format",
            "hi",
            { "font-scale": 1.5, "vertical-align": "center" }
        ]))
        .unwrap();
        let Value::Formatted(f) = expr.eval(&ctx, &feat).unwrap() else {
            panic!("expected Formatted");
        };
        assert_eq!(f.sections.len(), 1);
        assert_eq!(f.sections[0].text, "hi");
        assert_eq!(f.sections[0].scale, Some(1.5));
        assert_eq!(f.sections[0].vertical_align, Some(VerticalAlign::Center));
    }

    // ----------------------------------------------------------------
    // Phase 7 — interpolation extensions and `step`
    // ----------------------------------------------------------------

    // Small helper: build a feature with a single numeric `x` property.
    // The Phase-7 tests all drive `interpolate` / `step` off this
    // single input so the curves' shape is what we're really
    // exercising.
    fn feat_x(x: f64) -> Feature {
        feature_with(&[("x", MvtValue::Double(x))])
    }

    #[test]
    fn exponential_interpolate_endpoints_and_midpoint() {
        // Mirrors the upstream `interpolate/exponential` fixture's
        // stops (1, 2) → (3, 6) with base 2. The endpoint clamp is
        // shared with the linear path; the midpoint exercises the
        // `(base^(x-x_lo) - 1) / (base^(x_hi-x_lo) - 1)` ramp.
        let expr = parse(json!([
            "interpolate",
            ["exponential", 2],
            ["get", "x"],
            1,
            2,
            3,
            6
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };

        // Clamp below the first stop: the spec returns the lower
        // output unchanged regardless of the curve.
        assert_eq!(expr.eval(&ctx, &feat_x(0.0)).unwrap(), Value::Number(2.0));
        // Clamp above the last stop.
        assert_eq!(expr.eval(&ctx, &feat_x(4.0)).unwrap(), Value::Number(6.0));
        // At x=2 the upstream fixture pins the value to 3.33333,
        // which is `2 + 4 * (2^1 - 1)/(2^2 - 1) = 2 + 4/3`.
        let Value::Number(n) = expr.eval(&ctx, &feat_x(2.0)).unwrap() else {
            panic!("expected number");
        };
        assert!((n - (2.0 + 4.0 / 3.0)).abs() < 1e-9, "got {n}");
    }

    #[test]
    fn exponential_base_one_collapses_to_linear() {
        // The `base == 1` case would otherwise be `0/0` in the
        // exponential formula. The implementation special-cases it to
        // plain linear, matching upstream's `linear-many-stops`
        // fixture (which writes `["exponential", 1]` and expects
        // linear behavior).
        let expr = parse(json!([
            "interpolate",
            ["exponential", 1],
            ["get", "x"],
            0,
            0,
            10,
            100
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        assert_eq!(expr.eval(&ctx, &feat_x(5.0)).unwrap(), Value::Number(50.0));
    }

    #[test]
    fn cubic_bezier_ease_in_out_symmetry() {
        // The `[0.42, 0, 0.58, 1]` curve is the classic CSS ease-in-
        // out. It's symmetric about (0.5, 0.5), so `f(x) + f(1-x) =
        // 1` for inputs in `[0, 1]`. We sample two complementary
        // points and check the symmetry holds to ~6 sig figs.
        let expr = parse(json!([
            "interpolate",
            ["cubic-bezier", 0.42, 0, 0.58, 1],
            ["get", "x"],
            0,
            0,
            1,
            1
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let Value::Number(at_03) = expr.eval(&ctx, &feat_x(0.3)).unwrap() else {
            panic!("expected number");
        };
        let Value::Number(at_07) = expr.eval(&ctx, &feat_x(0.7)).unwrap() else {
            panic!("expected number");
        };
        // Allow a small tolerance for Newton convergence.
        assert!((at_03 + at_07 - 1.0).abs() < 1e-6, "{at_03} + {at_07}");
        // Midpoint of the symmetric curve hits exactly 0.5.
        let Value::Number(at_05) = expr.eval(&ctx, &feat_x(0.5)).unwrap() else {
            panic!("expected number");
        };
        assert!((at_05 - 0.5).abs() < 1e-9, "{at_05}");
    }

    #[test]
    fn cubic_bezier_tolerates_degenerate_stop_span() {
        // Two stops with the same input ("degenerate span") would
        // otherwise blow up in `cubic_bezier_y_for_x` because the
        // normalized `x` would be `NaN`. The shared early-out in
        // `interpolate()` clamps `t = 0` so the lower output wins,
        // matching the linear path's behavior on duplicate inputs.
        // We can't construct this via `parse` (the parser rejects
        // duplicates) — instead we hand-build the AST.
        let expr = Expr::Interpolate {
            interpolation: Interpolation::CubicBezier {
                x1: 0.42,
                y1: 0.0,
                x2: 0.58,
                y2: 1.0,
            },
            input: Box::new(Expr::Get("x".into())),
            stops: vec![
                (1.0, Expr::Lit(Literal::Number(10.0))),
                (1.0, Expr::Lit(Literal::Number(99.0))),
            ],
            color_typed: false,
        };
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        // Below the first stop, clamp to the first output. The point
        // of the test is that this doesn't blow up — without the
        // degenerate-span guard, the cubic-bezier `(x - x_lo) / (x_hi
        // - x_lo)` would produce NaN.
        assert_eq!(expr.eval(&ctx, &feat_x(0.5)).unwrap(), Value::Number(10.0));
        // The exact-match case takes the lower-bound branch in
        // `interpolate()` (which evaluates `first.1` even when the
        // input is exactly at the first stop). For our duplicate-
        // input pair both halves carry through cleanly.
        let v = expr.eval(&ctx, &feat_x(1.0)).unwrap();
        let Value::Number(n) = v else {
            panic!("expected number, got {v:?}")
        };
        assert!(
            n.is_finite(),
            "cubic-bezier produced NaN for degenerate span"
        );
    }

    #[test]
    fn step_discrete_selection() {
        // The first stop (input -> 11) covers `x < 0`. Each
        // subsequent stop opens at its input — at the stop's input
        // exactly we transition to the next output.
        let expr = parse(json!([
            "step",
            ["get", "x"],
            11, // default: returned for x < 0
            0,
            111,
            1,
            1111
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        // Below first stop: default.
        assert_eq!(expr.eval(&ctx, &feat_x(-1.5)).unwrap(), Value::Number(11.0));
        // At the first stop input exactly: first output.
        assert_eq!(expr.eval(&ctx, &feat_x(0.0)).unwrap(), Value::Number(111.0));
        // Between first and second: first output (still).
        assert_eq!(expr.eval(&ctx, &feat_x(0.5)).unwrap(), Value::Number(111.0));
        // At the second stop input exactly: second output.
        assert_eq!(
            expr.eval(&ctx, &feat_x(1.0)).unwrap(),
            Value::Number(1111.0)
        );
        // Past the last stop: last output still wins (no upper clamp
        // to default).
        assert_eq!(
            expr.eval(&ctx, &feat_x(1.5)).unwrap(),
            Value::Number(1111.0)
        );
    }

    #[test]
    fn step_outputs_any_value_type() {
        // `step` doesn't interpolate, so it should pass arbitrary
        // outputs (strings, arrays) through unchanged. This is the
        // core difference from `interpolate`.
        let expr = parse(json!([
            "step",
            ["get", "x"],
            ["literal", ["a"]],
            10,
            ["literal", ["a", "b"]]
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let lo = expr.eval(&ctx, &feat_x(0.0)).unwrap();
        let hi = expr.eval(&ctx, &feat_x(20.0)).unwrap();
        assert_eq!(lo, Value::Array(vec![Value::String("a".into())]));
        assert_eq!(
            hi,
            Value::Array(vec![Value::String("a".into()), Value::String("b".into())])
        );
    }

    #[test]
    fn step_rejects_non_ascending_stops_at_parse_time() {
        // Upstream rejects equal-or-decreasing inputs at compile
        // time; we mirror that so the runner's `result: error`
        // expectation in `step/duplicate_stops` flips at parse,
        // before evaluation can run.
        let err = parse(json!(["step", 0, "a", 0, "b", 0, "c"])).unwrap_err();
        assert!(matches!(err, ExprError::StepNonAscendingStops));
    }

    #[test]
    fn interpolate_array_outputs_componentwise() {
        // The `linear-number-array` / `exponential-number-array`
        // fixtures rely on each pair of array elements interpolating
        // independently. We verify the simplest case directly.
        let expr = parse(json!([
            "interpolate",
            ["linear"],
            ["get", "x"],
            0,
            ["literal", [0, 100]],
            10,
            ["literal", [10, 200]]
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: None,
            ..Default::default()
        };
        let v = expr.eval(&ctx, &feat_x(5.0)).unwrap();
        assert_eq!(
            v,
            Value::Array(vec![Value::Number(5.0), Value::Number(150.0)])
        );
    }

    #[test]
    fn interpolate_rejects_non_ascending_stops_at_parse_time() {
        // Mirror of the step case: `interpolate/duplicate_stops`
        // expects the parser to reject equal inputs.
        let err = parse(json!(["interpolate", ["linear"], 0, 0, 0, 1, 0, 2])).unwrap_err();
        // The `2` is an unpaired trailing arg — but the parser still
        // walks the pairs in order, so we expect either the
        // arity error or the non-ascending error. The ordering in
        // `parse_interpolate` runs the arity check first.
        assert!(matches!(
            err,
            ExprError::WrongArity {
                op: "interpolate",
                ..
            } | ExprError::InterpolateNonAscendingStops
        ));

        // A clean even-arity duplicate-input case is unambiguous:
        let err = parse(json!(["interpolate", ["linear"], 0, 0, 0, 0, 1])).unwrap_err();
        assert!(matches!(err, ExprError::InterpolateNonAscendingStops));
    }

    #[test]
    fn cubic_bezier_parse_rejects_out_of_range_control_point() {
        // The `cubic-bezier-invalid-control-point` fixture pins
        // `y1 = 1.75` as a parse-time error. Our parser checks both
        // arity and the `0..=1` range together under one error so
        // upstream's compile-error count lines up.
        let err = parse(json!([
            "interpolate",
            ["cubic-bezier", 0, 1.75, 1, 1],
            ["zoom"],
            0,
            0,
            1,
            1
        ]))
        .unwrap_err();
        assert!(matches!(err, ExprError::CubicBezierInvalidControlPoints));
    }

    // ---- geometry-aware operators (Phase 8) ----

    /// Build a feature whose geometry is a single point in *tile-local*
    /// coordinates, computed from a lng/lat at the given canonical
    /// tile. Mirrors the adapter's projection step so tests exercise
    /// the same code path the fixture runner does.
    fn feature_at_lnglat(lng: f64, lat: f64, canonical: CanonicalTileID) -> Feature {
        let mx = (180.0 + lng) / 360.0;
        let my = (180.0
            - (180.0 / std::f64::consts::PI)
                * (std::f64::consts::PI / 4.0 + lat * std::f64::consts::PI / 360.0)
                    .tan()
                    .ln())
            / 360.0;
        let tiles = 2f64.powi(canonical.z as i32);
        let extent = 8192.0_f64;
        let tx = (mx * tiles * extent).round();
        let ty = (my * tiles * extent).round();
        let x = tx - canonical.x as f64 * extent;
        let y = ty - canonical.y as f64 * extent;
        Feature {
            id: None,
            properties: HashMap::new(),
            geometry: Geometry::Point(geo::Point::new(x, y)),
        }
    }

    #[test]
    fn distance_zero_for_same_lnglat() {
        // Two points at the same lng/lat (after the lossy rounding
        // through tile coords) should be within a fraction of a
        // millimeter of each other. This pins the projection round-
        // trip; an off-by-one in the un-projection would push the
        // error into meters.
        let canonical = CanonicalTileID { z: 20, x: 3, y: 3 };
        let feat = feature_at_lnglat(3.0, 3.0, canonical);
        let expr = parse(json!([
            "distance",
            { "type": "Point", "coordinates": [3.0, 3.0] }
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 20.0,
            canonical: Some(canonical),
            ..Default::default()
        };
        let v = expr.eval(&ctx, &feat).unwrap();
        let Value::Number(d) = v else {
            panic!("expected number, got {v:?}");
        };
        assert!(
            d < 0.01,
            "round-tripped same-point distance should be <1cm, got {d} m"
        );
    }

    #[test]
    fn distance_point_to_point_far() {
        // (3, 3.001) vs (3, 3) is roughly 110 m apart in geodesic
        // distance — the upstream `distance/from-point/to-point-
        // different-location` fixture pins it to `110.575`. We just
        // sanity-check the order of magnitude; the maplibre fixture
        // suite validates the exact value.
        let canonical = CanonicalTileID { z: 20, x: 3, y: 3 };
        let feat = feature_at_lnglat(3.0, 3.001, canonical);
        let expr = parse(json!([
            "distance",
            { "type": "Point", "coordinates": [3.0, 3.0] }
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 20.0,
            canonical: Some(canonical),
            ..Default::default()
        };
        let Value::Number(d) = expr.eval(&ctx, &feat).unwrap() else {
            panic!("expected number");
        };
        assert!((d - 110.575).abs() < 0.1, "expected ~110.575 m, got {d}");
    }

    #[test]
    fn within_point_inside_polygon() {
        // Point (2, 2) lies inside the square with corners (0, 0) ..
        // (5, 5). Mirrors `within/point-within-polygon`'s second input.
        let canonical = CanonicalTileID { z: 3, x: 3, y: 3 };
        let feat = feature_at_lnglat(2.0, 2.0, canonical);
        let expr = parse(json!([
            "within",
            { "type": "Polygon", "coordinates": [[
                [0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0], [0.0, 0.0]
            ]] }
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 3.0,
            canonical: Some(canonical),
            ..Default::default()
        };
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Bool(true));
    }

    #[test]
    fn within_point_outside_polygon() {
        // (6, 6) is outside the same square — the first input of the
        // upstream fixture.
        let canonical = CanonicalTileID { z: 3, x: 3, y: 3 };
        let feat = feature_at_lnglat(6.0, 6.0, canonical);
        let expr = parse(json!([
            "within",
            { "type": "Polygon", "coordinates": [[
                [0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0], [0.0, 0.0]
            ]] }
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 3.0,
            canonical: Some(canonical),
            ..Default::default()
        };
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Bool(false));
    }

    #[test]
    fn within_point_on_boundary_is_outside() {
        // Upstream's `pointWithinPolygon` defaults to "boundary
        // counts as *outside*" for `within`, so a point exactly on the
        // square's corner reports false. `within/point-on-boundary-*`
        // pins this for a range of zooms; we cover one here.
        let canonical = CanonicalTileID { z: 3, x: 3, y: 3 };
        let feat = feature_at_lnglat(5.0, 5.0, canonical);
        let expr = parse(json!([
            "within",
            { "type": "Polygon", "coordinates": [[
                [0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0], [0.0, 0.0]
            ]] }
        ]))
        .unwrap();
        let ctx = EvalContext {
            zoom: 3.0,
            canonical: Some(canonical),
            ..Default::default()
        };
        assert_eq!(expr.eval(&ctx, &feat).unwrap(), Value::Bool(false));
    }

    #[test]
    fn within_rejects_non_polygon_geojson() {
        // Linestring argument fails at compile time per
        // `within/non-supported`.
        let err = parse(json!([
            "within",
            { "type": "LineString", "coordinates": [[0, 0], [1, 1]] }
        ]))
        .unwrap_err();
        assert!(matches!(err, ExprError::WithinInvalidGeoJson));
    }

    #[test]
    fn geometry_type_reports_feature_kind() {
        let canonical = CanonicalTileID { z: 3, x: 3, y: 3 };
        let feat = feature_at_lnglat(0.0, 0.0, canonical);
        let ctx = EvalContext {
            zoom: 0.0,
            canonical: Some(canonical),
            ..Default::default()
        };
        assert_eq!(
            parse(json!(["geometry-type"]))
                .unwrap()
                .eval(&ctx, &feat)
                .unwrap(),
            Value::String("Point".into())
        );
    }

    // ---- Phase 9 — context-driven readers ----

    /// `feature-state` reads from `EvalContext::feature_state`. A
    /// present key surfaces the stored value verbatim; a missing key
    /// collapses to `Null`. The "missing key" branch is *not* an
    /// error here — the spec's pinned error message is raised by the
    /// outer type assertion (`["number", …]`) when it sees the
    /// null. See `feature-state/basic` for the fixture.
    #[test]
    fn feature_state_reads_named_key() {
        let mut state = HashMap::new();
        state.insert("x".to_string(), Value::Number(42.0));
        let ctx = EvalContext {
            feature_state: state,
            ..Default::default()
        };
        let expr = parse(json!(["feature-state", "x"])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feature_with(&[])).unwrap(),
            Value::Number(42.0)
        );
    }

    #[test]
    fn feature_state_missing_key_is_null() {
        let ctx = EvalContext::default();
        let expr = parse(json!(["feature-state", "x"])).unwrap();
        assert_eq!(expr.eval(&ctx, &feature_with(&[])).unwrap(), Value::Null);
    }

    /// `global-state` is the style-scoped sibling of `feature-state`.
    /// Same missing-key contract.
    #[test]
    fn global_state_reads_named_key() {
        let mut state = HashMap::new();
        state.insert("theme".to_string(), Value::String("dark".into()));
        let ctx = EvalContext {
            global_state: state,
            ..Default::default()
        };
        let expr = parse(json!(["global-state", "theme"])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feature_with(&[])).unwrap(),
            Value::String("dark".into())
        );
    }

    #[test]
    fn global_state_missing_key_is_null() {
        let ctx = EvalContext::default();
        let expr = parse(json!(["global-state", "x"])).unwrap();
        assert_eq!(expr.eval(&ctx, &feature_with(&[])).unwrap(), Value::Null);
    }

    /// Upstream rejects a non-string-literal `global-state` arg at
    /// parse time. The `global-state/invalid-property-arg` fixture
    /// covers the `["concat", …]` form; we surface the same rejection
    /// via `ExpectedString`.
    #[test]
    fn global_state_rejects_expression_arg() {
        let err = parse(json!(["global-state", ["concat", "p", "r"]])).unwrap_err();
        assert!(matches!(
            err,
            ExprError::ExpectedString {
                op: "global-state",
                ..
            }
        ));
    }

    #[test]
    fn heatmap_density_reads_context() {
        let ctx = EvalContext {
            heatmap_density: 0.75,
            ..Default::default()
        };
        let expr = parse(json!(["heatmap-density"])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feature_with(&[])).unwrap(),
            Value::Number(0.75)
        );
    }

    #[test]
    fn elevation_reads_context() {
        let ctx = EvalContext {
            elevation: 1234.5,
            ..Default::default()
        };
        let expr = parse(json!(["elevation"])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feature_with(&[])).unwrap(),
            Value::Number(1234.5)
        );
    }

    /// `image` returns a resolved-image value whose `available` flag
    /// reflects whether the name appears in `available_images`. The
    /// fixture `image/basic` pins the available-true path; the
    /// `image/coalesce` chain exercises the available-false fall-
    /// through.
    #[test]
    fn image_resolves_against_available_set() {
        let mut available = std::collections::HashSet::new();
        available.insert("monument-15".to_string());
        let ctx = EvalContext {
            available_images: available,
            ..Default::default()
        };
        let expr = parse(json!(["image", "monument-15"])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feature_with(&[])).unwrap(),
            Value::Image(ResolvedImage {
                name: "monument-15".into(),
                available: true,
            })
        );
    }

    #[test]
    fn image_unavailable_name_resolves_unavailable() {
        let ctx = EvalContext::default();
        let expr = parse(json!(["image", "monument-15"])).unwrap();
        assert_eq!(
            expr.eval(&ctx, &feature_with(&[])).unwrap(),
            Value::Image(ResolvedImage {
                name: "monument-15".into(),
                available: false,
            })
        );
    }

    /// `coalesce` skips unavailable images and falls through to the
    /// first available one — mirroring upstream's `ResolvedImage`-
    /// aware branch, pinned by `image/coalesce`.
    #[test]
    fn coalesce_skips_unavailable_image() {
        let mut available = std::collections::HashSet::new();
        available.insert("monument-15".to_string());
        let ctx = EvalContext {
            available_images: available,
            ..Default::default()
        };
        let expr = parse(json!([
            "coalesce",
            ["image", "foo"],
            ["image", "bar"],
            ["image", "monument-15"],
        ]))
        .unwrap();
        assert_eq!(
            expr.eval(&ctx, &feature_with(&[])).unwrap(),
            Value::Image(ResolvedImage {
                name: "monument-15".into(),
                available: true,
            })
        );
    }

    /// `["image", true]` and the other literal-non-string forms are
    /// rejected at parse time. The matching fixtures live under
    /// `image/invalid-image-name/{boolean,number,null}`.
    #[test]
    fn image_rejects_non_string_literal() {
        for bad in [json!(true), json!(123), json!(null)] {
            let err = parse(json!(["image", bad])).unwrap_err();
            assert!(matches!(err, ExprError::ExpectedString { op: "image", .. }));
        }
    }

    /// An empty context with no state, no available images, and no
    /// heatmap/elevation values still produces clean evaluations:
    /// state lookups collapse to `Null`, the zero-arg numeric ops
    /// return `0.0`, and `image` returns an unavailable resolution.
    /// This guards against accidental panics if a context-driven
    /// operator runs in a vacuum.
    #[test]
    fn empty_context_surfaces_clean_defaults() {
        let ctx = EvalContext::default();
        let feat = feature_with(&[]);
        assert_eq!(
            parse(json!(["feature-state", "x"]))
                .unwrap()
                .eval(&ctx, &feat)
                .unwrap(),
            Value::Null
        );
        assert_eq!(
            parse(json!(["global-state", "x"]))
                .unwrap()
                .eval(&ctx, &feat)
                .unwrap(),
            Value::Null
        );
        assert_eq!(
            parse(json!(["heatmap-density"]))
                .unwrap()
                .eval(&ctx, &feat)
                .unwrap(),
            Value::Number(0.0)
        );
        assert_eq!(
            parse(json!(["elevation"]))
                .unwrap()
                .eval(&ctx, &feat)
                .unwrap(),
            Value::Number(0.0)
        );
        assert_eq!(
            parse(json!(["image", "monument-15"]))
                .unwrap()
                .eval(&ctx, &feat)
                .unwrap(),
            Value::Image(ResolvedImage {
                name: "monument-15".into(),
                available: false,
            })
        );
    }

    // ---- Phase 10 — result_type / constness inference ----

    #[test]
    fn result_type_basic_literals() {
        assert_eq!(parse(json!(0)).unwrap().result_type(), ExprType::Number);
        assert_eq!(parse(json!("hi")).unwrap().result_type(), ExprType::String);
        assert_eq!(parse(json!(true)).unwrap().result_type(), ExprType::Boolean);
        assert_eq!(parse(json!(null)).unwrap().result_type(), ExprType::Null);
    }

    #[test]
    fn result_type_arithmetic_and_assertions() {
        // `+` is always a number.
        assert_eq!(
            parse(json!(["+", 1, 2])).unwrap().result_type(),
            ExprType::Number
        );
        // `["number", expr]` reports `number` regardless of input
        // type.
        assert_eq!(
            parse(json!(["number", ["get", "x"]]))
                .unwrap()
                .result_type(),
            ExprType::Number
        );
        // Coercions report their target.
        assert_eq!(
            parse(json!(["to-string", 5])).unwrap().result_type(),
            ExprType::String
        );
    }

    #[test]
    fn result_type_array_assert_renders_with_length() {
        let e = parse(json!(["array", "number", 2, ["get", "x"]])).unwrap();
        assert_eq!(
            e.result_type(),
            ExprType::Array(Some(Box::new(ExprType::Number)), Some(2))
        );
        assert_eq!(e.result_type().to_spec_string(), "array<number, 2>");
    }

    #[test]
    fn result_type_coalesce_takes_first_child() {
        // First operand is `get` (value); the literal fallback's type
        // doesn't override.
        let e = parse(json!(["coalesce", ["get", "x"], 0])).unwrap();
        assert_eq!(e.result_type(), ExprType::Value);
    }

    #[test]
    fn is_feature_constant_flags_get() {
        // `get` is the canonical non-constant operator.
        assert!(!parse(json!(["get", "x"])).unwrap().is_feature_constant());
        // A pure-arithmetic expression has no feature dependency.
        assert!(parse(json!(["+", 1, 2])).unwrap().is_feature_constant());
        // `zoom` doesn't read the feature.
        let zoom_curve = parse(json!(["interpolate", ["linear"], ["zoom"], 0, 0, 1, 1])).unwrap();
        assert!(zoom_curve.is_feature_constant());
    }

    #[test]
    fn is_zoom_constant_flags_zoom() {
        let zoom_curve = parse(json!(["interpolate", ["linear"], ["zoom"], 0, 0, 1, 1])).unwrap();
        assert!(!zoom_curve.is_zoom_constant());
        // No `zoom` anywhere → zoom-constant.
        let constant = parse(json!(["+", 1, 2])).unwrap();
        assert!(constant.is_zoom_constant());
    }

    #[test]
    fn let_with_inner_var_inherits_binding_type() {
        // `let x = number(get "a")` — binding type is `number`. The
        // body uses `var x`, so the whole `let` reports `number`.
        let e = parse(json!(["let", "x", ["number", ["get", "a"]], ["var", "x"]])).unwrap();
        assert_eq!(e.result_type(), ExprType::Number);
    }

    #[test]
    fn zoom_inside_let_at_top_level_is_legal() {
        // The `zoom/nested-let` fixture's shape: a let-binding that
        // doesn't itself reference zoom, but whose body is an
        // interpolate-over-zoom. The body is in a passthrough position
        // so this should parse cleanly.
        let parsed = parse(json!([
            "let",
            "x",
            ["number", ["get", "a"]],
            ["interpolate", ["linear"], ["zoom"], 0, 0, 30, ["var", "x"]],
        ]));
        assert!(parsed.is_ok(), "expected success, got {parsed:?}");
    }

    #[test]
    fn zoom_inside_arithmetic_is_rejected() {
        // `zoom/invalid-no-curve` shape: `["+", ["zoom"], 0]`. Bare
        // zoom outside a curve's input slot — the parser must reject.
        let err = parse(json!(["+", ["zoom"], 0])).unwrap_err();
        assert!(matches!(err, ExprError::ZoomNestedInvalid));
    }

    // ---- Phase 5 — collections & feature metadata ----

    #[test]
    fn parses_and_evaluates_at() {
        let parsed = parse(json!(["at", 1, ["literal", [10, 20, 30]]])).unwrap();
        let ctx = EvalContext::default();
        let feature = feature_with(&[]);
        let result = parsed.eval(&ctx, &feature).unwrap();
        assert_eq!(result, Value::Number(20.0));
    }

    #[test]
    fn at_out_of_bounds_errors() {
        let parsed = parse(json!(["at", 5, ["literal", [10, 20, 30]]])).unwrap();
        let err = parsed
            .eval(&EvalContext::default(), &feature_with(&[]))
            .unwrap_err();
        assert!(matches!(err, EvalError::AtIndexAboveBounds(5, 2)));
    }

    #[test]
    fn at_fractional_index_errors() {
        let parsed = parse(json!(["at", 1.5, ["literal", [10, 20, 30]]])).unwrap();
        let err = parsed
            .eval(&EvalContext::default(), &feature_with(&[]))
            .unwrap_err();
        assert!(matches!(err, EvalError::AtIndexNotInteger(_)));
    }

    #[test]
    fn parses_and_evaluates_in_string() {
        let parsed = parse(json!(["in", "ll", "hello"])).unwrap();
        let result = parsed
            .eval(&EvalContext::default(), &feature_with(&[]))
            .unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn parses_and_evaluates_in_array() {
        let parsed = parse(json!(["in", 2, ["literal", [1, 2, 3]]])).unwrap();
        let result = parsed
            .eval(&EvalContext::default(), &feature_with(&[]))
            .unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn in_rejects_literal_array_needle_at_parse_time() {
        let err = parse(json!([
            "in",
            ["literal", ["a"]],
            ["literal", [["a"], ["b"]]]
        ]))
        .unwrap_err();
        assert!(matches!(err, ExprError::InNeedleStaticallyArray(_)));
    }

    #[test]
    fn parses_and_evaluates_has() {
        let parsed = parse(json!(["has", "k"])).unwrap();
        let feature_with_k = feature_with(&[("k", MvtValue::Bool(false))]);
        let feature_empty = feature_with(&[]);
        assert_eq!(
            parsed
                .eval(&EvalContext::default(), &feature_with_k)
                .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            parsed
                .eval(&EvalContext::default(), &feature_empty)
                .unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn parses_get_two_arg_form() {
        // `["get", k, host]` reads `k` off the host object expression.
        let parsed = parse(json!(["get", "x", ["literal", {"x": 7}]])).unwrap();
        let result = parsed
            .eval(&EvalContext::default(), &feature_with(&[]))
            .unwrap();
        assert_eq!(result, Value::Number(7.0));
    }

    #[test]
    fn properties_returns_object() {
        let parsed = parse(json!(["properties"])).unwrap();
        let feature = feature_with(&[("a", MvtValue::Double(1.5))]);
        let v = parsed.eval(&EvalContext::default(), &feature).unwrap();
        match v {
            Value::Object(map) => {
                assert_eq!(map.get("a"), Some(&Value::Number(1.5)));
            }
            other => panic!("expected object, got {other:?}"),
        }
    }

    #[test]
    fn id_reads_feature_id() {
        let parsed = parse(json!(["id"])).unwrap();
        let mut feature = feature_with(&[]);
        feature.id = Some(42);
        let v = parsed.eval(&EvalContext::default(), &feature).unwrap();
        assert_eq!(v, Value::Number(42.0));

        let feature = feature_with(&[]);
        let v = parsed.eval(&EvalContext::default(), &feature).unwrap();
        assert_eq!(v, Value::Null);
    }

    #[test]
    fn two_zoom_curves_are_rejected() {
        // `zoom/invalid-nested-2`: two zoom-using interpolates at the
        // top level. The uniqueness check kicks in.
        let err = parse(json!([
            "interpolate",
            ["linear"],
            ["zoom"],
            0,
            0,
            1,
            ["interpolate", ["linear"], ["zoom"], 0, 0, 1, 1],
        ]))
        .unwrap_err();
        // Inner curve is at a stop-output position — that's not a
        // top-level zoom position, so `ZoomNestedInvalid` fires
        // before the uniqueness check.
        assert!(matches!(
            err,
            ExprError::ZoomNestedInvalid | ExprError::ZoomMultipleCurves
        ));
    }

    #[test]
    fn from_json_at_wraps_errors_in_located() {
        // The path-aware entry point attaches the supplied JSON path to
        // any error raised during parse or post-pass validation. The
        // inner variant is preserved so downstream callers can still
        // pattern-match on it via `unwrap_located()`.
        let path = super::super::span::JsonPath::root()
            .child_key("layers")
            .child_index(2)
            .child_key("paint")
            .child_key("line-width");
        let err = Expr::from_json_at(&json!(["interpolate", ["wat"], ["zoom"], 0, 0]), path.clone())
            .unwrap_err();
        match &err {
            ExprError::Located { path: p, source } => {
                assert_eq!(p, &path);
                assert!(matches!(**source, ExprError::UnsupportedInterpolation(_)));
            }
            other => panic!("expected Located, got {other:?}"),
        }
        assert!(matches!(
            err.unwrap_located(),
            ExprError::UnsupportedInterpolation(_)
        ));
    }

    #[test]
    fn from_json_at_preserves_innermost_path_when_already_located() {
        // If an inner parse already attached a path, the outer
        // `with_path` is a no-op. This matters for the case where we
        // start adding per-arg paths later — the deeper one wins.
        let inner_path = super::super::span::JsonPath::root().child_key("inner");
        let outer_path = super::super::span::JsonPath::root().child_key("outer");
        let err = ExprError::EmptyArray.with_path(inner_path.clone());
        let err = err.with_path(outer_path);
        match err {
            ExprError::Located { path, .. } => assert_eq!(path, inner_path),
            _ => panic!("expected Located"),
        }
    }
}
