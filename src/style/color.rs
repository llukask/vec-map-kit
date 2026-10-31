// Color values appear all over a style document — `background-color`,
// `fill-color`, `line-color`, `text-color`. The MapLibre spec accepts
// CSS-color forms; we parse:
//
//   * Hex: `#rgb`, `#rrggbb`, `#rrggbbaa`.
//   * `rgb(r, g, b)` / `rgba(r, g, b, a)` — `r/g/b` as 0-255 ints or
//     0-100% percentages; `a` as a 0-1 float.
//   * `hsl(h, s%, l%)` / `hsla(h, s%, l%, a)` — `h` in degrees,
//     `s` / `l` as percentages, `a` as a 0-1 float.
//
// Real-world stylesheets (MapTiler's `maptiler_style.json` in
// particular) use HSL liberally for organic colour ramps, so the
// `hsl*` forms are not optional.
//
// Named colors (`red`, `rebeccapurple`, …) are still rejected — the
// CSS named-color table is ~150 entries and we haven't seen them in
// the styles we exercise. Easy to add later if a fixture needs one.

use std::fmt;

use serde::de::{self, Deserialize, Deserializer};

/// An RGBA color stored as 8-bit channels. The default channel order in
/// renderers tends to vary; we keep R, G, B, A as plain `u8` and let the
/// caller adapt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Parse a CSS-style color string. Recognises hex (`#rgb` /
    /// `#rrggbb` / `#rrggbbaa`), `rgb()` / `rgba()`, and `hsl()` /
    /// `hsla()`. Surrounding whitespace is trimmed. Any other form
    /// (named colors, `color()`, missing prefix, …) returns
    /// [`ColorError::UnsupportedFormat`].
    pub fn parse(s: &str) -> Result<Self, ColorError> {
        let trimmed = s.trim();
        if let Some(hex) = trimmed.strip_prefix('#') {
            return parse_hex(s, hex);
        }
        if let Some(args) = strip_func(trimmed, "rgba") {
            return parse_rgb_args(args, true);
        }
        if let Some(args) = strip_func(trimmed, "rgb") {
            return parse_rgb_args(args, false);
        }
        if let Some(args) = strip_func(trimmed, "hsla") {
            return parse_hsl_args(args, true);
        }
        if let Some(args) = strip_func(trimmed, "hsl") {
            return parse_hsl_args(args, false);
        }
        Err(ColorError::UnsupportedFormat(s.to_string()))
    }

    /// Back-compat alias for [`Color::parse`]. Existing callers passed
    /// hex strings exclusively; the new entry handles every form.
    #[deprecated(note = "use Color::parse — accepts hex, rgb/rgba, hsl/hsla")]
    pub fn from_hex(s: &str) -> Result<Self, ColorError> {
        Self::parse(s)
    }
}

// ==============================================================================
// Hex parsing
// ==============================================================================

fn parse_hex(original: &str, hex: &str) -> Result<Color, ColorError> {
    match hex.len() {
        3 => {
            // `#rgb` is shorthand for `#rrggbb` where each nibble is
            // doubled, e.g. `#abc` → `#aabbcc`.
            let r = double_nibble(parse_nibble(hex, 0)?);
            let g = double_nibble(parse_nibble(hex, 1)?);
            let b = double_nibble(parse_nibble(hex, 2)?);
            Ok(Color { r, g, b, a: 255 })
        }
        6 => {
            let r = parse_byte(hex, 0)?;
            let g = parse_byte(hex, 2)?;
            let b = parse_byte(hex, 4)?;
            Ok(Color { r, g, b, a: 255 })
        }
        8 => {
            let r = parse_byte(hex, 0)?;
            let g = parse_byte(hex, 2)?;
            let b = parse_byte(hex, 4)?;
            let a = parse_byte(hex, 6)?;
            Ok(Color { r, g, b, a })
        }
        _ => Err(ColorError::WrongLength(original.to_string())),
    }
}

fn parse_nibble(hex: &str, index: usize) -> Result<u8, ColorError> {
    let ch = hex
        .as_bytes()
        .get(index)
        .copied()
        .ok_or_else(|| ColorError::WrongLength(hex.to_string()))?;
    match ch {
        b'0'..=b'9' => Ok(ch - b'0'),
        b'a'..=b'f' => Ok(ch - b'a' + 10),
        b'A'..=b'F' => Ok(ch - b'A' + 10),
        _ => Err(ColorError::InvalidDigit(ch as char)),
    }
}

fn parse_byte(hex: &str, index: usize) -> Result<u8, ColorError> {
    let hi = parse_nibble(hex, index)?;
    let lo = parse_nibble(hex, index + 1)?;
    Ok((hi << 4) | lo)
}

fn double_nibble(n: u8) -> u8 {
    (n << 4) | n
}

// ==============================================================================
// Functional-notation helpers
// ==============================================================================

/// If `s` starts with `name(` (case-insensitive) and ends with `)`,
/// return the inner argument list with the parens trimmed.
fn strip_func<'a>(s: &'a str, name: &str) -> Option<&'a str> {
    let bytes = s.as_bytes();
    if bytes.len() < name.len() + 2 || !s.ends_with(')') {
        return None;
    }
    if !s.get(..name.len())?.eq_ignore_ascii_case(name) {
        return None;
    }
    let after_name = s[name.len()..].trim_start();
    let inner = after_name.strip_prefix('(')?.strip_suffix(')')?;
    Some(inner)
}

/// Split a comma- or slash-separated CSS arg list. The CSS color spec
/// allows `rgb(R G B / A)` as well as the legacy comma form; we
/// support both for the common case where a style document mixes
/// notations between authoring tools.
fn split_args(args: &str) -> Vec<&str> {
    args.split([',', '/'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_rgb_args(args: &str, expect_alpha: bool) -> Result<Color, ColorError> {
    let parts = split_args(args);
    let want = if expect_alpha { 4 } else { 3 };
    if parts.len() != want {
        return Err(ColorError::UnsupportedFormat(format!(
            "expected {want} components in rgb args: {args:?}"
        )));
    }
    let r = parse_byte_component(parts[0])?;
    let g = parse_byte_component(parts[1])?;
    let b = parse_byte_component(parts[2])?;
    let a = if expect_alpha {
        parse_alpha(parts[3])?
    } else {
        255
    };
    Ok(Color { r, g, b, a })
}

fn parse_hsl_args(args: &str, expect_alpha: bool) -> Result<Color, ColorError> {
    let parts = split_args(args);
    let want = if expect_alpha { 4 } else { 3 };
    if parts.len() != want {
        return Err(ColorError::UnsupportedFormat(format!(
            "expected {want} components in hsl args: {args:?}"
        )));
    }
    let h = parse_hue(parts[0])?;
    let s = parse_percent(parts[1])?;
    let l = parse_percent(parts[2])?;
    let a = if expect_alpha {
        parse_alpha(parts[3])?
    } else {
        255
    };
    let (r, g, b) = hsl_to_rgb(h, s, l);
    Ok(Color { r, g, b, a })
}

/// Parse one rgb component: `0-255` integer or `0-100%` percentage.
fn parse_byte_component(s: &str) -> Result<u8, ColorError> {
    if let Some(pct) = s.strip_suffix('%') {
        let v: f32 = pct
            .trim()
            .parse()
            .map_err(|_| ColorError::UnsupportedFormat(s.to_string()))?;
        Ok((v.clamp(0.0, 100.0) / 100.0 * 255.0).round() as u8)
    } else {
        let v: f32 = s
            .parse()
            .map_err(|_| ColorError::UnsupportedFormat(s.to_string()))?;
        Ok(v.clamp(0.0, 255.0).round() as u8)
    }
}

/// Parse a percentage like `42%` into `[0, 1]`.
fn parse_percent(s: &str) -> Result<f32, ColorError> {
    let pct = s
        .strip_suffix('%')
        .ok_or_else(|| ColorError::UnsupportedFormat(s.to_string()))?;
    let v: f32 = pct
        .trim()
        .parse()
        .map_err(|_| ColorError::UnsupportedFormat(s.to_string()))?;
    Ok((v / 100.0).clamp(0.0, 1.0))
}

/// Parse a hue in degrees. `deg` / `rad` / `turn` suffixes accepted
/// per CSS; bare numbers default to degrees.
fn parse_hue(s: &str) -> Result<f32, ColorError> {
    let (num, mul) = if let Some(rest) = s.strip_suffix("deg") {
        (rest.trim(), 1.0)
    } else if let Some(rest) = s.strip_suffix("rad") {
        (rest.trim(), 180.0 / std::f32::consts::PI)
    } else if let Some(rest) = s.strip_suffix("turn") {
        (rest.trim(), 360.0)
    } else {
        (s, 1.0)
    };
    let v: f32 = num
        .parse()
        .map_err(|_| ColorError::UnsupportedFormat(s.to_string()))?;
    Ok((v * mul).rem_euclid(360.0))
}

/// Parse the alpha component: a 0-1 float, or `0%-100%`.
fn parse_alpha(s: &str) -> Result<u8, ColorError> {
    if let Some(pct) = s.strip_suffix('%') {
        let v: f32 = pct
            .trim()
            .parse()
            .map_err(|_| ColorError::UnsupportedFormat(s.to_string()))?;
        Ok((v.clamp(0.0, 100.0) / 100.0 * 255.0).round() as u8)
    } else {
        let v: f32 = s
            .parse()
            .map_err(|_| ColorError::UnsupportedFormat(s.to_string()))?;
        Ok((v.clamp(0.0, 1.0) * 255.0).round() as u8)
    }
}

/// Convert HSL (hue degrees, sat 0-1, light 0-1) into 8-bit RGB.
/// Standard CSS formula — kept inline rather than pulled from a
/// crate because the math is short and we don't need any of the
/// `palette`-style colour-space gymnastics.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match h_prime as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c * 0.5;
    let to_u8 = |v: f32| ((v + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    (to_u8(r1), to_u8(g1), to_u8(b1))
}

// ==============================================================================
// Error type
// ==============================================================================

#[derive(Debug, thiserror::Error)]
pub enum ColorError {
    /// Missing recognised prefix or otherwise malformed form.
    #[error(
        "unsupported color format: {0:?} (expected `#rgb`, `#rrggbb`, `#rrggbbaa`, `rgb(...)`, `rgba(...)`, `hsl(...)`, or `hsla(...)`)"
    )]
    UnsupportedFormat(String),
    /// Right prefix, wrong number of hex digits.
    #[error("color {0:?} has wrong length (expected 3, 6, or 8 hex digits)")]
    WrongLength(String),
    /// One of the digits wasn't `[0-9a-fA-F]`.
    #[error("invalid hex digit: {0:?}")]
    InvalidDigit(char),
}

// ==============================================================================
// Deserialize: a JSON string parsed as a CSS color
// ==============================================================================

impl<'de> Deserialize<'de> for Color {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CssVisitor;
        impl de::Visitor<'_> for CssVisitor {
            type Value = Color;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(
                    "a CSS color string: `#rgb` / `#rrggbb` / `#rrggbbaa`, `rgb(...)` / `rgba(...)`, or `hsl(...)` / `hsla(...)`",
                )
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Color, E> {
                Color::parse(v).map_err(E::custom)
            }
        }
        d.deserialize_str(CssVisitor)
    }
}

// ==============================================================================
// Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_six_digit_hex() {
        assert_eq!(
            Color::parse("#f0eee6").unwrap(),
            Color {
                r: 0xf0,
                g: 0xee,
                b: 0xe6,
                a: 0xff
            }
        );
        assert_eq!(
            Color::parse("#a0c8f0").unwrap(),
            Color {
                r: 0xa0,
                g: 0xc8,
                b: 0xf0,
                a: 0xff
            }
        );
    }

    #[test]
    fn parses_three_digit_hex_by_doubling_nibbles() {
        assert_eq!(
            Color::parse("#fff").unwrap(),
            Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255
            }
        );
        assert_eq!(
            Color::parse("#333").unwrap(),
            Color {
                r: 0x33,
                g: 0x33,
                b: 0x33,
                a: 0xff
            }
        );
        assert_eq!(
            Color::parse("#abc").unwrap(),
            Color {
                r: 0xaa,
                g: 0xbb,
                b: 0xcc,
                a: 0xff
            }
        );
    }

    #[test]
    fn parses_eight_digit_hex_with_alpha() {
        assert_eq!(
            Color::parse("#f0eee6ff").unwrap(),
            Color {
                r: 0xf0,
                g: 0xee,
                b: 0xe6,
                a: 0xff
            }
        );
        assert_eq!(
            Color::parse("#00000080").unwrap(),
            Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0x80
            }
        );
    }

    #[test]
    fn accepts_uppercase_digits() {
        assert_eq!(
            Color::parse("#FFFFFF").unwrap(),
            Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255
            }
        );
    }

    #[test]
    fn rejects_missing_prefix() {
        assert!(matches!(
            Color::parse("ffffff"),
            Err(ColorError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn rejects_named_color() {
        // Named colors aren't supported in this pass — `red` should be
        // a hard error, not silently mapped to anything.
        assert!(matches!(
            Color::parse("red"),
            Err(ColorError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn parses_rgb_integer_components() {
        assert_eq!(
            Color::parse("rgb(255, 0, 128)").unwrap(),
            Color {
                r: 255,
                g: 0,
                b: 128,
                a: 0xff
            }
        );
    }

    #[test]
    fn parses_rgb_percentage_components() {
        assert_eq!(
            Color::parse("rgb(100%, 0%, 50%)").unwrap(),
            Color {
                r: 255,
                g: 0,
                b: 128,
                a: 0xff
            }
        );
    }

    #[test]
    fn parses_rgba_with_float_alpha() {
        let c = Color::parse("rgba(0, 0, 0, 0.5)").unwrap();
        assert_eq!(c.r, 0);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        // 0.5 * 255 = 127.5 → rounds to 128.
        assert_eq!(c.a, 128);
    }

    #[test]
    fn parses_hsl_pure_red() {
        // hsl(0, 100%, 50%) = #ff0000.
        assert_eq!(
            Color::parse("hsl(0, 100%, 50%)").unwrap(),
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 0xff
            }
        );
    }

    #[test]
    fn parses_hsl_pure_white_and_black() {
        // lightness 100% / 0% short-circuits to white / black regardless
        // of saturation or hue.
        assert_eq!(
            Color::parse("hsl(0, 0%, 100%)").unwrap(),
            Color {
                r: 255,
                g: 255,
                b: 255,
                a: 0xff
            }
        );
        assert_eq!(
            Color::parse("hsl(0, 0%, 0%)").unwrap(),
            Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0xff
            }
        );
    }

    #[test]
    fn parses_hsl_from_maptiler_style() {
        // Background-color from `maptiler_style.json` — the value
        // that motivated HSL support. CSS-spec algorithm:
        //   C = (1-|2·0.93-1|)·0.19 ≈ 0.0266
        //   H'=1 → (R',G',B') = (X, C, 0)
        //   m = 0.93 - C/2 ≈ 0.9167
        // → (R, G, B) ≈ (0xf1, 0xf1, 0xea).
        let c = Color::parse("hsl(60, 19%, 93%)").unwrap();
        let ok = |got: u8, want: u8| got.abs_diff(want) <= 1;
        assert!(ok(c.r, 0xf1), "R was {:#x}", c.r);
        assert!(ok(c.g, 0xf1), "G was {:#x}", c.g);
        assert!(ok(c.b, 0xea), "B was {:#x}", c.b);
        assert_eq!(c.a, 0xff);
    }

    #[test]
    fn parses_hsla_with_alpha() {
        let c = Color::parse("hsla(120, 100%, 50%, 0.25)").unwrap();
        assert_eq!(c.r, 0);
        assert_eq!(c.g, 255);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 64);
    }

    #[test]
    fn parses_hue_with_deg_suffix() {
        assert_eq!(
            Color::parse("hsl(180deg, 100%, 50%)").unwrap(),
            Color::parse("hsl(180, 100%, 50%)").unwrap(),
        );
    }

    #[test]
    fn case_insensitive_function_name() {
        assert_eq!(
            Color::parse("HSL(0, 100%, 50%)").unwrap(),
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 0xff
            }
        );
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(matches!(
            Color::parse("#ffff"),
            Err(ColorError::WrongLength(_))
        ));
        assert!(matches!(
            Color::parse("#1234567"),
            Err(ColorError::WrongLength(_))
        ));
    }

    #[test]
    fn rejects_non_hex_digit() {
        assert!(matches!(
            Color::parse("#xyz"),
            Err(ColorError::InvalidDigit(_))
        ));
    }

    #[test]
    fn deserializes_from_json_string() {
        let c: Color = serde_json::from_str("\"#a0c8f0\"").unwrap();
        assert_eq!(
            c,
            Color {
                r: 0xa0,
                g: 0xc8,
                b: 0xf0,
                a: 0xff
            }
        );
        let c2: Color = serde_json::from_str("\"hsl(0, 100%, 50%)\"").unwrap();
        assert_eq!(
            c2,
            Color {
                r: 255,
                g: 0,
                b: 0,
                a: 0xff
            }
        );
    }

    #[test]
    fn deserialize_error_propagates() {
        let err = serde_json::from_str::<Color>("\"red\"").unwrap_err();
        assert!(err.to_string().contains("unsupported color format"));
    }
}
