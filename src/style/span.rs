// JSON source-span tracking. Lets style/TileJSON errors point back into
// the original document for snippet-style diagnostics.
//
// We're handed an already-parsed `serde_json::Value` (so the source is
// known-valid JSON) plus the source text. The job is just to walk the
// two in lockstep and remember, for every interesting `JsonPath`, the
// byte range in the source where that value lives.
//
// The walker isn't a full RFC-8259 parser — it relies on serde_json
// having already accepted the document. It still needs to recognise
// every token shape correctly (strings with escapes, numbers, the
// literals) so the path/value alignment stays in sync.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::Value as Json;

// ==============================================================================
// Public types
// ==============================================================================

/// A half-open byte range into the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ByteSpan {
    pub start: usize,
    pub end: usize,
}

impl ByteSpan {
    pub fn new(start: usize, end: usize) -> Self {
        ByteSpan { start, end }
    }

    /// Empty span at position 0 — used as a "no location available"
    /// sentinel when error sites fail to look up a real span.
    pub fn empty() -> Self {
        ByteSpan { start: 0, end: 0 }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

impl From<ByteSpan> for std::ops::Range<usize> {
    fn from(s: ByteSpan) -> std::ops::Range<usize> {
        s.start..s.end
    }
}

/// One step of a JSON pointer. Object keys keep their original spelling
/// (no escaping); array indices are plain `usize`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PathSegment {
    Key(String),
    Index(usize),
}

/// JSON path through a document. Display formats it as
/// `.layers[2].paint.line-width` — convenient for log lines and error
/// messages.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JsonPath {
    segments: Vec<PathSegment>,
}

impl JsonPath {
    pub fn root() -> Self {
        JsonPath::default()
    }

    pub fn segments(&self) -> &[PathSegment] {
        &self.segments
    }

    pub fn push_key(&mut self, key: impl Into<String>) {
        self.segments.push(PathSegment::Key(key.into()));
    }

    pub fn push_index(&mut self, idx: usize) {
        self.segments.push(PathSegment::Index(idx));
    }

    pub fn pop(&mut self) {
        self.segments.pop();
    }

    pub fn child_key(&self, key: impl Into<String>) -> Self {
        let mut out = self.clone();
        out.push_key(key);
        out
    }

    pub fn child_index(&self, idx: usize) -> Self {
        let mut out = self.clone();
        out.push_index(idx);
        out
    }
}

impl fmt::Display for JsonPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.segments.is_empty() {
            return f.write_str(".");
        }
        for seg in &self.segments {
            match seg {
                PathSegment::Key(k) => write!(f, ".{k}")?,
                PathSegment::Index(i) => write!(f, "[{i}]")?,
            }
        }
        Ok(())
    }
}

/// `JsonPath → ByteSpan` map built once per source document. Lookups
/// are by exact path equality.
#[derive(Debug, Default, Clone)]
pub struct SpanMap {
    spans: BTreeMap<JsonPath, ByteSpan>,
    // Length of the source the map was built from. Used as a fallback
    // when callers ask for the document root span.
    source_len: usize,
}

impl SpanMap {
    pub fn lookup(&self, path: &JsonPath) -> Option<ByteSpan> {
        self.spans.get(path).copied()
    }

    /// Look up `path`, falling back to the nearest enclosing path that
    /// is present in the map. Useful when an expression-parse error
    /// targets a nested fragment we didn't index (e.g. an interpolation
    /// stop output that is itself a literal) — the closest ancestor
    /// still points the user at the right region.
    pub fn lookup_or_ancestor(&self, path: &JsonPath) -> Option<ByteSpan> {
        let mut cur = path.clone();
        loop {
            if let Some(span) = self.lookup(&cur) {
                return Some(span);
            }
            if cur.segments.is_empty() {
                return None;
            }
            cur.pop();
        }
    }

    pub fn source_len(&self) -> usize {
        self.source_len
    }
}

// ==============================================================================
// Map builder
// ==============================================================================

/// Walk the source text and the already-parsed value tree together,
/// recording a `ByteSpan` for every visited node, keyed by the path
/// from the document root.
///
/// Precondition: `root` is what `serde_json::from_str(src)` returns. If
/// `src` doesn't actually parse to `root` (a serde bug or a hand-built
/// tree), the walker can desync; in that case we return whatever spans
/// we managed to collect so far rather than panicking. Callers fall
/// back to [`ByteSpan::empty`] when a lookup misses.
pub fn build_span_map(src: &str, root: &Json) -> SpanMap {
    let mut walker = Walker {
        bytes: src.as_bytes(),
        pos: 0,
        spans: BTreeMap::new(),
        path: JsonPath::root(),
        desynced: false,
    };
    walker.skip_ws();
    let _ = walker.visit(root);
    SpanMap {
        spans: walker.spans,
        source_len: src.len(),
    }
}

// ------------------------------------------------------------------------------
// Internal walker
// ------------------------------------------------------------------------------
//
// The walker tracks a byte cursor through the source and pushes/pops
// path segments as it descends. At each `visit(node)` call:
//   1. Skip whitespace.
//   2. Record `start = pos`.
//   3. Consume tokens appropriate to the node's `Json` variant.
//   4. Record `(path, ByteSpan { start, end = pos })`.
//
// String scanning honours JSON escape rules (including `\u`) so the
// cursor lands on the byte after the closing quote, matching whatever
// serde_json consumed.

struct Walker<'a> {
    bytes: &'a [u8],
    pos: usize,
    spans: BTreeMap<JsonPath, ByteSpan>,
    path: JsonPath,
    desynced: bool,
}

impl Walker<'_> {
    fn visit(&mut self, node: &Json) -> Result<(), Desync> {
        if self.desynced {
            return Err(Desync);
        }
        self.skip_ws();
        let start = self.pos;
        match node {
            Json::Null => self.consume_literal(b"null")?,
            Json::Bool(true) => self.consume_literal(b"true")?,
            Json::Bool(false) => self.consume_literal(b"false")?,
            Json::Number(_) => self.consume_number()?,
            Json::String(_) => self.consume_string()?,
            Json::Array(items) => self.consume_array(items)?,
            Json::Object(map) => self.consume_object(map)?,
        }
        let end = self.pos;
        self.spans
            .insert(self.path.clone(), ByteSpan { start, end });
        Ok(())
    }

    fn consume_literal(&mut self, expected: &[u8]) -> Result<(), Desync> {
        if self.bytes.len() < self.pos + expected.len()
            || &self.bytes[self.pos..self.pos + expected.len()] != expected
        {
            self.desynced = true;
            return Err(Desync);
        }
        self.pos += expected.len();
        Ok(())
    }

    // JSON numbers: optional `-`, integer part, optional `.frac`,
    // optional `e[+-]?digits`. We don't validate — serde_json already
    // did — we just consume the matching bytes.
    fn consume_number(&mut self) -> Result<(), Desync> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if self.pos == start {
            self.desynced = true;
            return Err(Desync);
        }
        Ok(())
    }

    fn consume_string(&mut self) -> Result<(), Desync> {
        if self.peek() != Some(b'"') {
            self.desynced = true;
            return Err(Desync);
        }
        self.pos += 1;
        while let Some(b) = self.peek() {
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(());
                }
                b'\\' => {
                    // Skip the backslash and the next escape character.
                    // `\u` is followed by four hex digits but they're
                    // not allowed to be `"` so a generic two-byte skip
                    // is enough — we'll keep advancing past `\uXXXX` as
                    // normal characters after the escape byte.
                    self.pos += 1;
                    if self.peek().is_some() {
                        self.pos += 1;
                    }
                }
                _ => self.pos += 1,
            }
        }
        self.desynced = true;
        Err(Desync)
    }

    fn consume_array(&mut self, items: &[Json]) -> Result<(), Desync> {
        if self.peek() != Some(b'[') {
            self.desynced = true;
            return Err(Desync);
        }
        self.pos += 1;
        for (idx, item) in items.iter().enumerate() {
            if idx > 0 {
                self.expect_separator(b',')?;
            }
            self.path.push_index(idx);
            self.visit(item)?;
            self.path.pop();
        }
        self.skip_ws();
        if self.peek() != Some(b']') {
            self.desynced = true;
            return Err(Desync);
        }
        self.pos += 1;
        Ok(())
    }

    fn consume_object(
        &mut self,
        map: &serde_json::Map<String, Json>,
    ) -> Result<(), Desync> {
        if self.peek() != Some(b'{') {
            self.desynced = true;
            return Err(Desync);
        }
        self.pos += 1;
        // serde_json::Map preserves insertion order under the
        // `preserve_order` feature, which we don't enable. Without it,
        // the map is alphabetical-by-key. We can't trust map iteration
        // for source order, so we walk the source ourselves: at each
        // position skip ws, read the key, skip `:`, walk the value.
        let mut first = true;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(());
                }
                Some(_) if !first => {
                    self.expect_separator(b',')?;
                    self.skip_ws();
                }
                Some(_) => {}
                None => {
                    self.desynced = true;
                    return Err(Desync);
                }
            }
            first = false;
            // Capture the key as a string, then descend.
            let key_start = self.pos;
            self.consume_string()?;
            let key_bytes = &self.bytes[key_start + 1..self.pos - 1];
            let key = decode_key(key_bytes);
            let Some(child) = map.get(&key) else {
                // Unknown-to-us key. The serde Value tree drops a key
                // (shouldn't happen for `serde_json::Value` since it
                // keeps all keys), so we treat this as a desync.
                self.desynced = true;
                return Err(Desync);
            };
            self.skip_ws();
            if self.peek() != Some(b':') {
                self.desynced = true;
                return Err(Desync);
            }
            self.pos += 1;
            self.path.push_key(key);
            self.visit(child)?;
            self.path.pop();
        }
    }

    fn expect_separator(&mut self, sep: u8) -> Result<(), Desync> {
        self.skip_ws();
        if self.peek() != Some(sep) {
            self.desynced = true;
            return Err(Desync);
        }
        self.pos += 1;
        self.skip_ws();
        Ok(())
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
}

#[derive(Debug)]
struct Desync;

// Decode JSON escape sequences in a key. We only need this to recover
// the canonical map key as serde_json sees it — control characters and
// `\uXXXX` escapes are uncommon in style documents but we handle them
// to stay correct. Anything we can't decode falls back to the raw bytes
// as UTF-8 lossy, which keeps the walker advancing rather than
// panicking.
fn decode_key(bytes: &[u8]) -> String {
    let s = std::str::from_utf8(bytes).unwrap_or_default();
    if !s.contains('\\') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some('b') => out.push('\u{0008}'),
            Some('f') => out.push('\u{000C}'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('u') => {
                let mut hex = String::new();
                for _ in 0..4 {
                    if let Some(h) = chars.next() {
                        hex.push(h);
                    }
                }
                if let Ok(code) = u32::from_str_radix(&hex, 16)
                    && let Some(ch) = char::from_u32(code)
                {
                    out.push(ch);
                } else {
                    out.push('\u{FFFD}');
                }
            }
            Some(other) => out.push(other),
            None => break,
        }
    }
    out
}

// ==============================================================================
// Source-text helpers
// ==============================================================================

/// Convert a 1-based (line, column) coordinate from `serde_json::Error`
/// into a byte offset into `src`. Out-of-range coordinates clamp to the
/// source length so the caller still gets a valid offset for snippet
/// rendering.
///
/// `serde_json` documents column as 1-based, counted in UTF-8 *bytes*
/// since the last newline. We mirror that exactly: walk to the target
/// line, then add `column - 1`.
pub fn byte_offset_from_line_col(src: &str, line: usize, column: usize) -> usize {
    if line == 0 {
        return 0;
    }
    let mut offset = 0;
    let mut current_line = 1;
    let bytes = src.as_bytes();
    while current_line < line && offset < bytes.len() {
        if bytes[offset] == b'\n' {
            current_line += 1;
        }
        offset += 1;
    }
    let col_offset = column.saturating_sub(1);
    (offset + col_offset).min(bytes.len())
}

// ==============================================================================
// Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_STYLE: &str = include_str!("../../tests/fixtures/minimal_style.json");

    fn make_map(src: &str) -> SpanMap {
        let v: Json = serde_json::from_str(src).expect("valid json");
        build_span_map(src, &v)
    }

    fn span_text(src: &str, span: ByteSpan) -> &str {
        &src[span.start..span.end]
    }

    #[test]
    fn root_path_covers_whole_document() {
        let src = r#"{ "a": 1 }"#;
        let map = make_map(src);
        let root = map.lookup(&JsonPath::root()).expect("root span");
        assert_eq!(span_text(src, root), r#"{ "a": 1 }"#);
    }

    #[test]
    fn object_keys_each_get_their_value_span() {
        let src = r#"{ "a": 1, "b": "two", "c": true }"#;
        let map = make_map(src);
        let a = map.lookup(&JsonPath::root().child_key("a")).expect("a");
        let b = map.lookup(&JsonPath::root().child_key("b")).expect("b");
        let c = map.lookup(&JsonPath::root().child_key("c")).expect("c");
        assert_eq!(span_text(src, a), "1");
        assert_eq!(span_text(src, b), r#""two""#);
        assert_eq!(span_text(src, c), "true");
    }

    #[test]
    fn nested_arrays_index_correctly() {
        let src = r#"[10, [20, [30, 40]]]"#;
        let map = make_map(src);
        let nested = map
            .lookup(
                &JsonPath::root()
                    .child_index(1)
                    .child_index(1)
                    .child_index(0),
            )
            .expect("nested[1][1][0]");
        assert_eq!(span_text(src, nested), "30");
    }

    #[test]
    fn handles_escaped_strings_without_desync() {
        let src = r#"{ "tricky": "a\"b\\c", "next": 7 }"#;
        let map = make_map(src);
        let tricky = map
            .lookup(&JsonPath::root().child_key("tricky"))
            .expect("tricky");
        assert_eq!(span_text(src, tricky), r#""a\"b\\c""#);
        let next = map
            .lookup(&JsonPath::root().child_key("next"))
            .expect("next");
        assert_eq!(span_text(src, next), "7");
    }

    #[test]
    fn handles_unicode_keys_and_escapes() {
        // A key written with a `\u` escape decodes to the canonical
        // form. serde_json::Map keys by the decoded string, so the
        // walker must match.
        let src = r#"{ "caé": 1 }"#;
        let map = make_map(src);
        let value = map
            .lookup(&JsonPath::root().child_key("ca\u{00e9}"))
            .expect("value");
        assert_eq!(span_text(src, value), "1");
    }

    #[test]
    fn line_width_in_minimal_style_points_at_interpolate_array() {
        // .layers[2].paint.line-width is the interpolate expression on
        // the highways layer. The span should cover the whole array,
        // starting at `[` and ending after the matching `]`.
        let map = make_map(MINIMAL_STYLE);
        let path = JsonPath::root()
            .child_key("layers")
            .child_index(2)
            .child_key("paint")
            .child_key("line-width");
        let span = map.lookup(&path).expect("line-width span");
        let text = span_text(MINIMAL_STYLE, span);
        assert!(text.starts_with('['), "got: {text:?}");
        assert!(text.trim_end().ends_with(']'), "got: {text:?}");
        assert!(text.contains("\"interpolate\""), "got: {text:?}");
    }

    #[test]
    fn version_field_in_minimal_style_is_the_literal_8() {
        let map = make_map(MINIMAL_STYLE);
        let span = map
            .lookup(&JsonPath::root().child_key("version"))
            .expect("version span");
        assert_eq!(span_text(MINIMAL_STYLE, span), "8");
    }

    #[test]
    fn negative_and_exponent_numbers_are_consumed_correctly() {
        let src = r#"[ -1.5e2, 0, -0.25 ]"#;
        let map = make_map(src);
        let a = map.lookup(&JsonPath::root().child_index(0)).unwrap();
        let b = map.lookup(&JsonPath::root().child_index(1)).unwrap();
        let c = map.lookup(&JsonPath::root().child_index(2)).unwrap();
        assert_eq!(span_text(src, a), "-1.5e2");
        assert_eq!(span_text(src, b), "0");
        assert_eq!(span_text(src, c), "-0.25");
    }

    #[test]
    fn display_format_matches_spec() {
        let path = JsonPath::root()
            .child_key("layers")
            .child_index(2)
            .child_key("paint")
            .child_key("line-width");
        assert_eq!(path.to_string(), ".layers[2].paint.line-width");
        assert_eq!(JsonPath::root().to_string(), ".");
    }

    #[test]
    fn line_col_to_byte_offset_matches_serde_json() {
        let src = "{\n  \"a\": ?\n}";
        // serde_json reports the `?` at line 2, column 8 (1-based).
        let err = serde_json::from_str::<Json>(src).unwrap_err();
        let offset = byte_offset_from_line_col(src, err.line(), err.column());
        assert_eq!(src.as_bytes()[offset], b'?');
    }
}
