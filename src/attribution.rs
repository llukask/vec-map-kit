// The credit line a tile source asks to be shown. TileJSON's `attribution`
// is an HTML fragment, in practice a few `<a href>` links and entities
// like `&copy;`:
//
//     <a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a>
//     <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>
//
// `Attribution::parse` reduces that to plain text plus the byte ranges
// that link somewhere, which is the shape GPUI's `StyledText` and
// `InteractiveText` take. It is a tolerant scan for that small subset,
// not an HTML parser: other tags are dropped and their text kept.

use std::ops::Range;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Attribution {
    pub text: String,
    /// Byte ranges of `text` and the http(s) URL each links to.
    pub links: Vec<(Range<usize>, String)>,
}

impl Attribution {
    pub fn parse(html: &str) -> Self {
        let mut out = Builder::default();
        let mut rest = html;
        while let Some(c) = rest.chars().next() {
            // A tag runs to the next `>`; a `<` without one is just text.
            if c == '<'
                && let Some(end) = rest.find('>')
            {
                out.tag(&rest[1..end]);
                rest = &rest[end + 1..];
                continue;
            }
            if c == '&'
                && let Some((decoded, len)) = entity(rest)
            {
                out.push(decoded);
                rest = &rest[len..];
                continue;
            }
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
        out.close_link();
        Attribution { text: out.text, links: out.links }
    }
}

/// Accumulates text with HTML's whitespace rules: any run of whitespace
/// becomes one space, and leading and trailing whitespace disappears.
#[derive(Default)]
struct Builder {
    text: String,
    links: Vec<(Range<usize>, String)>,
    space_pending: bool,
    /// The open `<a>`'s URL and where its text starts, once it has some.
    link: Option<(String, Option<usize>)>,
}

impl Builder {
    fn push(&mut self, c: char) {
        if c.is_whitespace() {
            self.space_pending = true;
            return;
        }
        // The space goes before the link starts, so it isn't underlined.
        if std::mem::take(&mut self.space_pending) && !self.text.is_empty() {
            self.text.push(' ');
        }
        if let Some((_, start @ None)) = &mut self.link {
            *start = Some(self.text.len());
        }
        self.text.push(c);
    }

    fn tag(&mut self, tag: &str) {
        let tag = tag.trim();
        let (closing, tag) = match tag.strip_prefix('/') {
            Some(name) => (true, name.trim_start()),
            None => (false, tag),
        };
        let name = tag.split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("").to_ascii_lowercase();
        match (name.as_str(), closing) {
            ("a", false) => {
                self.close_link();
                self.link = href(tag).filter(|url| is_web_url(url)).map(|url| (url, None));
            }
            ("a", true) => self.close_link(),
            ("br" | "p" | "div" | "li", _) => self.space_pending = true,
            // TODO: other block-level tags; attributions don't use them.
            _ => {}
        }
    }

    fn close_link(&mut self) {
        if let Some((url, Some(start))) = self.link.take() {
            self.links.push((start..self.text.len(), url));
        }
    }
}

/// The `href` attribute of an `<a ...>` tag body, entities decoded.
fn href(tag: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let at = lower.find("href")?;
    let value = tag[at + 4..].trim_start().strip_prefix('=')?.trim_start();
    let raw = match value.chars().next()? {
        quote @ ('"' | '\'') => value[1..].split(quote).next()?,
        _ => value.split(char::is_whitespace).next()?,
    };
    Some(Attribution::parse(raw).text)
}

/// Only web links are opened: a `file:` or `javascript:` URL from a
/// downloaded TileJSON document has no business behind a click.
fn is_web_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.starts_with("https://") || lower.starts_with("http://")
}

/// Decode the entity at the start of `s` (which begins with `&`),
/// returning the character and the entity's length in bytes.
fn entity(s: &str) -> Option<(char, usize)> {
    // Entity names are short; give up after a dozen characters.
    let end = s.char_indices().take(12).find(|&(_, c)| c == ';')?.0;
    let name = &s[1..end];
    let c = match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "copy" => '©',
        _ => {
            let code = match name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => name.strip_prefix('#')?.parse().ok()?,
            };
            char::from_u32(code)?
        }
    };
    Some((c, end + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linked(attribution: &Attribution) -> Vec<(&str, &str)> {
        attribution.links.iter().map(|(range, url)| (&attribution.text[range.clone()], url.as_str())).collect()
    }

    #[test]
    fn maptiler_links_become_ranges() {
        let parsed = Attribution::parse(
            r#"<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>"#,
        );
        assert_eq!(parsed.text, "© MapTiler © OpenStreetMap contributors");
        assert_eq!(
            linked(&parsed),
            [
                ("© MapTiler", "https://www.maptiler.com/copyright/"),
                ("© OpenStreetMap contributors", "https://www.openstreetmap.org/copyright"),
            ]
        );
    }

    #[test]
    fn text_between_links_is_kept() {
        let parsed = Attribution::parse(
            r#"<a href="https://github.com/protomaps/basemaps">Protomaps</a> © <a href='https://openstreetmap.org'>OpenStreetMap</a>"#,
        );
        assert_eq!(parsed.text, "Protomaps © OpenStreetMap");
        assert_eq!(
            linked(&parsed),
            [("Protomaps", "https://github.com/protomaps/basemaps"), ("OpenStreetMap", "https://openstreetmap.org")]
        );
    }

    #[test]
    fn entities_are_decoded() {
        let parsed = Attribution::parse("&copy;&#169;&#xA9; a&amp;b &lt;c&gt; &quot;d&quot; &#39;e&#39;&nbsp;f &bogus; g & h");
        assert_eq!(parsed.text, "©©© a&b <c> \"d\" 'e' f &bogus; g & h");
    }

    #[test]
    fn ampersand_before_multibyte_text_is_literal() {
        assert_eq!(Attribution::parse("Wien & Österreich Straßenkarte").text, "Wien & Österreich Straßenkarte");
        assert_eq!(Attribution::parse("&ÖÖÖÖÖÖ;").text, "&ÖÖÖÖÖÖ;");
    }

    #[test]
    fn other_tags_are_dropped_and_whitespace_collapsed() {
        let parsed = Attribution::parse("  <b>Tiles</b><br/>by\n\t<span class=\"x\">Example</span>  ");
        assert_eq!(parsed.text, "Tiles by Example");
        assert!(parsed.links.is_empty());
    }

    #[test]
    fn only_http_links_are_clickable() {
        // A TileJSON document is fetched from the network; a `javascript:`
        // or `file:` link in it must not become something a click opens.
        let parsed = Attribution::parse(
            r#"<a href="file:///etc/passwd">one</a> <a href="javascript:alert(1)">two</a> <a>three</a> <a href="http://example.com">four</a>"#,
        );
        assert_eq!(parsed.text, "one two three four");
        assert_eq!(linked(&parsed), [("four", "http://example.com")]);
    }

    #[test]
    fn plain_text_passes_through() {
        assert_eq!(Attribution::parse("© Example").text, "© Example");
        assert_eq!(Attribution::parse(""), Attribution::default());
    }
}
