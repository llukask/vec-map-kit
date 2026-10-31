// Decode an MVT geometry command stream into a `geo` value.
//
// MVT stores geometry as a flat `Vec<u32>` of interleaved command words
// and zig-zag-encoded coordinate deltas (spec § 4.3). A *command word*
// packs an id (low 3 bits — 1=MoveTo, 2=LineTo, 7=ClosePath) and a count
// (high 29 bits — how many parameter pairs follow). Coordinates are
// expressed as deltas from a running cursor that starts at `(0, 0)`.
//
// This module is the only place that has to understand any of that. The
// rest of the crate works with `Geometry` values whose variants line up
// with the MVT GeomType enum.

use super::error::{Error, Result};
use super::proto;

use geo::{Coord, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon};

// ==============================================================================
// Public API
// ==============================================================================

/// High-level geometry as decoded from an MVT feature.
///
/// Variants mirror `proto::tile::GeomType`. UNKNOWN tiles surface as
/// [`Geometry::Unknown`] so callers can choose to skip them; the spec
/// (§ 4.3.4.1) explicitly permits decoders to do so.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    Unknown,
    Point(Point<f64>),
    MultiPoint(MultiPoint<f64>),
    LineString(LineString<f64>),
    MultiLineString(MultiLineString<f64>),
    Polygon(Polygon<f64>),
    MultiPolygon(MultiPolygon<f64>),
}

pub fn decode_geometry(geom_type: proto::tile::GeomType, raw: &[u32]) -> Result<Geometry> {
    use proto::tile::GeomType::*;
    match geom_type {
        Unknown => Ok(Geometry::Unknown),
        Point => decode_point(raw),
        Linestring => decode_linestring(raw),
        Polygon => decode_polygon(raw),
    }
}

// ==============================================================================
// Command-stream primitives
// ==============================================================================

// Zig-zag decode a single parameter integer. Defined by the spec
// (§ 4.3.2) so that small positive *and* negative deltas both round-trip
// to small unsigned integers.
#[inline]
fn zigzag(p: u32) -> i32 {
    ((p >> 1) as i32) ^ -((p & 1) as i32)
}

const ID_MOVE_TO: u32 = 1;
const ID_LINE_TO: u32 = 2;
const ID_CLOSE_PATH: u32 = 7;

fn cmd_name(id: u32) -> &'static str {
    match id {
        ID_MOVE_TO => "MoveTo",
        ID_LINE_TO => "LineTo",
        ID_CLOSE_PATH => "ClosePath",
        _ => "Unknown",
    }
}

// Stateful walker over a geometry stream. Owns the running cursor that
// every command's parameters are decoded relative to.
struct Parser<'a> {
    raw: &'a [u32],
    pos: usize,
    cx: i32,
    cy: i32,
}

impl<'a> Parser<'a> {
    fn new(raw: &'a [u32]) -> Self {
        Self {
            raw,
            pos: 0,
            cx: 0,
            cy: 0,
        }
    }

    fn done(&self) -> bool {
        self.pos >= self.raw.len()
    }

    fn next_word(&mut self) -> Option<u32> {
        let w = self.raw.get(self.pos).copied()?;
        self.pos += 1;
        Some(w)
    }

    // Pull the next command word; reject anything that isn't a known id so
    // we never feed bogus counts into the parameter loop.
    fn next_command(&mut self) -> Result<(u32, u32)> {
        let w = self
            .next_word()
            .ok_or(Error::Truncated { consumed: self.pos })?;
        let id = w & 0x7;
        let count = w >> 3;
        if !matches!(id, ID_MOVE_TO | ID_LINE_TO | ID_CLOSE_PATH) {
            return Err(Error::InvalidCommandId { id });
        }
        Ok((id, count))
    }

    fn next_pair(&mut self) -> Result<(i32, i32)> {
        let dx = self
            .next_word()
            .ok_or(Error::Truncated { consumed: self.pos })?;
        let dy = self
            .next_word()
            .ok_or(Error::Truncated { consumed: self.pos })?;
        Ok((zigzag(dx), zigzag(dy)))
    }

    // Apply a delta to the cursor with overflow checking. The spec caps
    // each parameter at ±(2³¹−1), but successive moves can still drift
    // out of `i32` range on a malformed tile.
    fn advance(&mut self, dx: i32, dy: i32, cmd_index: usize) -> Result<(i32, i32)> {
        self.cx = self
            .cx
            .checked_add(dx)
            .ok_or(Error::CoordOverflow { cmd_index })?;
        self.cy = self
            .cy
            .checked_add(dy)
            .ok_or(Error::CoordOverflow { cmd_index })?;
        Ok((self.cx, self.cy))
    }
}

#[inline]
fn coord(x: i32, y: i32) -> Coord<f64> {
    Coord {
        x: x as f64,
        y: y as f64,
    }
}

// ==============================================================================
// Per-GeomType decoders
// ==============================================================================

// POINT (spec § 4.3.4.2): exactly one MoveTo with count ≥ 1. Count == 1
// is a single Point; count > 1 is a MultiPoint whose vertices are the
// successive cursor positions.
fn decode_point(raw: &[u32]) -> Result<Geometry> {
    if raw.is_empty() {
        return Err(Error::EmptyGeometry { geom_type: "POINT" });
    }
    let mut p = Parser::new(raw);
    let cmd_index = p.pos;
    let (id, count) = p.next_command()?;
    if id != ID_MOVE_TO {
        return Err(Error::UnexpectedCommand {
            cmd: cmd_name(id),
            geom_type: "POINT",
        });
    }
    if count == 0 {
        return Err(Error::InvalidCount {
            cmd: "MoveTo",
            count,
        });
    }

    let mut pts = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let (dx, dy) = p.next_pair()?;
        let (x, y) = p.advance(dx, dy, cmd_index)?;
        pts.push(Point::new(x as f64, y as f64));
    }

    // No more words allowed after the single MoveTo block.
    if !p.done() {
        return Err(Error::UnexpectedCommand {
            cmd: "trailing",
            geom_type: "POINT",
        });
    }

    Ok(if pts.len() == 1 {
        Geometry::Point(pts.remove(0))
    } else {
        Geometry::MultiPoint(MultiPoint(pts))
    })
}

// LINESTRING (spec § 4.3.4.3): one or more repetitions of
// `MoveTo(1)`/`LineTo(≥1)`. Single repetition → LineString; multiple →
// MultiLineString.
fn decode_linestring(raw: &[u32]) -> Result<Geometry> {
    let mut p = Parser::new(raw);
    let mut lines: Vec<LineString<f64>> = Vec::new();

    while !p.done() {
        // MoveTo(1) — starts a new linestring at `cursor + (dx, dy)`.
        let cmd_index = p.pos;
        let (id, count) = p.next_command()?;
        if id != ID_MOVE_TO {
            return Err(Error::UnexpectedCommand {
                cmd: cmd_name(id),
                geom_type: "LINESTRING",
            });
        }
        if count != 1 {
            return Err(Error::InvalidCount {
                cmd: "MoveTo",
                count,
            });
        }
        let (dx, dy) = p.next_pair()?;
        let (x, y) = p.advance(dx, dy, cmd_index)?;
        let mut coords = vec![coord(x, y)];

        // LineTo(≥1) — extends the line by `count` more vertices.
        let cmd_index = p.pos;
        let (id, count) = p.next_command()?;
        if id != ID_LINE_TO {
            return Err(Error::UnexpectedCommand {
                cmd: cmd_name(id),
                geom_type: "LINESTRING",
            });
        }
        if count == 0 {
            return Err(Error::InvalidCount {
                cmd: "LineTo",
                count,
            });
        }
        for _ in 0..count {
            let (dx, dy) = p.next_pair()?;
            let (x, y) = p.advance(dx, dy, cmd_index)?;
            coords.push(coord(x, y));
        }
        lines.push(LineString::new(coords));
    }

    if lines.is_empty() {
        return Err(Error::EmptyGeometry {
            geom_type: "LINESTRING",
        });
    }
    Ok(if lines.len() == 1 {
        Geometry::LineString(lines.remove(0))
    } else {
        Geometry::MultiLineString(MultiLineString(lines))
    })
}

// POLYGON (spec § 4.3.4.4): one or more rings, each `MoveTo(1) /
// LineTo(≥2) / ClosePath(1)`. Ring orientation, computed via the signed
// shoelace area in tile coordinates (Y positive down), is the only thing
// that disambiguates exteriors from holes:
//
//   * area > 0 (clockwise on screen)         → exterior ring → new polygon
//   * area < 0 (counter-clockwise on screen) → interior ring of last polygon
//   * area == 0                              → degenerate; reject
//
// A single exterior produces a Polygon; two or more produce a
// MultiPolygon. Interior rings always attach to the most recently
// emitted exterior, matching the spec's requirement (§ 4.3.4.4) that
// holes follow their enclosing polygon.
fn decode_polygon(raw: &[u32]) -> Result<Geometry> {
    let mut p = Parser::new(raw);
    let mut polys: Vec<(LineString<f64>, Vec<LineString<f64>>)> = Vec::new();
    // Sign of the first non-degenerate ring's area; see the ring
    // classification below.
    let mut exterior_winding: Option<bool> = None;

    while !p.done() {
        // MoveTo(1) — start vertex of the ring. We remember the starting
        // tile-space coords so we can explicitly close the ring.
        let cmd_index = p.pos;
        let (id, count) = p.next_command()?;
        if id != ID_MOVE_TO {
            return Err(Error::UnexpectedCommand {
                cmd: cmd_name(id),
                geom_type: "POLYGON",
            });
        }
        if count != 1 {
            return Err(Error::InvalidCount {
                cmd: "MoveTo",
                count,
            });
        }
        let (dx, dy) = p.next_pair()?;
        let (sx, sy) = p.advance(dx, dy, cmd_index)?;
        let mut coords = vec![coord(sx, sy)];

        // LineTo(≥2) — fills in the rest of the ring's vertices.
        let cmd_index = p.pos;
        let (id, count) = p.next_command()?;
        if id != ID_LINE_TO {
            return Err(Error::UnexpectedCommand {
                cmd: cmd_name(id),
                geom_type: "POLYGON",
            });
        }
        if count < 2 {
            return Err(Error::InvalidCount {
                cmd: "LineTo",
                count,
            });
        }
        for _ in 0..count {
            let (dx, dy) = p.next_pair()?;
            let (x, y) = p.advance(dx, dy, cmd_index)?;
            coords.push(coord(x, y));
        }

        // ClosePath(1) — no parameters; spec says it MUST NOT move the
        // cursor, so we don't touch `p.cx/cy`. We *do* push the start
        // vertex onto the ring so `geo::Polygon` sees a properly closed
        // LineString.
        let (id, count) = p.next_command()?;
        if id != ID_CLOSE_PATH {
            return Err(Error::UnexpectedCommand {
                cmd: cmd_name(id),
                geom_type: "POLYGON",
            });
        }
        if count != 1 {
            return Err(Error::InvalidCount {
                cmd: "ClosePath",
                count,
            });
        }
        coords.push(coord(sx, sy));

        // Ring classification follows MapLibre's `classifyRings` rather
        // than the letter of § 4.3.4.4: zero-area rings are dropped, and
        // the winding of the first remaining ring is what marks an
        // exterior. Real tiles (e.g. tilemaker output) contain both
        // cases, and rejecting them would blank the whole tile.
        let area = signed_area(&coords);
        if area == 0.0 {
            continue;
        }
        let positive = area > 0.0;
        let exterior_is_positive = *exterior_winding.get_or_insert(positive);
        let ring = LineString::new(coords);
        match polys.last_mut() {
            Some((_, interiors)) if positive != exterior_is_positive => interiors.push(ring),
            _ => polys.push((ring, Vec::new())),
        }
    }

    if polys.is_empty() {
        return Err(Error::EmptyGeometry {
            geom_type: "POLYGON",
        });
    }
    let polygons: Vec<Polygon<f64>> = polys
        .into_iter()
        .map(|(ext, ints)| Polygon::new(ext, ints))
        .collect();
    Ok(if polygons.len() == 1 {
        Geometry::Polygon(polygons.into_iter().next().expect("len == 1"))
    } else {
        Geometry::MultiPolygon(MultiPolygon(polygons))
    })
}

// Signed shoelace area of a closed ring (first == last). The sign tells
// us whether the ring is wound clockwise (positive in tile-space Y-down,
// i.e. exterior) or counter-clockwise (negative, i.e. interior).
fn signed_area(coords: &[Coord<f64>]) -> f64 {
    if coords.len() < 4 {
        return 0.0;
    }
    let mut sum = 0.0;
    for w in coords.windows(2) {
        sum += w[0].x * w[1].y - w[1].x * w[0].y;
    }
    sum / 2.0
}

// ==============================================================================
// Tests
//
// Every numeric input below is lifted directly from the MVT 2.1 spec
// § 4.3.5 examples; the expected coordinates come from the same section.
// ==============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use geo::{line_string, polygon};

    fn pt(x: f64, y: f64) -> Point<f64> {
        Point::new(x, y)
    }

    #[test]
    fn spec_4_3_5_1_point() {
        let g = decode_geometry(proto::tile::GeomType::Point, &[9, 50, 34]).unwrap();
        assert_eq!(g, Geometry::Point(pt(25.0, 17.0)));
    }

    #[test]
    fn spec_4_3_5_2_multipoint() {
        let g = decode_geometry(proto::tile::GeomType::Point, &[17, 10, 14, 3, 9]).unwrap();
        assert_eq!(
            g,
            Geometry::MultiPoint(MultiPoint(vec![pt(5.0, 7.0), pt(3.0, 2.0)]))
        );
    }

    #[test]
    fn spec_4_3_5_3_linestring() {
        let g = decode_geometry(
            proto::tile::GeomType::Linestring,
            &[9, 4, 4, 18, 0, 16, 16, 0],
        )
        .unwrap();
        assert_eq!(
            g,
            Geometry::LineString(line_string![
                (x: 2.0, y: 2.0),
                (x: 2.0, y: 10.0),
                (x: 10.0, y: 10.0),
            ])
        );
    }

    #[test]
    fn spec_4_3_5_4_multilinestring() {
        let g = decode_geometry(
            proto::tile::GeomType::Linestring,
            &[9, 4, 4, 18, 0, 16, 16, 0, 9, 17, 17, 10, 4, 8],
        )
        .unwrap();
        let expected = MultiLineString(vec![
            line_string![(x: 2.0, y: 2.0), (x: 2.0, y: 10.0), (x: 10.0, y: 10.0)],
            line_string![(x: 1.0, y: 1.0), (x: 3.0, y: 5.0)],
        ]);
        assert_eq!(g, Geometry::MultiLineString(expected));
    }

    #[test]
    fn spec_4_3_5_5_polygon() {
        let g = decode_geometry(
            proto::tile::GeomType::Polygon,
            &[9, 6, 12, 18, 10, 12, 24, 44, 15],
        )
        .unwrap();
        assert_eq!(
            g,
            Geometry::Polygon(polygon![
                (x: 3.0, y: 6.0),
                (x: 8.0, y: 12.0),
                (x: 20.0, y: 34.0),
            ])
        );
    }

    #[test]
    fn spec_4_3_5_6_multipolygon() {
        let g = decode_geometry(
            proto::tile::GeomType::Polygon,
            &[
                9, 0, 0, 26, 20, 0, 0, 20, 19, 0, 15, 9, 22, 2, 26, 18, 0, 0, 18, 17, 0, 15, 9, 4,
                13, 26, 0, 8, 8, 0, 0, 7, 15,
            ],
        )
        .unwrap();

        let outer_a = line_string![
            (x: 0.0, y: 0.0),
            (x: 10.0, y: 0.0),
            (x: 10.0, y: 10.0),
            (x: 0.0, y: 10.0),
            (x: 0.0, y: 0.0),
        ];
        let outer_b = line_string![
            (x: 11.0, y: 11.0),
            (x: 20.0, y: 11.0),
            (x: 20.0, y: 20.0),
            (x: 11.0, y: 20.0),
            (x: 11.0, y: 11.0),
        ];
        let hole_b = line_string![
            (x: 13.0, y: 13.0),
            (x: 13.0, y: 17.0),
            (x: 17.0, y: 17.0),
            (x: 17.0, y: 13.0),
            (x: 13.0, y: 13.0),
        ];
        let expected = MultiPolygon(vec![
            Polygon::new(outer_a, vec![]),
            Polygon::new(outer_b, vec![hole_b]),
        ]);
        assert_eq!(g, Geometry::MultiPolygon(expected));
    }

    #[test]
    fn unknown_geom_type_round_trips() {
        let g = decode_geometry(proto::tile::GeomType::Unknown, &[9, 0, 0]).unwrap();
        assert_eq!(g, Geometry::Unknown);
    }

    #[test]
    fn rejects_unknown_command_id() {
        // Word 3 → id 3 (reserved). Should fire InvalidCommandId.
        let err = decode_geometry(proto::tile::GeomType::Point, &[3, 0, 0]).unwrap_err();
        assert_eq!(err, Error::InvalidCommandId { id: 3 });
    }

    #[test]
    fn rejects_empty_point_stream() {
        let err = decode_geometry(proto::tile::GeomType::Point, &[]).unwrap_err();
        assert_eq!(err, Error::EmptyGeometry { geom_type: "POINT" });
    }

    #[test]
    fn rejects_polygon_starting_with_lineto() {
        // Begin with LineTo(2) instead of MoveTo(1).
        let err = decode_geometry(proto::tile::GeomType::Polygon, &[18, 0, 0, 2, 0]).unwrap_err();
        assert_eq!(
            err,
            Error::UnexpectedCommand {
                cmd: "LineTo",
                geom_type: "POLYGON"
            }
        );
    }

    #[test]
    fn rejects_truncated_linestring_params() {
        // MoveTo(1) ok, LineTo(2) declared but only one pair follows.
        let err =
            decode_geometry(proto::tile::GeomType::Linestring, &[9, 2, 2, 18, 2, 2]).unwrap_err();
        assert!(matches!(err, Error::Truncated { .. }));
    }

    #[test]
    fn polygon_of_only_degenerate_rings_is_empty() {
        // Triangle collapsed to a line: (0,0) → (10,0) → (20,0) → close.
        // Signed area = 0, so the ring is skipped and nothing is left.
        let err = decode_geometry(
            proto::tile::GeomType::Polygon,
            &[9, 0, 0, 18, 20, 0, 20, 0, 15],
        )
        .unwrap_err();
        assert_eq!(err, Error::EmptyGeometry { geom_type: "POLYGON" });
    }

    #[test]
    fn degenerate_ring_is_skipped() {
        // Exterior square (0,0) → (10,0) → (10,10) → (0,10), then a
        // collapsed ring starting at (0,0) that runs along y = 0.
        let geom = decode_geometry(
            proto::tile::GeomType::Polygon,
            &[
                9, 0, 0, 26, 20, 0, 0, 20, 19, 0, 15, //
                9, 0, 19, 18, 20, 0, 20, 0, 15,
            ],
        )
        .unwrap();
        let Geometry::Polygon(p) = geom else { panic!("{geom:?}") };
        assert_eq!(p.exterior().0.len(), 5);
        assert!(p.interiors().is_empty());
    }

    #[test]
    fn first_ring_winding_defines_exterior() {
        // Counter-clockwise ring first: (0,0) → (0,10) → (10,0) → close,
        // signed area = -50. Like MapLibre's `classifyRings`, the first
        // ring's winding marks exteriors, so this is a polygon, not an
        // error.
        let geom = decode_geometry(
            proto::tile::GeomType::Polygon,
            &[9, 0, 0, 18, 0, 20, 20, 19, 15],
        )
        .unwrap();
        let Geometry::Polygon(p) = geom else { panic!("{geom:?}") };
        assert_eq!(p.exterior().0.len(), 4);
    }

    #[test]
    fn zigzag_decodes_spec_examples() {
        // Hand-checked from § 4.3.5 commentary.
        assert_eq!(zigzag(50), 25);
        assert_eq!(zigzag(34), 17);
        assert_eq!(zigzag(3), -2);
        assert_eq!(zigzag(9), -5);
        assert_eq!(zigzag(0), 0);
    }
}
