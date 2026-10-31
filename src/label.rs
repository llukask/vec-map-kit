// ==============================================================================
// Label geometry
// ==============================================================================
//
// The parts of label placement that are pure geometry, kept free of GPUI so
// they can be tested directly: where along a line labels go, how glyphs
// follow a bent line, how long names wrap, where a box sits relative to its
// anchor, and which boxes collide.
//
// Coordinates are y-down pixels throughout, as on screen.

use std::ops::Range;

use crate::style::TextAnchor;

pub fn path_length(path: &[[f32; 2]]) -> f32 {
    path.windows(2).map(|w| distance(w[0], w[1])).sum()
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

/// Distances along `path` where `symbol-placement: line` labels are
/// anchored: every `spacing`, starting half a spacing in, keeping only
/// anchors with room for `label_length` on both sides. A line shorter than
/// one spacing still gets a single centred anchor if the label fits.
pub fn line_anchors(path: &[[f32; 2]], spacing: f32, label_length: f32) -> Vec<f32> {
    let length = path_length(path);
    if length < label_length || spacing <= 0.0 {
        return Vec::new();
    }
    let half = label_length / 2.0;
    let anchors: Vec<f32> = std::iter::successors(Some(spacing / 2.0), |a| Some(a + spacing))
        .take_while(|&a| a < length)
        .filter(|&a| a >= half && length - a >= half)
        .collect();
    if anchors.is_empty() { vec![length / 2.0] } else { anchors }
}

/// The single anchor for `symbol-placement: line-center`, if the label fits.
pub fn line_center_anchor(path: &[[f32; 2]], label_length: f32) -> Option<f32> {
    let length = path_length(path);
    (length >= label_length).then_some(length / 2.0)
}

/// Where one glyph of a label following a line goes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphPlacement {
    /// Baseline start of the glyph.
    pub origin: [f32; 2],
    /// Where the glyph's horizontal centre meets the line, for collision.
    pub centre: [f32; 2],
    /// Clockwise rotation in radians.
    pub angle: f32,
}

/// One glyph of a laid-out label: its x offset from the label's start and
/// its advance width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphRun {
    pub x: f32,
    pub advance: f32,
}

/// Point and direction at distance `at` along `path`. Distances before the
/// start or past the end extend the first or last segment.
pub(crate) fn point_at(path: &[[f32; 2]], at: f32) -> ([f32; 2], f32) {
    let last = path.len().saturating_sub(2);
    let mut remaining = at;
    for (i, w) in path.windows(2).enumerate() {
        let length = distance(w[0], w[1]);
        if remaining <= length || i == last {
            let t = if length > 0.0 { remaining / length } else { 0.0 };
            let (dx, dy) = (w[1][0] - w[0][0], w[1][1] - w[0][1]);
            return ([w[0][0] + dx * t, w[0][1] + dy * t], dy.atan2(dx));
        }
        remaining -= length;
    }
    (path[0], 0.0)
}

/// How strictly a label must fit its line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathFit {
    /// Within the line's ends, and no sharper turn than `max_angle`.
    Strict,
    /// Anything goes: for a label that's fading out, where vanishing
    /// abruptly because its line got too short looks worse than a label
    /// briefly running past the line's end.
    Overhang,
}

/// How a label sits on its line, for [`place_along_path`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlongPath {
    /// Push along the normal; positive is "below" the line in reading
    /// direction.
    pub baseline_shift: f32,
    /// Sharpest turn allowed between neighbouring glyphs, in radians.
    pub max_angle: f32,
    pub keep_upright: bool,
    pub fit: PathFit,
}

/// Lay a label of `width` along `path`, centred on the distance `anchor`.
/// Each glyph sits with its horizontal centre on the line, rotated to the
/// line's direction there and pushed `baseline_shift` along the normal
/// (positive is "below" the line in reading direction) so the text is
/// vertically centred on it.
///
/// With `PathFit::Strict`, returns `None` when the label runs off either
/// end of the path or any two neighbouring glyphs differ in angle by more
/// than `max_angle` radians.
/// With `keep_upright`, a label whose run would read right-to-left is laid
/// along the reversed path instead, so it never appears upside down.
pub fn place_along_path(
    path: &[[f32; 2]],
    anchor: f32,
    glyphs: &[GlyphRun],
    width: f32,
    options: AlongPath,
) -> Option<Vec<GlyphPlacement>> {
    let AlongPath { baseline_shift, max_angle, keep_upright, fit } = options;
    let length = path_length(path);
    let (start, end) = (anchor - width / 2.0, anchor + width / 2.0);
    if path.len() < 2 || (fit == PathFit::Strict && (start < 0.0 || end > length)) {
        return None;
    }
    if keep_upright && point_at(path, end).0[0] < point_at(path, start).0[0] {
        let reversed: Vec<[f32; 2]> = path.iter().rev().copied().collect();
        let options = AlongPath { keep_upright: false, ..options };
        return place_along_path(&reversed, length - anchor, glyphs, width, options);
    }

    let mut placed: Vec<GlyphPlacement> = Vec::with_capacity(glyphs.len());
    for glyph in glyphs {
        let centre = start + glyph.x + glyph.advance / 2.0;
        let (point, angle) = point_at(path, centre);
        if let Some(previous) = placed.last() {
            let turn = (angle - previous.angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;
            if fit == PathFit::Strict && turn.abs() > max_angle {
                return None;
            }
        }
        let (sin, cos) = angle.sin_cos();
        let half = glyph.advance / 2.0;
        placed.push(GlyphPlacement {
            origin: [point[0] - cos * half - sin * baseline_shift, point[1] - sin * half + cos * baseline_shift],
            centre: point,
            angle,
        });
    }
    Some(placed)
}

/// Break `text` into lines no wider than `max_width` (where possible) at
/// spaces, balancing line widths the way MapLibre does: the number of lines
/// is fixed by the total width, then each break is the space nearest an
/// even share. `x_at` gives the x offset of a byte index in the single-line
/// layout; `width` is that layout's total width.
pub fn wrap_balanced(text: &str, x_at: impl Fn(usize) -> f32, width: f32, max_width: f32) -> Vec<Range<usize>> {
    let spaces: Vec<usize> = text.match_indices(' ').map(|(i, _)| i).collect();
    if width <= max_width || max_width <= 0.0 || spaces.is_empty() {
        return std::iter::once(0..text.len()).collect();
    }
    let lines = (width / max_width).ceil() as usize;
    let mut breaks: Vec<usize> = Vec::new();
    for k in 1..lines {
        let target = width * k as f32 / lines as f32;
        let nearest = spaces
            .iter()
            .copied()
            .filter(|&s| breaks.last().is_none_or(|&b| s > b))
            .min_by(|&a, &b| (x_at(a) - target).abs().total_cmp(&(x_at(b) - target).abs()));
        if let Some(space) = nearest {
            breaks.push(space);
        }
    }
    let mut ranges = Vec::with_capacity(breaks.len() + 1);
    let mut line_start = 0;
    for space in breaks {
        ranges.push(line_start..space);
        line_start = space + 1;
    }
    ranges.push(line_start..text.len());
    ranges
}

/// Offset from a label's anchor point to the top-left corner of its
/// `size`-sized box, per `text-anchor`.
pub fn anchor_offset(anchor: TextAnchor, size: [f32; 2]) -> [f32; 2] {
    let [w, h] = size;
    let x = match anchor {
        TextAnchor::Left | TextAnchor::TopLeft | TextAnchor::BottomLeft => 0.0,
        TextAnchor::Right | TextAnchor::TopRight | TextAnchor::BottomRight => -w,
        TextAnchor::Center | TextAnchor::Top | TextAnchor::Bottom => -w / 2.0,
    };
    let y = match anchor {
        TextAnchor::Top | TextAnchor::TopLeft | TextAnchor::TopRight => 0.0,
        TextAnchor::Bottom | TextAnchor::BottomLeft | TextAnchor::BottomRight => -h,
        TextAnchor::Center | TextAnchor::Left | TextAnchor::Right => -h / 2.0,
    };
    [x, y]
}

/// Axis-aligned box: left, top, right, bottom.
pub type Rect = [f32; 4];

fn overlaps(a: Rect, b: Rect) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

/// A uniform grid over the viewport for testing label boxes against those
/// already placed. Boxes reaching past the viewport are filed under the
/// edge cells, so they still meet anything on screen they overlap.
#[derive(Debug)]
pub struct CollisionGrid {
    cell: f32,
    columns: usize,
    rows: usize,
    cells: Vec<Vec<u32>>,
    boxes: Vec<Rect>,
}

impl CollisionGrid {
    pub fn new(width: f32, height: f32, cell: f32) -> Self {
        let columns = ((width / cell).ceil() as usize).max(1);
        let rows = ((height / cell).ceil() as usize).max(1);
        Self { cell, columns, rows, cells: vec![Vec::new(); columns * rows], boxes: Vec::new() }
    }

    fn cell_range(&self, rect: Rect) -> (Range<usize>, Range<usize>) {
        let index = |v: f32, count: usize| ((v / self.cell).floor().max(0.0) as usize).min(count - 1);
        (
            index(rect[0], self.columns)..index(rect[2], self.columns) + 1,
            index(rect[1], self.rows)..index(rect[3], self.rows) + 1,
        )
    }

    pub fn collides(&self, rect: Rect) -> bool {
        let (columns, rows) = self.cell_range(rect);
        rows.flat_map(|row| columns.clone().map(move |column| row * self.columns + column))
            .any(|cell| self.cells[cell].iter().any(|&i| overlaps(self.boxes[i as usize], rect)))
    }

    pub fn insert(&mut self, rect: Rect) {
        let index = self.boxes.len() as u32;
        self.boxes.push(rect);
        let (columns, rows) = self.cell_range(rect);
        for row in rows {
            for column in columns.clone() {
                self.cells[row * self.columns + column].push(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn anchors_repeat_every_spacing_with_room_for_the_label() {
        let path = [[0.0, 0.0], [1000.0, 0.0]];
        assert!(close(path_length(&path), 1000.0));
        // Half-spacing start: 125, 375, 625, 875.
        assert_eq!(line_anchors(&path, 250.0, 100.0), vec![125.0, 375.0, 625.0, 875.0]);
        // A 300 px label doesn't fit around 125 or 875.
        assert_eq!(line_anchors(&path, 250.0, 300.0), vec![375.0, 625.0]);
        assert!(line_anchors(&path, 250.0, 1200.0).is_empty());
    }

    #[test]
    fn short_lines_still_get_one_centred_anchor_if_the_label_fits() {
        let path = [[0.0, 0.0], [120.0, 0.0]];
        assert_eq!(line_anchors(&path, 250.0, 100.0), vec![60.0]);
        assert_eq!(line_center_anchor(&path, 100.0), Some(60.0));
        assert_eq!(line_center_anchor(&path, 130.0), None);
    }

    fn along(baseline_shift: f32, max_angle: f32, keep_upright: bool, fit: PathFit) -> AlongPath {
        AlongPath { baseline_shift, max_angle, keep_upright, fit }
    }

    fn glyphs(n: usize, advance: f32) -> Vec<GlyphRun> {
        (0..n).map(|i| GlyphRun { x: i as f32 * advance, advance }).collect()
    }

    #[test]
    fn glyphs_follow_a_straight_line() {
        let path = [[0.0, 100.0], [400.0, 100.0]];
        let placed = place_along_path(&path, 200.0, &glyphs(4, 10.0), 40.0, along(3.0, 0.8, true, PathFit::Strict)).expect("fits");
        assert_eq!(placed.len(), 4);
        for (i, g) in placed.iter().enumerate() {
            assert!(close(g.angle, 0.0));
            assert!(close(g.origin[0], 180.0 + 10.0 * i as f32), "{g:?}");
            assert!(close(g.origin[1], 103.0), "{g:?}");
        }
    }

    #[test]
    fn right_to_left_lines_are_flipped_to_stay_upright() {
        let path = [[400.0, 100.0], [0.0, 100.0]];
        let placed = place_along_path(&path, 200.0, &glyphs(4, 10.0), 40.0, along(3.0, 0.8, true, PathFit::Strict)).expect("fits");
        assert!(placed.iter().all(|g| close(g.angle, 0.0)), "{placed:?}");
        assert!(close(placed[0].origin[0], 180.0));
        let upside_down = place_along_path(&path, 200.0, &glyphs(4, 10.0), 40.0, along(3.0, 0.8, false, PathFit::Strict)).expect("fits");
        assert!(upside_down.iter().all(|g| close(g.angle.abs(), std::f32::consts::PI)));
    }

    #[test]
    fn glyphs_turn_with_the_line_within_the_angle_limit() {
        // A right angle at x = 100: along +x, then down (+y).
        let path = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]];
        let label = glyphs(6, 10.0);
        let placed = place_along_path(&path, 100.0, &label, 60.0, along(0.0, 2.0, true, PathFit::Strict)).expect("fits");
        assert!(close(placed[0].angle, 0.0));
        assert!(close(placed[5].angle, std::f32::consts::FRAC_PI_2), "{placed:?}");
        assert_eq!(place_along_path(&path, 100.0, &label, 60.0, along(0.0, 0.5, true, PathFit::Strict)), None);
    }

    #[test]
    fn labels_running_off_the_path_are_rejected() {
        let path = [[0.0, 0.0], [50.0, 0.0]];
        assert_eq!(place_along_path(&path, 10.0, &glyphs(4, 10.0), 40.0, along(0.0, 1.0, true, PathFit::Strict)), None);
    }

    #[test]
    fn fading_labels_may_overhang_the_path_and_bend_sharply() {
        let short = [[0.0, 0.0], [50.0, 0.0]];
        let placed = place_along_path(&short, 10.0, &glyphs(4, 10.0), 40.0, along(0.0, 1.0, true, PathFit::Overhang))
            .expect("overhang allowed");
        assert!(close(placed[0].origin[0], -10.0), "{placed:?}");
        let corner = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]];
        assert!(place_along_path(&corner, 100.0, &glyphs(6, 10.0), 60.0, along(0.0, 0.5, true, PathFit::Overhang)).is_some());
    }

    #[test]
    fn wrapping_balances_lines_at_spaces() {
        let text = "Rathausplatz am Ring Wien";
        // A monospace layout: every byte is 10 px wide.
        let x_at = |byte: usize| byte as f32 * 10.0;
        let width = text.len() as f32 * 10.0;
        let lines = wrap_balanced(text, x_at, width, 150.0);
        let lines: Vec<&str> = lines.iter().map(|r| text[r.clone()].trim()).collect();
        assert_eq!(lines, vec!["Rathausplatz", "am Ring Wien"]);
        assert_eq!(wrap_balanced("Wien", x_at, 40.0, 150.0), vec![0..4]);
        // A single long word cannot be broken.
        assert_eq!(wrap_balanced("Donaudampfschifffahrt", x_at, 210.0, 50.0), vec![0..21]);
    }

    #[test]
    fn anchor_offsets_put_the_anchor_on_the_named_edge() {
        let size = [40.0, 20.0];
        assert_eq!(anchor_offset(TextAnchor::Center, size), [-20.0, -10.0]);
        assert_eq!(anchor_offset(TextAnchor::Top, size), [-20.0, 0.0]);
        assert_eq!(anchor_offset(TextAnchor::BottomRight, size), [-40.0, -20.0]);
        assert_eq!(anchor_offset(TextAnchor::Left, size), [0.0, -10.0]);
    }

    #[test]
    fn grid_finds_overlaps_across_cells() {
        let mut grid = CollisionGrid::new(500.0, 500.0, 64.0);
        assert!(!grid.collides([10.0, 10.0, 100.0, 30.0]));
        grid.insert([10.0, 10.0, 100.0, 30.0]);
        assert!(grid.collides([90.0, 25.0, 200.0, 50.0]));
        assert!(!grid.collides([101.0, 10.0, 200.0, 30.0]));
        // Boxes partly off screen still collide with what's on screen.
        assert!(grid.collides([-50.0, 0.0, 20.0, 20.0]));
        assert!(!grid.collides([-50.0, -50.0, -10.0, -10.0]));
    }
}
