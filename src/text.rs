// ==============================================================================
// Text labels: per-frame layout, placement, fading and painting
// ==============================================================================
//
// Tiles carry label candidates (`mesh::LabelSource`); this module turns them
// into glyphs on screen each frame:
//
// 1. Layout. Each candidate is shaped with GPUI's text system (which caches
//    layouts across frames) and positioned: point labels as a block of
//    balanced, wrapped lines around their anchor; line labels glyph by glyph
//    along the projected line.
// 2. Placement. Candidates go through a collision grid in priority order:
//    upper style layers first, then `symbol-sort-key`. The pass runs at most
//    every `PLACEMENT_INTERVAL`; in between, the last decisions are reused by
//    label identity, so a label doesn't flip on and off every frame while the
//    map zooms.
// 3. Fading. Each label's opacity eases towards its placement decision over
//    `FADE_DURATION`, as in MapLibre, so the changes that do happen blend.
// 4. Painting, above all geometry, through `Window::paint_glyph_instances`
//    (from this repo's GPUI patches): each glyph is rasterised once and drawn
//    rotated along its line and, for halos, as eight offset copies beneath.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::{Duration, Instant};

use gpui_kit::{
    Bounds, Font, FontId, FontStyle, FontWeight, GlyphId, GlyphInstance, Hsla, LineLayout, Pixels, Point, Radians, Rgba,
    TextRun, Window, point, px,
};
use crate::style::{EvalContext, LayerKind, Style, SymbolLayer, TextAnchor, Visibility};
use crate::mvt::{Feature, Geometry};

use crate::label::{self, CollisionGrid, GlyphRun, PathFit, Rect};
use crate::mesh::{self, LabelAnchor, LabelSource, LayerMesh, MESH_EXTENT};
use crate::camera::Camera;
use crate::view::{DrawTile, PlacedTile, place_tiles};

/// Minimum time between collision passes.
const PLACEMENT_INTERVAL: Duration = Duration::from_millis(150);
/// How long a label takes to fade fully in or out.
const FADE_DURATION: Duration = Duration::from_millis(250);
/// Labels anchored further than this outside the map aren't laid out.
const VIEWPORT_MARGIN: f32 = 64.0;
/// GPUI rasterises glyphs per size, so sizes are rounded to this step to
/// keep zoom-interpolated `text-size` from rasterising every frame.
const SIZE_STEP: f32 = 0.5;
const COLLISION_CELL: f32 = 64.0;

// ------------------------------------------------------------------------------
// Style
// ------------------------------------------------------------------------------

/// A symbol layer's text properties together with the font they resolve to.
struct LayerStyle {
    text: LayerText,
    font: Font,
}

/// Feature-constant text properties of one symbol layer at the camera's zoom,
/// with the style spec's defaults filled in.
struct LayerText {
    font_stack: Vec<String>,
    size: f32,
    color: Rgba,
    halo_color: Rgba,
    halo_width: f32,
    anchor: TextAnchor,
    /// `text-offset`, in ems.
    offset: [f32; 2],
    /// `text-max-width` and `text-line-height`, in ems.
    max_width: f32,
    line_height: f32,
    padding: f32,
    /// `text-max-angle`, in radians.
    max_angle: f32,
    keep_upright: bool,
    allow_overlap: bool,
    ignore_placement: bool,
}

/// `in_range_only: false` resolves layers outside their zoom range too, for
/// labels fading out after the camera left it.
fn resolve_layer_text(layer: &crate::style::Layer, zoom: f64, in_range_only: bool) -> Option<LayerText> {
    let LayerKind::Symbol(SymbolLayer { layout, paint }) = &layer.kind else { return None };
    let in_range =
        layer.minzoom.is_none_or(|min| zoom >= min) && layer.maxzoom.is_none_or(|max| zoom < max);
    let visible = layer.visibility == Visibility::Visible
        && (in_range || !in_range_only)
        && layout.text_field.is_some();
    if !visible {
        return None;
    }
    let ctx = EvalContext { zoom, ..Default::default() };
    let feature = Feature { id: None, properties: HashMap::new(), geometry: Geometry::Unknown };
    let number = |p: &Option<crate::style::Property<f32>>, default: f32| {
        p.as_ref().and_then(|p| p.resolve(&ctx, &feature).ok()).unwrap_or(default)
    };
    let halo = paint.text_halo_color.as_ref().and_then(|p| p.resolve(&ctx, &feature).ok());
    Some(LayerText {
        font_stack: layout
            .text_font
            .clone()
            .unwrap_or_else(|| vec!["Open Sans Regular".into(), "Arial Unicode MS Regular".into()]),
        size: number(&layout.text_size, 16.0),
        color: mesh::resolve_color(&paint.text_color, &paint.text_opacity, &ctx, &feature),
        halo_color: halo.map_or(Rgba { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }, |c| Rgba {
            r: c.r as f32 / 255.0,
            g: c.g as f32 / 255.0,
            b: c.b as f32 / 255.0,
            a: c.a as f32 / 255.0 * number(&paint.text_opacity, 1.0),
        }),
        halo_width: number(&paint.text_halo_width, 0.0),
        anchor: layout.text_anchor.unwrap_or_default(),
        offset: layout.text_offset.unwrap_or([0.0, 0.0]),
        max_width: number(&layout.text_max_width, 10.0),
        line_height: number(&layout.text_line_height, 1.2),
        padding: layout.text_padding.as_ref().and_then(|p| p.first().copied()).unwrap_or(2.0),
        max_angle: number(&layout.text_max_angle, 45.0).to_radians(),
        keep_upright: layout.text_keep_upright.unwrap_or(true),
        allow_overlap: layout.text_allow_overlap.unwrap_or(false),
        ignore_placement: layout.text_ignore_placement.unwrap_or(false),
    })
    // TODO: `text-letter-spacing`, `text-justify`, `text-variable-anchor`,
    // `text-rotate` and `text-halo-blur` are not applied yet.
}

/// Split a glyph-server font name like "Open Sans Semi Bold Italic" into
/// a family, weight and style, the way font stacks in styles are written.
pub(crate) fn parse_font_name(name: &str) -> (String, FontWeight, FontStyle) {
    const WEIGHTS: [(&str, f32); 14] = [
        ("Extra Light", 200.0),
        ("ExtraLight", 200.0),
        ("Semi Bold", 600.0),
        ("SemiBold", 600.0),
        ("Semibold", 600.0),
        ("Extra Bold", 800.0),
        ("ExtraBold", 800.0),
        ("Thin", 100.0),
        ("Light", 300.0),
        ("Regular", 400.0),
        ("Medium", 500.0),
        ("Bold", 700.0),
        ("Black", 900.0),
        ("Heavy", 900.0),
    ];
    let mut family = name.trim();
    let mut style = FontStyle::Normal;
    let mut weight = FontWeight::NORMAL;
    for slant in ["Italic", "Oblique"] {
        if let Some(rest) = family.strip_suffix(slant) {
            family = rest.trim_end();
            style = FontStyle::Italic;
        }
    }
    for (word, value) in WEIGHTS {
        if let Some(rest) = family.strip_suffix(word)
            && rest.ends_with(' ')
        {
            family = rest.trim_end();
            weight = FontWeight(value);
            break;
        }
    }
    (family.to_string(), weight, style)
}

// ------------------------------------------------------------------------------
// Per-frame state
// ------------------------------------------------------------------------------

pub(crate) struct LabelFrame {
    pub(crate) draws: Vec<LabelDraw>,
    /// Labels are still fading or awaiting placement; keep requesting frames.
    pub(crate) changing: bool,
    /// Labels laid out this frame.
    pub(crate) candidates: usize,
    /// Labels drawn last frame at 30%+ opacity that vanished this frame.
    pub(crate) pops: usize,
}

/// What to paint for one label this frame.
pub(crate) struct LabelDraw {
    glyphs: Vec<GlyphDraw>,
    font_size: Pixels,
    color: Hsla,
    halo: Option<(Hsla, f32)>,
}

struct GlyphDraw {
    font_id: FontId,
    glyph_id: GlyphId,
    origin: Point<Pixels>,
    rotation: Radians,
}

/// A laid-out label waiting for placement.
struct Candidate {
    key: u64,
    /// Layer and text: labels in the same group near each other across
    /// frames are the same label, even when they come from different tiles.
    group: (usize, u64),
    /// Screen position of the anchor, and font size, for that matching.
    anchor: [f32; 2],
    size: f32,
    origin: LabelOrigin,
    /// A label whose tile is gone, laid out from its old tile while it
    /// fades out. Never placed.
    fading_out: bool,
    glyphs: Vec<GlyphDraw>,
    boxes: Vec<Rect>,
    font_size: Pixels,
    color: Rgba,
    halo: Option<(Rgba, f32)>,
    allow_overlap: bool,
    ignore_placement: bool,
}

/// Where a label came from, so it can be laid out again after its tile has
/// left the frame.
#[derive(Clone)]
struct LabelOrigin {
    layer: usize,
    source: LabelSource,
    tile: DrawTile,
}

/// A label drawn last frame.
struct Shown {
    group: (usize, u64),
    anchor: [f32; 2],
    size: f32,
    origin: LabelOrigin,
}

#[derive(Default)]
pub(crate) struct Labels {
    /// Installed font families, read once.
    families: Option<HashSet<String>>,
    fonts: HashMap<Vec<String>, Font>,
    /// Labels the last placement pass showed, and every label it considered.
    placed: HashSet<u64>,
    decided: HashSet<u64>,
    opacity: HashMap<u64, f32>,
    shown: HashMap<u64, Shown>,
    last_pass: Option<Instant>,
    last_frame: Option<Instant>,
}

impl Labels {
    /// Forget everything tied to the style, e.g. after `set_style`.
    pub(crate) fn clear(&mut self) {
        *self = Self { families: self.families.take(), ..Self::default() };
    }

    /// Lay out, place and fade this frame's labels.
    pub(crate) fn frame(
        &mut self,
        style: &Style,
        camera: Camera,
        viewport: [f64; 2],
        bounds: Bounds<Pixels>,
        tiles: &[PlacedTile],
        window: &mut Window,
    ) -> LabelFrame {
        let now = Instant::now();
        let dt = self.last_frame.map_or(Duration::ZERO, |t| now - t).min(Duration::from_millis(100));
        self.last_frame = Some(now);
        let zoom = camera.zoom;

        // Upper layers are placed first, so e.g. city names win over streets.
        let mut candidates = Vec::new();
        for (index, layer) in style.layers.iter().enumerate().rev() {
            let Some(text) = resolve_layer_text(layer, zoom, true) else { continue };
            let font = self.font(&text.font_stack, window);
            let style = LayerStyle { text, font };
            let mut sources: Vec<(&LabelSource, &PlacedTile)> = tiles
                .iter()
                .filter_map(|tile| match tile.tile.mesh.layers.get(index) {
                    Some(Some(LayerMesh::Symbol(labels))) => Some(labels.iter().map(move |l| (l, tile))),
                    _ => None,
                })
                .flatten()
                .collect();
            sources.sort_by(|a, b| a.0.sort_key.total_cmp(&b.0.sort_key));
            for (source, tile) in sources {
                if let Some(candidate) = layout(index, source, tile, &style, bounds, PathFit::Strict, window) {
                    candidates.push(candidate);
                }
            }
        }

        // Carry labels over between frames. When a tile is replaced (by a
        // stand-in, or by the real tile once loaded) its labels come back
        // under new keys; matching them to last frame's by layer, text and
        // position keeps their opacity and placement, so they don't blink.
        let current: HashSet<u64> = candidates.iter().map(|c| c.key).collect();
        let mut gone: HashMap<(usize, u64), Vec<u64>> = HashMap::new();
        for (&key, shown) in &self.shown {
            if !current.contains(&key) {
                gone.entry(shown.group).or_default().push(key);
            }
        }
        let mut carried = HashSet::new();
        for candidate in &candidates {
            if self.opacity.contains_key(&candidate.key) {
                continue;
            }
            let Some(keys) = gone.get_mut(&candidate.group) else { continue };
            let near = |key: &u64| {
                let previous = &self.shown[key];
                let reach = candidate.size.max(previous.size).max(4.0);
                (previous.anchor[0] - candidate.anchor[0]).hypot(previous.anchor[1] - candidate.anchor[1]) <= reach
            };
            let Some(i) = keys.iter().position(near) else { continue };
            let previous = keys.swap_remove(i);
            carried.insert(previous);
            if let Some(&o) = self.opacity.get(&previous) {
                self.opacity.insert(candidate.key, o);
            }
            if self.placed.contains(&previous) {
                self.placed.insert(candidate.key);
            }
            if self.decided.contains(&previous) {
                self.decided.insert(candidate.key);
            }
        }

        // Labels that left without a successor fade out where they are,
        // laid out from their old tile at the current camera.
        let mut layer_styles: HashMap<usize, Option<LayerStyle>> = HashMap::new();
        for key in gone.into_values().flatten() {
            let Some(shown) = self.shown.get(&key) else { continue };
            let origin = shown.origin.clone();
            let entry = layer_styles.entry(origin.layer).or_insert_with(|| {
                let text = resolve_layer_text(&style.layers[origin.layer], zoom, false)?;
                let font = self.font(&text.font_stack, window);
                Some(LayerStyle { text, font })
            });
            let Some(layer_style) = entry.as_ref() else { continue };
            let Some(tile) = place_tiles(camera, viewport, bounds, vec![origin.tile.clone()]).pop() else { continue };
            let fading = layout(origin.layer, &origin.source, &tile, layer_style, bounds, PathFit::Overhang, window);
            if let Some(mut candidate) = fading {
                candidate.key = key;
                candidate.fading_out = true;
                candidates.push(candidate);
            }
        }

        let pass_due = self.last_pass.is_none_or(|t| now - t >= PLACEMENT_INTERVAL);
        if pass_due {
            let mut grid = CollisionGrid::new(f32::from(bounds.size.width), f32::from(bounds.size.height), COLLISION_CELL);
            let local = |r: Rect| {
                let (x, y) = (f32::from(bounds.origin.x), f32::from(bounds.origin.y));
                [r[0] - x, r[1] - y, r[2] - x, r[3] - y]
            };
            self.placed.clear();
            self.decided.clear();
            for candidate in candidates.iter().filter(|c| !c.fading_out) {
                self.decided.insert(candidate.key);
                let boxes = candidate.boxes.iter().map(|&b| local(b));
                if candidate.allow_overlap || !boxes.clone().any(|b| grid.collides(b)) {
                    self.placed.insert(candidate.key);
                    if !candidate.ignore_placement {
                        boxes.for_each(|b| grid.insert(b));
                    }
                }
            }
            self.last_pass = Some(now);
        }

        let step = dt.as_secs_f32() / FADE_DURATION.as_secs_f32();
        let mut changing = candidates.iter().any(|c| !c.fading_out && !self.decided.contains(&c.key));
        let candidate_count = candidates.len();
        let mut opacity = HashMap::with_capacity(candidates.len());
        let mut shown = HashMap::with_capacity(candidates.len());
        let mut draws = Vec::new();
        for candidate in candidates {
            let target = if !candidate.fading_out && self.placed.contains(&candidate.key) { 1.0 } else { 0.0 };
            let current = self.opacity.get(&candidate.key).copied().unwrap_or(0.0);
            let next = if current < target { (current + step).min(target) } else { (current - step).max(target) };
            changing |= next != target;
            if next > 0.0 {
                opacity.insert(candidate.key, next);
                shown.insert(
                    candidate.key,
                    Shown { group: candidate.group, anchor: candidate.anchor, size: candidate.size, origin: candidate.origin },
                );
                let fade = |c: Rgba| Hsla::from(Rgba { a: c.a * next, ..c });
                draws.push(LabelDraw {
                    glyphs: candidate.glyphs,
                    font_size: candidate.font_size,
                    color: fade(candidate.color),
                    halo: candidate.halo.map(|(c, w)| (fade(c), w)),
                });
            }
        }
        let pops = self
            .opacity
            .iter()
            .filter(|&(key, &o)| o >= 0.3 && !opacity.contains_key(key) && !carried.contains(key))
            .count();
        self.opacity = opacity;
        self.shown = shown;
        LabelFrame { draws, changing, candidates: candidate_count, pops }
    }

    /// The first family of `stack` that's installed, else the system UI
    /// font, with the weight and slant the stack's first name asks for.
    fn font(&mut self, stack: &[String], window: &Window) -> Font {
        if let Some(font) = self.fonts.get(stack) {
            return font.clone();
        }
        let families =
            self.families.get_or_insert_with(|| window.text_system().all_font_names().into_iter().collect());
        let parsed: Vec<_> = stack.iter().map(|name| parse_font_name(name)).collect();
        let (family, weight, style) = parsed
            .iter()
            .find(|(family, _, _)| families.contains(family))
            .cloned()
            .or_else(|| parsed.first().map(|(_, w, s)| (".SystemUIFont".to_string(), *w, *s)))
            .unwrap_or((".SystemUIFont".to_string(), FontWeight::NORMAL, FontStyle::Normal));
        let font = Font { weight, style, ..gpui_kit::font(family) };
        self.fonts.insert(stack.to_vec(), font.clone());
        font
    }
}

// ------------------------------------------------------------------------------
// Layout
// ------------------------------------------------------------------------------

/// A stable identity for a label across frames: its layer, text and where
/// on the world it's anchored.
fn label_key(layer: usize, text: &str, tile: &PlacedTile, mesh_point: [f32; 2]) -> u64 {
    let id = tile.tile.geometry;
    let origin = id.origin();
    let extent = id.world_extent() / MESH_EXTENT as f64;
    // About 1 m at the equator: enough to tell neighbouring labels apart,
    // coarse enough that the same anchor from two tile levels matches.
    let quantum = (1u64 << 25) as f64;
    let x = ((origin.x + mesh_point[0] as f64 * extent) * quantum).round() as i64;
    let y = ((origin.y + mesh_point[1] as f64 * extent) * quantum).round() as i64;
    let mut hasher = DefaultHasher::new();
    (layer, text, x, y).hash(&mut hasher);
    hasher.finish()
}

fn layout(
    layer: usize,
    source: &LabelSource,
    tile: &PlacedTile,
    style: &LayerStyle,
    bounds: Bounds<Pixels>,
    fit: PathFit,
    window: &Window,
) -> Option<Candidate> {
    let LayerStyle { text, font } = style;
    let size = (source.size.unwrap_or(text.size) / SIZE_STEP).round() * SIZE_STEP;
    let color = source.color.unwrap_or(text.color);
    if size < 1.0 || color.a <= 0.0 {
        return None;
    }
    let font_size = px(size);
    let shape = |s: &str| {
        let run = TextRun {
            len: s.len(),
            font: font.clone(),
            color: Hsla::black(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        window.text_system().layout_line(s, font_size, &[run], None)
    };
    let line = shape(&source.text);
    let glyphs_of = |layout: &LineLayout| -> Vec<(FontId, GlyphId, f32, usize)> {
        layout
            .runs
            .iter()
            .flat_map(|run| run.glyphs.iter().map(move |g| (run.font_id, g.id, f32::from(g.position.x), g.index)))
            .collect()
    };
    let ascent = f32::from(line.ascent);
    let descent = f32::from(line.descent);
    let clip = tile.clip;
    let in_clip = |p: [f32; 2]| clip.contains(&point(px(p[0]), px(p[1])));
    let near_view = |r: Rect| {
        let b = bounds.dilate(px(VIEWPORT_MARGIN));
        r[2] >= f32::from(b.left()) && r[0] <= f32::from(b.right()) && r[3] >= f32::from(b.top()) && r[1] <= f32::from(b.bottom())
    };

    let (key, glyphs, boxes, screen_anchor) = match &source.anchor {
        LabelAnchor::Point(p) => {
            let anchor = tile.transform.apply(*p);
            if !in_clip(anchor) {
                return None;
            }
            let width = f32::from(line.width);
            let x_at = |byte: usize| {
                glyphs_of(&line).iter().find(|g| g.3 >= byte).map_or(width, |g| g.2)
            };
            let ranges = label::wrap_balanced(&source.text, x_at, width, text.max_width * size);
            let lines: Vec<_> = if ranges.len() == 1 {
                vec![line.clone()]
            } else {
                ranges.iter().map(|r| shape(source.text[r.clone()].trim())).collect()
            };
            let line_height = text.line_height * size;
            let block = [lines.iter().map(|l| f32::from(l.width)).fold(0.0, f32::max), line_height * lines.len() as f32];
            let offset = label::anchor_offset(text.anchor, block);
            let left = anchor[0] + text.offset[0] * size + offset[0];
            let top = anchor[1] + text.offset[1] * size + offset[1];
            let pad = text.padding;
            let rect = [left - pad, top - pad, left + block[0] + pad, top + block[1] + pad];
            if !near_view(rect) {
                return None;
            }
            let mut glyphs = Vec::new();
            for (i, l) in lines.iter().enumerate() {
                let slack = block[0] - f32::from(l.width);
                let justify = match text.anchor {
                    TextAnchor::Left | TextAnchor::TopLeft | TextAnchor::BottomLeft => 0.0,
                    TextAnchor::Right | TextAnchor::TopRight | TextAnchor::BottomRight => slack,
                    TextAnchor::Center | TextAnchor::Top | TextAnchor::Bottom => slack / 2.0,
                };
                let baseline = top + line_height * (i as f32 + 0.5) + (ascent - descent) / 2.0;
                glyphs.extend(glyphs_of(l).into_iter().map(|(font_id, glyph_id, x, _)| GlyphDraw {
                    font_id,
                    glyph_id,
                    origin: point(px(left + justify + x), px(baseline)),
                    rotation: Radians(0.0),
                }));
            }
            (label_key(layer, &source.text, tile, *p), glyphs, vec![rect], anchor)
        }
        LabelAnchor::Line { path, at } => {
            let screen_path: Vec<[f32; 2]> = path.iter().map(|&p| tile.transform.apply(p)).collect();
            let at = at * tile.transform.scale;
            let anchor = label::point_at(&screen_path, at).0;
            let half = size / 2.0;
            if !in_clip(anchor) || !near_view([anchor[0] - half, anchor[1] - half, anchor[0] + half, anchor[1] + half]) {
                return None;
            }
            let width = f32::from(line.width);
            let shaped = glyphs_of(&line);
            let runs: Vec<GlyphRun> = shaped
                .iter()
                .enumerate()
                .map(|(i, g)| GlyphRun { x: g.2, advance: shaped.get(i + 1).map_or(width, |n| n.2) - g.2 })
                .collect();
            let options = label::AlongPath {
                baseline_shift: (ascent - descent) / 2.0 + text.offset[1] * size,
                max_angle: text.max_angle,
                keep_upright: text.keep_upright,
                fit,
            };
            let placed = label::place_along_path(&screen_path, at, &runs, width, options)?;
            let boxes = placed
                .iter()
                .map(|g| [g.centre[0] - half, g.centre[1] - half, g.centre[0] + half, g.centre[1] + half])
                .collect();
            let glyphs = shaped
                .iter()
                .zip(&placed)
                .map(|(&(font_id, glyph_id, _, _), g)| GlyphDraw {
                    font_id,
                    glyph_id,
                    origin: point(px(g.origin[0]), px(g.origin[1])),
                    rotation: Radians(g.angle),
                })
                .collect();
            let mesh_anchor = label::point_at(path, at / tile.transform.scale).0;
            (label_key(layer, &source.text, tile, mesh_anchor), glyphs, boxes, anchor)
        }
    };

    let halo = (text.halo_width > 0.0 && text.halo_color.a > 0.0).then_some((text.halo_color, text.halo_width));
    let mut text_hash = DefaultHasher::new();
    source.text.hash(&mut text_hash);
    Some(Candidate {
        key,
        group: (layer, text_hash.finish()),
        anchor: screen_anchor,
        size,
        origin: LabelOrigin { layer, source: source.clone(), tile: tile.tile.clone() },
        fading_out: false,
        glyphs,
        boxes,
        font_size,
        color,
        halo,
        allow_overlap: text.allow_overlap,
        ignore_placement: text.ignore_placement,
    })
}

// ------------------------------------------------------------------------------
// Painting
// ------------------------------------------------------------------------------

/// Paint labels: every halo first, then every glyph, each set in its own
/// GPUI layer. Inside a layer all primitives share one draw order, which
/// skips GPUI's per-primitive overlap search; with eight halo copies per
/// glyph that search was most of the cost. It also puts all halos beneath
/// all text. Returns the number of glyph sprites submitted.
pub(crate) fn paint(labels: &[LabelDraw], bounds: Bounds<Pixels>, window: &mut Window) -> usize {
    const RING: [[f32; 2]; 8] =
        [[1.0, 0.0], [0.707, 0.707], [0.0, 1.0], [-0.707, 0.707], [-1.0, 0.0], [-0.707, -0.707], [0.0, -1.0], [0.707, -0.707]];
    let mut sprites = 0;
    window.paint_layer(bounds, |window| {
        for label in labels {
            let Some((color, width)) = label.halo else { continue };
            sprites += label.glyphs.len() * RING.len();
            for g in &label.glyphs {
                let copies = RING.iter().map(|o| GlyphInstance {
                    origin: g.origin + point(px(o[0] * width), px(o[1] * width)),
                    color,
                    rotation: g.rotation,
                });
                if let Err(error) = window.paint_glyph_instances(g.font_id, g.glyph_id, label.font_size, copies) {
                    log::debug!("could not paint label halo: {error}");
                }
            }
        }
    });
    window.paint_layer(bounds, |window| {
        for label in labels {
            sprites += label.glyphs.len();
            for g in &label.glyphs {
                let fill = GlyphInstance { origin: g.origin, color: label.color, rotation: g.rotation };
                if let Err(error) = window.paint_glyph_instances(g.font_id, g.glyph_id, label.font_size, [fill]) {
                    log::debug!("could not paint label: {error}");
                }
            }
        }
    });
    sprites
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_names_split_into_family_weight_and_slant() {
        assert_eq!(parse_font_name("Noto Sans Regular"), ("Noto Sans".into(), FontWeight::NORMAL, FontStyle::Normal));
        assert_eq!(parse_font_name("Open Sans Semi Bold Italic"), ("Open Sans".into(), FontWeight(600.0), FontStyle::Italic));
        assert_eq!(parse_font_name("Open Sans Bold"), ("Open Sans".into(), FontWeight::BOLD, FontStyle::Normal));
        assert_eq!(parse_font_name("Roboto Italic"), ("Roboto".into(), FontWeight::NORMAL, FontStyle::Italic));
        // A family whose name merely ends in a weight word keeps it.
        assert_eq!(parse_font_name("Black"), ("Black".into(), FontWeight::NORMAL, FontStyle::Normal));
    }
}
