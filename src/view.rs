// ==============================================================================
// MapView: the GPUI entity
// ==============================================================================
//
// `MapView` owns the camera, the style, the tile source and the tile cache.
// It renders as a `canvas` whose two phases split the work:
//
// 1. Prepaint (has the element bounds and `&mut App`): work out which
//    tiles cover the viewport, request missing ones on the background
//    executor, pick a loaded ancestor to stand in for each tile still in
//    flight, and resolve every layer's per-frame paint (colour, width) at
//    the exact fractional zoom. The result is a self-contained `FramePlan`.
// 2. Paint: project each tile's mesh into screen space and hand GPUI one
//    path per batch, in style-layer order, clipped to the tile's rect.
//
// Everything that's drawn is a GPUI path, so the map is re-rasterised from
// vectors on every frame and stays sharp at any zoom. The price is that
// GPUI copies every submitted vertex several times per frame, so the paint
// loop culls triangles against the clip rect before submitting them.

use std::collections::HashMap;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::*;
use crate::style::{EvalContext, LayerKind, Style, Visibility};
use crate::mvt::{Feature, Geometry};

use crate::camera::Camera;
use crate::mesh::{self, BatchColor, LayerMesh, LineWidth, MESH_EXTENT, MeshDetail, TileMesh};
use crate::source::TileJsonSource;
use crate::text;
use crate::tile::TileId;

/// Tiles kept in memory beyond those on screen.
const CACHE_CAPACITY: usize = 256;
/// Background tile builds allowed at once; more queue up for later frames.
const MAX_IN_FLIGHT: usize = 16;
/// How many levels up to look for a loaded tile to stand in for one that
/// is still loading.
const MAX_FALLBACK_LEVELS: u8 = 6;
/// How many levels down to look for loaded tiles to stand in for one that
/// is still loading. One level is what a zoom-out just showed; deeper
/// levels multiply the geometry drawn in place of one tile by four each.
const MAX_FALLBACK_DESCENT: u8 = 1;
/// Scroll distance, in logical pixels, that zooms by one level.
const SCROLL_PX_PER_ZOOM: f64 = 300.0;

pub struct MapView {
    camera: Camera,
    style: Arc<Style>,
    source: Arc<TileJsonSource>,
    tiles: HashMap<MeshKey, TileEntry>,
    in_flight: usize,
    frame: u64,
    /// Bounds of the last paint, for mapping pointer events to the map.
    bounds: Bounds<Pixels>,
    drag_from: Option<Point<Pixels>>,
    stats: Rc<Cell<FrameStats>>,
    labels: text::Labels,
}

/// What the last painted frame cost, for overlays and benchmarks.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Tiles drawn, counting ancestors standing in for loading tiles.
    pub tiles: usize,
    /// Visible tiles still being fetched or built (drawn from an ancestor
    /// or not at all meanwhile).
    pub loading: usize,
    /// Visible tiles drawn with nothing at all this frame: neither they nor
    /// any stand-in were loaded. Each is a hole in the map.
    pub uncovered: usize,
    /// Labels that vanished this frame while still at least 30% opaque,
    /// rather than fading out. Each is a visible label blink.
    pub label_pops: usize,
    pub paths: usize,
    /// Triangles handed to GPUI after culling. GPUI's own per-frame cost
    /// grows linearly with this.
    pub triangles: usize,
    pub prepare: Duration,
    pub paint: Duration,
    /// Labels laid out this frame, and those drawn (placed or fading).
    pub label_candidates: usize,
    pub labels: usize,
    /// Glyph sprites submitted for labels, halo copies included.
    pub glyph_sprites: usize,
    /// Part of `prepare` spent laying out and placing labels.
    pub label_layout: Duration,
}

/// Tiles from a source's deepest level are meshed once per detail level,
/// so the cache is keyed by both.
type MeshKey = (TileId, MeshDetail);

struct TileEntry {
    state: TileState,
    last_used: u64,
}

enum TileState {
    Loading,
    Ready {
        mesh: Arc<TileMesh>,
        /// Raw MVT bytes of deepest-level tiles, kept so building another
        /// detail level doesn't fetch the tile again.
        bytes: Option<Arc<[u8]>>,
    },
    /// The source has no tile here, or it failed to load. Either way there
    /// is nothing to draw and nothing to retry until the cache drops it.
    Absent,
}

impl MapView {
    pub fn new(source: Arc<TileJsonSource>, style: Arc<Style>, camera: Camera) -> Self {
        Self {
            camera,
            style,
            source,
            tiles: HashMap::new(),
            in_flight: 0,
            frame: 0,
            bounds: Bounds::default(),
            drag_from: None,
            stats: Rc::default(),
            labels: text::Labels::default(),
        }
    }

    pub fn last_frame_stats(&self) -> FrameStats {
        self.stats.get()
    }

    pub fn camera(&self) -> Camera {
        self.camera
    }

    pub fn set_camera(&mut self, camera: Camera, cx: &mut Context<Self>) {
        self.camera = camera;
        cx.notify();
    }

    /// Swap the style. Meshes bake style-dependent data, so the cache is
    /// dropped; in-flight builds for the old style land and are discarded
    /// by the cache's normal eviction.
    pub fn set_style(&mut self, style: Arc<Style>, cx: &mut Context<Self>) {
        self.style = style;
        self.tiles.clear();
        self.labels.clear();
        cx.notify();
    }

    // --------------------------------------------------------------------------
    // Frame preparation
    // --------------------------------------------------------------------------

    fn prepare(&mut self, bounds: Bounds<Pixels>, window: &mut Window, cx: &mut Context<Self>) -> FramePlan {
        self.frame += 1;
        self.bounds = bounds;
        let viewport = [f64::from(f32::from(bounds.size.width)), f64::from(f32::from(bounds.size.height))];
        let (min_z, max_z) = self.source.zoom_range();

        let mut tiles = Vec::new();
        let mut loading = 0;
        let mut uncovered = 0;
        for id in self.camera.tile_cover(viewport, min_z, max_z) {
            let key = (id, MeshDetail::for_camera(id.z, max_z, self.camera.zoom));
            match self.tiles.get_mut(&key) {
                Some(entry) => entry.last_used = self.frame,
                None if self.in_flight < MAX_IN_FLIGHT => self.request(key, cx),
                None => {}
            }
            if let Some(mesh) = self.use_mesh(key) {
                tiles.push(DrawTile { geometry: id, clip: id, mesh });
                continue;
            }
            if !matches!(self.tiles.get(&key), Some(TileEntry { state: TileState::Absent, .. })) {
                loading += 1;
            }
            // While the wanted detail builds, any other detail of the same
            // tile is the closest match.
            let other_detail = Self::details().find_map(|detail| self.use_mesh((id, detail)));
            if let Some(mesh) = other_detail {
                tiles.push(DrawTile { geometry: id, clip: id, mesh });
                continue;
            }
            // Else loaded descendants stand in, each clipped to its own
            // area: after zooming out, those are the tiles that were just on
            // screen. What they don't cover falls to the nearest loaded
            // ancestor, clipped to the gap; ancestors are above the deepest
            // level, so `Regular`.
            let mut gaps = Vec::new();
            if !self.use_descendants(id, max_z, MAX_FALLBACK_DESCENT, &mut tiles, &mut gaps) {
                gaps = vec![id];
            }
            let ancestor = std::iter::successors(id.parent(), |t| t.parent())
                .take(MAX_FALLBACK_LEVELS as usize)
                .find_map(|a| self.use_mesh((a, MeshDetail::Regular)).map(|mesh| (a, mesh)));
            match ancestor {
                Some((ancestor, mesh)) => {
                    tiles.extend(gaps.into_iter().map(|gap| DrawTile { geometry: ancestor, clip: gap, mesh: mesh.clone() }));
                }
                None if !matches!(self.tiles.get(&key), Some(TileEntry { state: TileState::Absent, .. })) => {
                    uncovered += gaps.len();
                }
                None => {}
            }
        }
        self.prefetch_parents(viewport, min_z, max_z, cx);
        self.evict();

        let tiles = place_tiles(self.camera, viewport, bounds, tiles);
        let label_started = Instant::now();
        let label_frame = self.labels.frame(&self.style, self.camera, viewport, bounds, &tiles, window);
        let label_layout = label_started.elapsed();
        if label_frame.changing {
            window.request_animation_frame();
        }

        FramePlan {
            background: resolve_background(&self.style, self.camera.zoom),
            layers: resolve_layer_paints(&self.style, self.camera.zoom),
            tiles,
            labels: label_frame.draws,
            label_candidates: label_frame.candidates,
            label_pops: label_frame.pops,
            label_layout,
            loading,
            uncovered,
        }
    }

    /// Fetch the parent level of what's on screen, once the visible tiles
    /// are all requested. A quarter as many tiles, and they are what a
    /// zoom-out needs next: they stand in for (and become) the next level's
    /// tiles, and cover the edges zooming out reveals, so the map doesn't
    /// blank while those load. Kept warm by marking them used.
    fn prefetch_parents(&mut self, viewport: [f64; 2], min_z: u8, max_z: u8, cx: &mut Context<Self>) {
        let z = self.camera.tile_zoom(min_z, max_z);
        if z <= min_z {
            return;
        }
        let parent_camera = Camera::new(self.camera.center, self.camera.zoom.min(z as f64 + 0.999) - 1.0);
        for id in parent_camera.tile_cover(viewport, min_z, max_z) {
            let key = (id, MeshDetail::Regular);
            match self.tiles.get_mut(&key) {
                Some(entry) => entry.last_used = self.frame,
                None if self.in_flight < MAX_IN_FLIGHT => self.request(key, cx),
                None => return,
            }
        }
    }

    /// Draw whatever loaded descendants of `id` cover it, down to `levels`
    /// below, each clipped to itself, and collect the parts nothing covers
    /// in `gaps`. Returns whether anything was drawn.
    fn use_descendants(
        &mut self,
        id: TileId,
        max_z: u8,
        levels: u8,
        tiles: &mut Vec<DrawTile>,
        gaps: &mut Vec<TileId>,
    ) -> bool {
        if levels == 0 || id.z >= max_z {
            gaps.push(id);
            return false;
        }
        let mut drawn = false;
        for child in id.children() {
            if let Some(mesh) = Self::details().find_map(|detail| self.use_mesh((child, detail))) {
                tiles.push(DrawTile { geometry: child, clip: child, mesh });
                drawn = true;
            } else {
                drawn |= self.use_descendants(child, max_z, levels - 1, tiles, gaps);
            }
        }
        drawn
    }

    /// Every detail a mesh can be built at, regular first.
    fn details() -> impl Iterator<Item = MeshDetail> {
        std::iter::once(MeshDetail::Regular)
            .chain((0..=MeshDetail::FINEST_OVERZOOM).map(|levels| MeshDetail::Overzoomed { levels }))
    }

    /// The mesh for `key` if it is loaded, marking it as drawn this frame.
    fn use_mesh(&mut self, key: MeshKey) -> Option<Arc<TileMesh>> {
        let entry = self.tiles.get_mut(&key)?;
        match &entry.state {
            TileState::Ready { mesh, .. } => {
                entry.last_used = self.frame;
                Some(mesh.clone())
            }
            TileState::Loading | TileState::Absent => None,
        }
    }

    fn request(&mut self, key: MeshKey, cx: &mut Context<Self>) {
        let (id, detail) = key;
        let cached_bytes = Self::details().find_map(|d| match &self.tiles.get(&(id, d))?.state {
            TileState::Ready { bytes, .. } => bytes.clone(),
            TileState::Loading | TileState::Absent => None,
        });
        self.tiles.insert(key, TileEntry { state: TileState::Loading, last_used: self.frame });
        self.in_flight += 1;
        let source = self.source.clone();
        let style = self.style.clone();
        let build = cx
            .background_executor()
            .spawn(async move { load_tile(&source, &style, id, detail, cached_bytes).await });
        cx.spawn(async move |this, cx| {
            let state = match build.await {
                Ok(Some((mesh, bytes))) => TileState::Ready {
                    mesh: Arc::new(mesh),
                    bytes: matches!(detail, MeshDetail::Overzoomed { .. }).then_some(bytes),
                },
                Ok(None) => TileState::Absent,
                Err(error) => {
                    log::warn!("tile {id:?} failed to load: {error}");
                    TileState::Absent
                }
            };
            // The view may have been dropped while the tile built.
            let _ = this.update(cx, |this, cx| {
                this.in_flight -= 1;
                if let Some(entry) = this.tiles.get_mut(&key) {
                    entry.state = state;
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Drop the least recently drawn tiles once over capacity. Tiles drawn
    /// this frame and builds still in flight are kept.
    fn evict(&mut self) {
        if self.tiles.len() <= CACHE_CAPACITY {
            return;
        }
        let mut candidates: Vec<(u64, MeshKey)> = self
            .tiles
            .iter()
            .filter(|(_, e)| e.last_used < self.frame && !matches!(e.state, TileState::Loading))
            .map(|(id, e)| (e.last_used, *id))
            .collect();
        candidates.sort_unstable_by_key(|(last_used, _)| *last_used);
        let excess = self.tiles.len() - CACHE_CAPACITY;
        for (_, key) in candidates.into_iter().take(excess) {
            self.tiles.remove(&key);
        }
    }

    // --------------------------------------------------------------------------
    // Input
    // --------------------------------------------------------------------------

    fn local(&self, position: Point<Pixels>) -> [f64; 2] {
        let p = position - self.bounds.origin;
        [f32::from(p.x) as f64, f32::from(p.y) as f64]
    }

    fn viewport(&self) -> [f64; 2] {
        [f32::from(self.bounds.size.width) as f64, f32::from(self.bounds.size.height) as f64]
    }

    fn on_scroll(&mut self, event: &ScrollWheelEvent, _: &mut Window, cx: &mut Context<Self>) {
        let delta = event.delta.pixel_delta(px(20.0));
        let zoom_delta = f32::from(delta.y) as f64 / SCROLL_PX_PER_ZOOM;
        self.camera.zoom_at(self.local(event.position), self.viewport(), zoom_delta);
        cx.notify();
    }

    fn on_mouse_down(&mut self, event: &MouseDownEvent, _: &mut Window, _: &mut Context<Self>) {
        self.drag_from = Some(event.position);
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(from) = self.drag_from else { return };
        if event.pressed_button != Some(MouseButton::Left) {
            self.drag_from = None;
            return;
        }
        let delta = event.position - from;
        self.camera.pan([f32::from(delta.x) as f64, f32::from(delta.y) as f64]);
        self.drag_from = Some(event.position);
        cx.notify();
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.drag_from = None;
    }
}

/// Fetch (unless `cached_bytes` already has it), decode and mesh one tile,
/// returning the bytes too so other detail levels can reuse them. Runs on
/// the background executor; awaiting the download parks the task
/// without holding a thread.
async fn load_tile(
    source: &TileJsonSource,
    style: &Style,
    id: TileId,
    detail: MeshDetail,
    cached_bytes: Option<Arc<[u8]>>,
) -> Result<Option<(TileMesh, Arc<[u8]>)>, Box<dyn std::error::Error + Send + Sync>> {
    let bytes = match cached_bytes {
        Some(bytes) => bytes,
        None => match source.tile(id).await? {
            Some(bytes) => Arc::from(bytes),
            None => return Ok(None),
        },
    };
    let tile = mesh::decode_tile(&bytes)?;
    Ok(Some((mesh::build_tile_mesh(&tile, id, style, detail), bytes)))
}

impl Render for MapView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
        let stats = self.stats.clone();
        div()
            .size_full()
            .on_scroll_wheel(cx.listener(Self::on_scroll))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .child(
                canvas(
                    move |bounds, window, cx| {
                        let started = Instant::now();
                        let plan = view.update(cx, |view, cx| view.prepare(bounds, window, cx));
                        (plan, started.elapsed())
                    },
                    move |bounds, (plan, prepare), window, _| {
                        let started = Instant::now();
                        let mut frame = FrameStats {
                            tiles: plan.tiles.len(),
                            loading: plan.loading,
                            prepare,
                            label_candidates: plan.label_candidates,
                            label_pops: plan.label_pops,
                            uncovered: plan.uncovered,
                            labels: plan.labels.len(),
                            label_layout: plan.label_layout,
                            ..Default::default()
                        };
                        paint(&plan, bounds, window, &mut frame);
                        frame.paint = started.elapsed();
                        stats.set(frame);
                    },
                )
                .size_full(),
            )
    }
}

// ==============================================================================
// Per-frame paint values
// ==============================================================================

struct FramePlan {
    background: Option<Rgba>,
    /// Indexed like `style.layers`; `None` for layers not drawn this frame.
    layers: Vec<Option<LayerPaint>>,
    tiles: Vec<PlacedTile>,
    labels: Vec<text::LabelDraw>,
    label_candidates: usize,
    label_pops: usize,
    label_layout: Duration,
    loading: usize,
    uncovered: usize,
}

/// A drawn tile with its screen placement for this frame.
pub(crate) struct PlacedTile {
    pub(crate) tile: DrawTile,
    /// The tile's on-screen rect, clipped to the map.
    pub(crate) clip: Bounds<Pixels>,
    pub(crate) transform: Transform,
}

/// Work out where each tile lands on screen this frame.
pub(crate) fn place_tiles(camera: Camera, viewport: [f64; 2], bounds: Bounds<Pixels>, tiles: Vec<DrawTile>) -> Vec<PlacedTile> {
    let world_size = camera.world_size();
    tiles
        .into_iter()
        .map(|tile| {
            let origin = camera.world_to_screen(tile.geometry.origin(), viewport);
            let scale = (tile.geometry.world_extent() * world_size) as f32 / MESH_EXTENT;
            let clip_origin = camera.world_to_screen(tile.clip.origin(), viewport);
            let clip_size = tile.clip.world_extent() * world_size;
            let clip = Bounds::new(
                bounds.origin + point(px(clip_origin[0] as f32), px(clip_origin[1] as f32)),
                size(px(clip_size as f32), px(clip_size as f32)),
            )
            .intersect(&bounds);
            let transform = Transform {
                origin: [f32::from(bounds.origin.x) + origin[0] as f32, f32::from(bounds.origin.y) + origin[1] as f32],
                scale,
                // Data-driven widths are sampled at the tile's zoom and the
                // next; this is how far past the first we are.
                zoom_fraction: (camera.zoom - tile.geometry.z as f64).max(0.0) as f32,
                clip: [
                    f32::from(clip.origin.x),
                    f32::from(clip.origin.y),
                    f32::from(clip.origin.x + clip.size.width),
                    f32::from(clip.origin.y + clip.size.height),
                ],
            };
            PlacedTile { tile, clip, transform }
        })
        .collect()
}

#[derive(Clone)]
pub(crate) struct DrawTile {
    /// The tile whose mesh is drawn; an ancestor of `clip` while the real
    /// tile is loading.
    pub(crate) geometry: TileId,
    pub(crate) clip: TileId,
    pub(crate) mesh: Arc<TileMesh>,
}

/// Feature-constant paint for one layer, resolved at the camera's zoom.
/// Batches marked `BatchColor::Layer` / `LineWidth::Layer` use these.
struct LayerPaint {
    color: Rgba,
    width: f32,
}

/// Style expressions need a feature even when they don't read one.
fn placeholder_feature() -> Feature {
    Feature { id: None, properties: HashMap::new(), geometry: Geometry::Unknown }
}

fn visible_at(layer: &crate::style::Layer, zoom: f64) -> bool {
    layer.visibility == Visibility::Visible
        && layer.minzoom.is_none_or(|min| zoom >= min)
        && layer.maxzoom.is_none_or(|max| zoom < max)
}

fn resolve_layer_paints(style: &Style, zoom: f64) -> Vec<Option<LayerPaint>> {
    let ctx = EvalContext { zoom, ..Default::default() };
    let feature = placeholder_feature();
    style
        .layers
        .iter()
        .map(|layer| {
            if !visible_at(layer, zoom) {
                return None;
            }
            let paint = match &layer.kind {
                LayerKind::Fill(fill) => LayerPaint {
                    color: mesh::resolve_color(&fill.paint.fill_color, &fill.paint.fill_opacity, &ctx, &feature),
                    width: 0.0,
                },
                LayerKind::Line(line) => LayerPaint {
                    color: mesh::resolve_color(&line.paint.line_color, &line.paint.line_opacity, &ctx, &feature),
                    width: mesh::resolve_width(&line.paint.line_width, &ctx, &feature),
                },
                LayerKind::Background(_) | LayerKind::Symbol(_) => return None,
            };
            Some(paint)
        })
        .collect()
}

/// The first visible background layer's colour.
// TODO: `background-pattern`, and backgrounds that aren't the bottom layer.
fn resolve_background(style: &Style, zoom: f64) -> Option<Rgba> {
    let ctx = EvalContext { zoom, ..Default::default() };
    style.layers.iter().filter(|l| visible_at(l, zoom)).find_map(|layer| match &layer.kind {
        LayerKind::Background(bg) => Some(mesh::resolve_color(
            &bg.paint.background_color,
            &bg.paint.background_opacity,
            &ctx,
            &placeholder_feature(),
        )),
        _ => None,
    })
}

// ==============================================================================
// Painting
// ==============================================================================

fn paint(plan: &FramePlan, bounds: Bounds<Pixels>, window: &mut Window, stats: &mut FrameStats) {
    window.with_content_mask(Some(ContentMask { bounds }), |window| {
        if let Some(background) = plan.background {
            window.paint_quad(fill(bounds, background));
        }

        for (index, layer_paint) in plan.layers.iter().enumerate() {
            let Some(layer_paint) = layer_paint else { continue };
            for PlacedTile { tile, clip, transform } in &plan.tiles {
                let Some(Some(layer_mesh)) = tile.mesh.layers.get(index) else { continue };
                window.with_content_mask(Some(ContentMask { bounds: *clip }), |window| match layer_mesh {
                    // Labels are placed and painted in their own pass, above
                    // all geometry.
                    LayerMesh::Symbol(_) => {}
                    LayerMesh::Fill(batches) => {
                        for batch in batches {
                            let color = match batch.color {
                                BatchColor::Layer => layer_paint.color,
                                BatchColor::Feature(c) => c,
                            };
                            if color.a <= 0.0 {
                                continue;
                            }
                            let mut path = PathSink::with_capacity(batch.triangles.len() / 3);
                            for t in batch.triangles.as_chunks::<3>().0 {
                                path.push(transform, [transform.apply(t[0]), transform.apply(t[1]), transform.apply(t[2])]);
                            }
                            path.paint(window, color, stats);
                        }
                    }
                    LayerMesh::Line(batches) => {
                        for batch in batches {
                            let color = match batch.color {
                                BatchColor::Layer => layer_paint.color,
                                BatchColor::Feature(c) => c,
                            };
                            if color.a <= 0.0 {
                                continue;
                            }
                            let mut path = PathSink::with_capacity(batch.vertices.len() / 3);
                            let half_width = |i: usize| match &batch.width {
                                LineWidth::Layer => layer_paint.width / 2.0,
                                LineWidth::PerVertex(widths) => {
                                    interpolate_width(widths[i], transform.zoom_fraction) / 2.0
                                }
                            };
                            for (n, t) in batch.vertices.as_chunks::<3>().0.iter().enumerate() {
                                let v = |k: usize| {
                                    let vertex = &t[k];
                                    let w = half_width(n * 3 + k);
                                    let c = transform.apply(vertex.center);
                                    [c[0] + vertex.normal[0] * w, c[1] + vertex.normal[1] * w]
                                };
                                path.push(transform, [v(0), v(1), v(2)]);
                            }
                            path.paint(window, color, stats);
                        }
                    }
                });
            }
        }

        stats.glyph_sprites = text::paint(&plan.labels, bounds, window);
    });
}

/// Exponential interpolation between the widths sampled at the tile's zoom
/// and the next, extrapolating for overzoomed tiles. Style widths mostly
/// use exponential stops, so this tracks them closely.
// TODO: evaluate the expression's real curve; this assumes one base.
fn interpolate_width([at_z, at_next]: [f32; 2], fraction: f32) -> f32 {
    if at_z <= 0.0 || at_next <= 0.0 {
        at_z + (at_next - at_z) * fraction
    } else {
        at_z * (at_next / at_z).powf(fraction)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Transform {
    pub(crate) origin: [f32; 2],
    pub(crate) scale: f32,
    pub(crate) zoom_fraction: f32,
    /// Left, top, right, bottom of the tile's clip rect in window pixels.
    pub(crate) clip: [f32; 4],
}

impl Transform {
    pub(crate) fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        [self.origin[0] + p[0] * self.scale, self.origin[1] + p[1] * self.scale]
    }

    /// Whether a triangle's bounding box misses the clip rect entirely.
    fn culls(&self, t: &[[f32; 2]; 3]) -> bool {
        let [l, top, r, b] = self.clip;
        (t[0][0] < l && t[1][0] < l && t[2][0] < l)
            || (t[0][0] > r && t[1][0] > r && t[2][0] > r)
            || (t[0][1] < top && t[1][1] < top && t[2][1] < top)
            || (t[0][1] > b && t[1][1] > b && t[2][1] > b)
    }
}

/// Collects visible triangles into a GPUI path. Vertices are written
/// straight into the path's buffer, sized up front, with bounds tracked
/// alongside: `Path::push_triangle` would grow the buffer and union the
/// bounds three times per triangle, which is measurable at 300k triangles.
struct PathSink {
    vertices: Vec<PathVertex<Pixels>>,
    min: [f32; 2],
    max: [f32; 2],
}

impl PathSink {
    /// `triangles` is an upper bound; culled triangles leave it unused.
    fn with_capacity(triangles: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(triangles * 3),
            min: [f32::INFINITY; 2],
            max: [f32::NEG_INFINITY; 2],
        }
    }

    fn push(&mut self, transform: &Transform, t: [[f32; 2]; 3]) {
        if transform.culls(&t) {
            return;
        }
        for v in t {
            self.min = [self.min[0].min(v[0]), self.min[1].min(v[1])];
            self.max = [self.max[0].max(v[0]), self.max[1].max(v[1])];
            self.vertices.push(PathVertex {
                xy_position: point(px(v[0]), px(v[1])),
                // A constant (0, 1) st-coordinate marks a solid,
                // non-curved triangle for GPUI's path rasteriser.
                st_position: point(0., 1.),
                content_mask: ContentMask::default(),
            });
        }
    }

    fn paint(self, window: &mut Window, color: Rgba, stats: &mut FrameStats) {
        if self.vertices.is_empty() {
            return;
        }
        stats.paths += 1;
        stats.triangles += self.vertices.len() / 3;
        let min = point(px(self.min[0]), px(self.min[1]));
        let mut path = Path::new(min);
        path.bounds = Bounds::from_corners(min, point(px(self.max[0]), px(self.max[1])));
        path.vertices = self.vertices;
        window.paint_path(path, color);
    }
}
