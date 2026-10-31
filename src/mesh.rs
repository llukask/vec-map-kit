// ==============================================================================
// Tile meshes
// ==============================================================================
//
// A `TileMesh` is everything the painter needs from one decoded tile,
// built once on a background thread and then re-projected every frame.
// Geometry lives in tile-local "mesh pixels" (`0..=MESH_EXTENT`, the size
// the tile has at its own integer zoom), so the per-frame work is a single
// scale-and-offset per vertex.
//
// Paint values split by what they depend on:
//
// - Feature-constant properties (most colours and widths in real styles,
//   which only vary with zoom) are *not* baked. Batches carry
//   `BatchColor::Layer` / `LineWidth::Layer` and the painter resolves the
//   value once per layer per frame at the exact fractional camera zoom.
// - Data-driven properties are evaluated per feature at build time and
//   baked into the batch (colour) or the vertices (width, sampled at the
//   tile's zoom and the next one so the painter can interpolate).
//
// Lines keep their centreline and lyon's extrusion normal instead of a
// finished outline, so stroke width is applied per frame in screen pixels
// and stays exact at any zoom.

use geo::{BoundingRect, InteriorPoint, Simplify};
use std::sync::Arc;

use gpui_kit::{Rgba, SharedString};
use lyon_tessellation::geom::point as lpoint;
use lyon_tessellation::path::Path as LPath;
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, StrokeOptions, StrokeTessellator,
    StrokeVertex, VertexBuffers,
};
use crate::style::{
    Color, EvalContext, Layer, LayerKind, LineCap, LineJoin, Property, Style, SymbolLayer, SymbolPlacement,
    TextTransform, Visibility,
};
use crate::mvt::{Feature, Geometry};

use crate::label;
use crate::tile::TileId;

/// Edge length of a tile in mesh pixels.
pub const MESH_EXTENT: f32 = 512.0;

#[derive(Debug, Default)]
pub struct TileMesh {
    /// Indexed by style layer; `None` where the layer has nothing in this
    /// tile or is not a fill / line layer.
    pub layers: Vec<Option<LayerMesh>>,
}

#[derive(Debug)]
pub enum LayerMesh {
    Fill(Vec<FillBatch>),
    Line(Vec<LineBatch>),
    Symbol(Vec<LabelSource>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchColor {
    /// Resolve from the style layer at frame time.
    Layer,
    /// Baked from a data-driven expression; opacity already multiplied in.
    Feature(Rgba),
}

#[derive(Debug)]
pub struct FillBatch {
    pub color: BatchColor,
    /// Triangle list, three vertices per triangle.
    pub triangles: Vec<[f32; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineVertex {
    pub center: [f32; 2],
    /// Displacing `center` by `normal * width / 2` lands on the stroke
    /// edge; miter joins make it longer than unit length.
    pub normal: [f32; 2],
}

#[derive(Debug, PartialEq)]
pub enum LineWidth {
    /// Resolve from the style layer at frame time.
    Layer,
    /// Per vertex: width at the tile's zoom and at the next zoom.
    PerVertex(Vec<[f32; 2]>),
}

#[derive(Debug)]
pub struct LineBatch {
    pub color: BatchColor,
    pub width: LineWidth,
    /// Triangle list, three vertices per triangle.
    pub vertices: Vec<LineVertex>,
}

/// Decode MVT bytes, dropping (and logging) layers that fail to decode
/// instead of failing the tile: one malformed feature shouldn't blank
/// every other layer, which is also how MapLibre behaves.
pub fn decode_tile(bytes: &[u8]) -> Result<crate::mvt::Tile, prost::DecodeError> {
    use prost::Message;

    let proto = crate::mvt::proto::Tile::decode(bytes)?;
    let layers = proto
        .layers
        .iter()
        .filter_map(|layer| match crate::mvt::Layer::try_from(layer) {
            Ok(layer) => Some(layer),
            Err(error) => {
                log::warn!("dropping malformed tile layer {:?}: {error}", layer.name);
                None
            }
        })
        .collect();
    Ok(crate::mvt::Tile { layers })
}

/// A label candidate from one feature of a symbol layer. Shaping, placement
/// and collision happen per frame on the main thread; this is everything
/// that can be worked out once per tile.
#[derive(Debug, Clone)]
pub struct LabelSource {
    /// `text-field` with `text-transform` applied.
    pub text: SharedString,
    pub anchor: LabelAnchor,
    /// `symbol-sort-key`; lower keys are placed first.
    pub sort_key: f32,
    /// Data-driven `text-size`, resolved at the tile's zoom. `None` means
    /// the layer's value, resolved per frame.
    pub size: Option<f32>,
    /// Data-driven `text-color` times `text-opacity`, as `size`.
    pub color: Option<Rgba>,
}

#[derive(Debug, Clone)]
pub enum LabelAnchor {
    /// A point in mesh pixels.
    Point([f32; 2]),
    /// Distance `at` along `path` (mesh pixels). Anchors on the same line
    /// share its path.
    Line { path: Arc<[[f32; 2]]>, at: f32 },
}

/// Which camera zooms a mesh is built to be drawn at, and so how far its
/// geometry can be simplified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshDetail {
    /// A tile drawn for camera zooms in `[z, z + 1)`.
    Regular,
    /// A tile from the source's deepest level, drawn `levels` zooms past
    /// its own (the finest level also covers everything deeper).
    Overzoomed { levels: u8 },
}

impl MeshDetail {
    pub const FINEST_OVERZOOM: u8 = 2;

    pub fn for_camera(tile_z: u8, max_z: u8, camera_zoom: f64) -> Self {
        if tile_z < max_z {
            return Self::Regular;
        }
        let past = (camera_zoom.floor() - tile_z as f64).clamp(0.0, Self::FINEST_OVERZOOM as f64);
        Self::Overzoomed { levels: past as u8 }
    }
}

/// Build the mesh for one decoded tile at the given detail.
pub fn build_tile_mesh(tile: &crate::mvt::Tile, id: TileId, style: &Style, detail: MeshDetail) -> TileMesh {
    let overzoomed = matches!(detail, MeshDetail::Overzoomed { .. });
    // Data-driven values are sampled at the tile's own zoom. Widths also
    // get the next zoom so the painter can interpolate between the two.
    // TODO: overzoomed tiles keep sampling at their own zoom, so
    // data-driven colours freeze past the source's max zoom.
    let ctx = EvalContext { zoom: id.z as f64, ..Default::default() };
    let next_ctx = EvalContext { zoom: id.z as f64 + 1.0, ..Default::default() };
    let mut tessellators = Tessellators::default();

    let layers = style
        .layers
        .iter()
        .map(|layer| {
            if !builds_at(layer, id.z, overzoomed) {
                return None;
            }
            let source_layer = tile.layers.iter().find(|l| Some(l.name.as_str()) == layer.source_layer.as_deref())?;
            let scale = MESH_EXTENT / source_layer.extent as f32;

            // Half a pixel at the deepest zoom this mesh is drawn at, in MVT
            // units. A mesh for zooms `[z + n, z + n + 1)` is drawn at up
            // to `2^(n+1)` times the tile's own size. Past the finest level
            // the tolerance is below MVT's integer precision anyway.
            let half_px = source_layer.extent as f64 / MESH_EXTENT as f64 * 0.5;
            let levels = match detail {
                MeshDetail::Regular => 0,
                MeshDetail::Overzoomed { levels } => levels,
            };
            let epsilon = half_px / 2.0 / (1u32 << levels) as f64;

            let features = source_layer.features.iter().filter(|feature| {
                layer.filter.as_ref().is_none_or(|filter| filter.matches(&ctx, feature).unwrap_or(false))
            });
            let geometry = Simplified { scale, epsilon, min_extent: 2.0 * half_px };

            match &layer.kind {
                LayerKind::Fill(fill) => {
                    let paint = &fill.paint;
                    let layer_color = is_feature_constant(&paint.fill_color) && is_feature_constant(&paint.fill_opacity);
                    let mut batches: Vec<FillBatch> = Vec::new();
                    for feature in features {
                        let Some(path) = geometry.polygon_path(&feature.geometry) else { continue };
                        let color = if layer_color {
                            BatchColor::Layer
                        } else {
                            BatchColor::Feature(resolve_color(&paint.fill_color, &paint.fill_opacity, &ctx, feature))
                        };
                        let triangles = tessellators.fill(&path);
                        if triangles.is_empty() {
                            continue;
                        }
                        match batches.iter_mut().find(|b| b.color == color) {
                            Some(batch) => batch.triangles.extend(triangles),
                            None => batches.push(FillBatch { color, triangles }),
                        }
                    }
                    (!batches.is_empty()).then_some(LayerMesh::Fill(batches))
                }
                LayerKind::Line(line) => {
                    let paint = &line.paint;
                    let layer_color = is_feature_constant(&paint.line_color) && is_feature_constant(&paint.line_opacity);
                    let layer_width = is_feature_constant(&paint.line_width);
                    let options = stroke_options(line.layout.line_join, line.layout.line_cap);
                    let mut batches: Vec<LineBatch> = Vec::new();
                    for feature in features {
                        let Some(path) = geometry.line_path(&feature.geometry) else { continue };
                        let color = if layer_color {
                            BatchColor::Layer
                        } else {
                            BatchColor::Feature(resolve_color(&paint.line_color, &paint.line_opacity, &ctx, feature))
                        };
                        let vertices = tessellators.stroke(&path, &options);
                        if vertices.is_empty() {
                            continue;
                        }
                        let batch = match batches.iter().position(|b| b.color == color) {
                            Some(i) => &mut batches[i],
                            None => {
                                let width =
                                    if layer_width { LineWidth::Layer } else { LineWidth::PerVertex(Vec::new()) };
                                batches.push(LineBatch { color, width, vertices: Vec::new() });
                                batches.last_mut().expect("just pushed")
                            }
                        };
                        if let LineWidth::PerVertex(widths) = &mut batch.width {
                            let width = |ctx| resolve_width(&paint.line_width, ctx, feature);
                            let pair = [width(&ctx), width(&next_ctx)];
                            widths.extend(std::iter::repeat_n(pair, vertices.len()));
                        }
                        batch.vertices.extend(vertices);
                    }
                    (!batches.is_empty()).then_some(LayerMesh::Line(batches))
                }
                LayerKind::Symbol(symbol) => {
                    let labels: Vec<LabelSource> = features
                        .flat_map(|feature| label_sources(symbol, feature, &ctx, scale))
                        .collect();
                    (!labels.is_empty()).then_some(LayerMesh::Symbol(labels))
                }
                // Backgrounds are a single quad.
                LayerKind::Background(_) => None,
            }
        })
        .collect();

    TileMesh { layers }
}

/// Whether a style layer can be visible while this tile is on screen. A
/// regular tile is shown for camera zooms in `[z, z + 1)`; the deepest
/// level of a source keeps being shown at every zoom past it.
fn builds_at(layer: &Layer, z: u8, overzoomed: bool) -> bool {
    let z = z as f64;
    // TODO: `fill-pattern` needs the sprite atlas. Until then pattern
    // layers are skipped; drawing them would paint their (absent) colour,
    // i.e. black, over the fills underneath.
    let drawable = match &layer.kind {
        LayerKind::Fill(fill) => fill.paint.fill_pattern.is_none(),
        LayerKind::Line(_) => true,
        // TODO: icons (`icon-image`) need the sprite atlas; text only for now.
        LayerKind::Symbol(symbol) => symbol.layout.text_field.is_some(),
        LayerKind::Background(_) => false,
    };
    drawable
        && layer.visibility == Visibility::Visible
        && layer.maxzoom.is_none_or(|max| max > z)
        && (overzoomed || layer.minzoom.is_none_or(|min| min < z + 1.0))
}

// ==============================================================================
// Labels
// ==============================================================================

/// MapLibre's default `text-size` and `symbol-spacing`, in pixels.
const DEFAULT_TEXT_SIZE: f32 = 16.0;
const DEFAULT_SYMBOL_SPACING: f32 = 250.0;
/// Average glyph advance as a fraction of the font size. Shaping happens per
/// frame on the main thread, so line anchors are chosen against this
/// estimate of the label's length; the real fit is checked when placing.
const ESTIMATED_ADVANCE_EM: f32 = 0.6;

/// The label candidates one feature contributes to a symbol layer, with
/// anchors in mesh pixels. Anchors outside the tile itself (in the MVT
/// buffer around it) are dropped: the neighbouring tile owns them, and
/// keeping both would label the same place twice.
fn label_sources(symbol: &SymbolLayer, feature: &Feature, ctx: &EvalContext, scale: f32) -> Vec<LabelSource> {
    let layout = &symbol.layout;
    let paint = &symbol.paint;
    let Some(text) = layout.text_field.as_ref().and_then(|p| p.resolve(ctx, feature).ok()) else {
        return Vec::new();
    };
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let text: SharedString = match layout.text_transform.unwrap_or_default() {
        TextTransform::None => text.to_string().into(),
        TextTransform::Uppercase => text.to_uppercase().into(),
        TextTransform::Lowercase => text.to_lowercase().into(),
    };
    let resolved_size = layout.text_size.as_ref().and_then(|p| p.resolve(ctx, feature).ok()).unwrap_or(DEFAULT_TEXT_SIZE);
    let size = (!is_feature_constant(&layout.text_size)).then_some(resolved_size);
    let color = (!(is_feature_constant(&paint.text_color) && is_feature_constant(&paint.text_opacity)))
        .then(|| resolve_color(&paint.text_color, &paint.text_opacity, ctx, feature));
    let sort_key = layout.symbol_sort_key.as_ref().and_then(|p| p.resolve(ctx, feature).ok()).unwrap_or(0.0);
    let inside = |p: [f32; 2]| (0.0..MESH_EXTENT).contains(&p[0]) && (0.0..MESH_EXTENT).contains(&p[1]);
    let to_mesh = |c: &geo::Coord<f64>| [c.x as f32 * scale, c.y as f32 * scale];

    let placement = layout.symbol_placement.as_ref().and_then(|p| p.resolve(ctx, feature).ok()).unwrap_or_default();
    let anchors: Vec<LabelAnchor> = match placement {
        SymbolPlacement::Point => {
            let points: Vec<[f32; 2]> = match &feature.geometry {
                Geometry::Point(p) => vec![to_mesh(&p.0)],
                Geometry::MultiPoint(mp) => mp.0.iter().map(|p| to_mesh(&p.0)).collect(),
                Geometry::LineString(ls) => {
                    let path: Vec<[f32; 2]> = ls.0.iter().map(to_mesh).collect();
                    let middle = label::path_length(&path) / 2.0;
                    (path.len() >= 2).then(|| label::point_at(&path, middle).0).into_iter().collect()
                }
                Geometry::Polygon(p) => p.interior_point().map(|p| to_mesh(&p.0)).into_iter().collect(),
                Geometry::MultiPolygon(mp) => mp.interior_point().map(|p| to_mesh(&p.0)).into_iter().collect(),
                Geometry::MultiLineString(_) | Geometry::Unknown => Vec::new(),
            };
            points.into_iter().filter(|&p| inside(p)).map(LabelAnchor::Point).collect()
        }
        placement @ (SymbolPlacement::Line | SymbolPlacement::LineCenter) => {
            let spacing = layout
                .symbol_spacing
                .as_ref()
                .and_then(|p| p.resolve(ctx, feature).ok())
                .unwrap_or(DEFAULT_SYMBOL_SPACING);
            let estimated_length = text.chars().count() as f32 * resolved_size * ESTIMATED_ADVANCE_EM;
            line_paths(&feature.geometry)
                .into_iter()
                .flat_map(|line| {
                    let path: Arc<[[f32; 2]]> = line.0.iter().map(to_mesh).collect();
                    let at = match placement {
                        SymbolPlacement::LineCenter => label::line_center_anchor(&path, estimated_length).into_iter().collect(),
                        _ => label::line_anchors(&path, spacing, estimated_length),
                    };
                    at.into_iter()
                        .filter(|&at| inside(label::point_at(&path, at).0))
                        .map(|at| LabelAnchor::Line { path: path.clone(), at })
                        .collect::<Vec<_>>()
                })
                .collect()
        }
    };
    anchors
        .into_iter()
        .map(|anchor| LabelSource { text: text.clone(), anchor, sort_key, size, color })
        .collect()
}

/// Lines a `symbol-placement: line` label can follow: linestrings, and
/// polygon outlines as MapLibre does.
fn line_paths(geometry: &Geometry) -> Vec<&geo::LineString<f64>> {
    match geometry {
        Geometry::LineString(ls) => vec![ls],
        Geometry::MultiLineString(mls) => mls.0.iter().collect(),
        Geometry::Polygon(p) => std::iter::once(p.exterior()).chain(p.interiors()).collect(),
        Geometry::MultiPolygon(mp) => mp.0.iter().flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors())).collect(),
        Geometry::Point(_) | Geometry::MultiPoint(_) | Geometry::Unknown => Vec::new(),
    }
}

pub(crate) fn is_feature_constant<T>(property: &Option<Property<T>>) -> bool {
    match property {
        None | Some(Property::Constant(_)) => true,
        Some(Property::Expression(expr)) => expr.is_feature_constant(),
    }
}

/// Colour times opacity, defaulting to the spec's opaque black. A failed
/// evaluation falls back to the default rather than dropping the feature,
/// matching MapLibre's behaviour for runtime expression errors.
pub(crate) fn resolve_color(
    color: &Option<Property<Color>>,
    opacity: &Option<Property<f32>>,
    ctx: &EvalContext,
    feature: &Feature,
) -> Rgba {
    let c = color.as_ref().and_then(|p| p.resolve(ctx, feature).ok()).unwrap_or(Color { r: 0, g: 0, b: 0, a: 255 });
    let opacity = opacity.as_ref().and_then(|p| p.resolve(ctx, feature).ok()).unwrap_or(1.0);
    Rgba {
        r: c.r as f32 / 255.0,
        g: c.g as f32 / 255.0,
        b: c.b as f32 / 255.0,
        a: c.a as f32 / 255.0 * opacity,
    }
}

/// `line-width`, defaulting to the spec's 1 px.
pub(crate) fn resolve_width(width: &Option<Property<f32>>, ctx: &EvalContext, feature: &Feature) -> f32 {
    width.as_ref().and_then(|p| p.resolve(ctx, feature).ok()).unwrap_or(1.0)
}

// ==============================================================================
// Geometry preparation
// ==============================================================================
//
// MVT geometry is simplified (Ramer–Douglas–Peucker) to sub-pixel error in
// MVT units, scaled to mesh pixels, and turned into a lyon path. Every
// vertex removed here is one GPUI does not have to copy on every frame.

struct Simplified {
    scale: f32,
    epsilon: f64,
    /// Polygons whose bounding box is smaller than this in both directions
    /// cover at most a pixel and are dropped.
    min_extent: f64,
}

impl Simplified {
    fn polygon_path(&self, geometry: &Geometry) -> Option<LPath> {
        let polygons: &[geo::Polygon<f64>] = match geometry {
            Geometry::Polygon(p) => std::slice::from_ref(p),
            Geometry::MultiPolygon(mp) => &mp.0,
            _ => return None,
        };
        let mut builder = LPath::builder();
        let mut any = false;
        for polygon in polygons {
            let Some(bbox) = polygon.bounding_rect() else { continue };
            if bbox.width() < self.min_extent && bbox.height() < self.min_extent {
                continue;
            }
            for ring in std::iter::once(polygon.exterior()).chain(polygon.interiors()) {
                any |= self.add(&mut builder, &ring.simplify(self.epsilon), true);
            }
        }
        any.then(|| builder.build())
    }

    /// Line layers stroke linestrings and, per the style spec, polygon
    /// outlines too.
    fn line_path(&self, geometry: &Geometry) -> Option<LPath> {
        let mut builder = LPath::builder();
        let mut any = false;
        let mut add_polygon = |builder: &mut _, polygon: &geo::Polygon<f64>| {
            for ring in std::iter::once(polygon.exterior()).chain(polygon.interiors()) {
                any |= self.add(builder, &ring.simplify(self.epsilon), true);
            }
        };
        match geometry {
            Geometry::LineString(ls) => any |= self.add(&mut builder, &ls.simplify(self.epsilon), false),
            Geometry::MultiLineString(mls) => {
                for ls in &mls.0 {
                    any |= self.add(&mut builder, &ls.simplify(self.epsilon), false);
                }
            }
            Geometry::Polygon(p) => add_polygon(&mut builder, p),
            Geometry::MultiPolygon(mp) => mp.0.iter().for_each(|p| add_polygon(&mut builder, p)),
            Geometry::Point(_) | Geometry::MultiPoint(_) | Geometry::Unknown => {}
        }
        any.then(|| builder.build())
    }

    fn add(&self, builder: &mut lyon_tessellation::path::path::Builder, line: &geo::LineString<f64>, closed: bool) -> bool {
        // A closed ring repeats its first point; lyon closes it itself.
        let coords = if closed && line.is_closed() { &line.0[..line.0.len().saturating_sub(1)] } else { &line.0[..] };
        if coords.len() < if closed { 3 } else { 2 } {
            return false;
        }
        let p = |c: &geo::Coord<f64>| lpoint(c.x as f32 * self.scale, c.y as f32 * self.scale);
        builder.begin(p(&coords[0]));
        for c in &coords[1..] {
            builder.line_to(p(c));
        }
        builder.end(closed);
        true
    }
}

// ==============================================================================
// Tessellation
// ==============================================================================

/// Lyon's tessellators keep scratch allocations between calls; one set is
/// reused for a whole tile.
#[derive(Default)]
struct Tessellators {
    fill: FillTessellator,
    stroke: StrokeTessellator,
}

impl Tessellators {
    fn fill(&mut self, path: &LPath) -> Vec<[f32; 2]> {
        let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
        // MVT rings are oriented (outer vs hole), but even-odd also copes
        // with the occasional mis-wound ring in real data.
        let options = FillOptions::default().with_fill_rule(FillRule::EvenOdd);
        let result = self.fill.tessellate_path(
            path,
            &options,
            &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| v.position().to_array()),
        );
        if let Err(error) = result {
            log::debug!("dropping polygon lyon could not fill: {error:?}");
            return Vec::new();
        }
        buffers.indices.iter().map(|&i| buffers.vertices[i as usize]).collect()
    }

    fn stroke(&mut self, path: &LPath, options: &StrokeOptions) -> Vec<LineVertex> {
        let mut buffers: VertexBuffers<LineVertex, u32> = VertexBuffers::new();
        let result = self.stroke.tessellate_path(
            path,
            options,
            &mut BuffersBuilder::new(&mut buffers, |v: StrokeVertex| LineVertex {
                center: v.position_on_path().to_array(),
                normal: v.normal().to_array(),
            }),
        );
        if let Err(error) = result {
            log::debug!("dropping line lyon could not stroke: {error:?}");
            return Vec::new();
        }
        buffers.indices.iter().map(|&i| buffers.vertices[i as usize]).collect()
    }
}

/// Lyon only needs a width to decide how finely to subdivide round joins
/// and caps; the real width is applied per frame along the normals. Round
/// caps on the many short pieces roads are split into are a large share
/// of all triangles, so arcs are cut coarsely: at a nominal 8 px with
/// 0.5 px tolerance a semicircle is about four segments. That reads as
/// round up to ~20 px; measured on a z13 Vaduz tile it removes a third of
/// all line triangles compared with 0.1 px.
// TODO: wide lines (> ~20 px) show facets on round joins and caps.
fn stroke_options(join: Option<LineJoin>, cap: Option<LineCap>) -> StrokeOptions {
    let join = match join.unwrap_or_default() {
        LineJoin::Miter => lyon_tessellation::LineJoin::Miter,
        LineJoin::Bevel => lyon_tessellation::LineJoin::Bevel,
        LineJoin::Round => lyon_tessellation::LineJoin::Round,
    };
    let cap = match cap.unwrap_or_default() {
        LineCap::Butt => lyon_tessellation::LineCap::Butt,
        LineCap::Round => lyon_tessellation::LineCap::Round,
        LineCap::Square => lyon_tessellation::LineCap::Square,
    };
    // TODO: honour `line-miter-limit`; lyon's default (4) is close to the
    // spec's default of 2 but not equal.
    StrokeOptions::default().with_line_width(8.0).with_tolerance(0.5).with_line_join(join).with_line_cap(cap)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use geo::{LineString, Polygon, coord};
    use crate::mvt::{Feature, Geometry, Value};

    use super::*;

    fn style(layers: &str) -> Style {
        Style::from_json_str(&format!(
            r#"{{"version": 8, "sources": {{"osm": {{"type": "vector"}}}}, "layers": [{layers}]}}"#
        ))
        .expect("test style parses")
    }

    fn feature(props: &[(&str, Value)], geometry: Geometry) -> Feature {
        Feature {
            id: None,
            properties: props.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<HashMap<_, _>>(),
            geometry,
        }
    }

    fn square(x0: f64, y0: f64, size: f64) -> Geometry {
        Geometry::Polygon(Polygon::new(
            LineString::from(vec![(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size), (x0, y0)]),
            vec![],
        ))
    }

    fn tile(layer: &str, features: Vec<Feature>) -> crate::mvt::Tile {
        crate::mvt::Tile {
            layers: vec![crate::mvt::Layer { name: layer.into(), version: 2, extent: 4096, features }],
        }
    }

    const Z14: TileId = TileId { z: 14, x: 8613, y: 5754 };

    #[test]
    fn constant_fill_becomes_one_layer_coloured_batch_in_mesh_pixels() {
        let style = style(
            r##"{"id": "bg", "type": "background", "paint": {"background-color": "#fff"}},
                {"id": "water", "type": "fill", "source": "osm", "source-layer": "water",
                 "paint": {"fill-color": "#a0c8f0"}}"##,
        );
        let tile = tile("water", vec![feature(&[], square(0.0, 0.0, 2048.0))]);
        let mesh = build_tile_mesh(&tile, Z14, &style, MeshDetail::Regular);

        assert_eq!(mesh.layers.len(), 2);
        assert!(mesh.layers[0].is_none(), "background is painted, not meshed");
        let Some(LayerMesh::Fill(batches)) = &mesh.layers[1] else { panic!("{:?}", mesh.layers[1]) };
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].color, BatchColor::Layer);
        assert_eq!(batches[0].triangles.len(), 6);
        assert!(batches[0].triangles.iter().all(|p| (0.0..=256.0).contains(&p[0]) && (0.0..=256.0).contains(&p[1])));
    }

    #[test]
    fn filter_drops_non_matching_features() {
        let style = style(
            r##"{"id": "lakes", "type": "fill", "source": "osm", "source-layer": "water",
                 "filter": ["==", "class", "lake"], "paint": {"fill-color": "#00f"}}"##,
        );
        let tile = tile("water", vec![feature(&[("class", Value::String("river".into()))], square(0.0, 0.0, 512.0))]);
        let mesh = build_tile_mesh(&tile, Z14, &style, MeshDetail::Regular);
        assert!(mesh.layers[0].is_none());
    }

    #[test]
    fn data_driven_fill_colour_is_baked_per_feature() {
        let style = style(
            r##"{"id": "water", "type": "fill", "source": "osm", "source-layer": "water",
                 "paint": {"fill-color": ["match", ["get", "class"], "lake", "#0000ff", "#ff0000"],
                           "fill-opacity": 0.5}}"##,
        );
        let tile = tile(
            "water",
            vec![
                feature(&[("class", Value::String("lake".into()))], square(0.0, 0.0, 512.0)),
                feature(&[("class", Value::String("sea".into()))], square(1024.0, 0.0, 512.0)),
                feature(&[("class", Value::String("lake".into()))], square(2048.0, 0.0, 512.0)),
            ],
        );
        let mesh = build_tile_mesh(&tile, Z14, &style, MeshDetail::Regular);
        let Some(LayerMesh::Fill(batches)) = &mesh.layers[0] else { panic!() };
        let colours: Vec<_> = batches.iter().map(|b| (b.color, b.triangles.len())).collect();
        assert_eq!(
            colours,
            vec![
                (BatchColor::Feature(Rgba { r: 0.0, g: 0.0, b: 1.0, a: 0.5 }), 12),
                (BatchColor::Feature(Rgba { r: 1.0, g: 0.0, b: 0.0, a: 0.5 }), 6),
            ]
        );
    }

    #[test]
    fn constant_line_width_is_resolved_per_frame() {
        let style = style(
            r##"{"id": "roads", "type": "line", "source": "osm", "source-layer": "transportation",
                 "paint": {"line-color": "#888",
                           "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1, 16, 8]}}"##,
        );
        let road = Geometry::LineString(LineString::from(vec![(100.0, 2048.0), (4000.0, 2048.0)]));
        let mesh = build_tile_mesh(&tile("transportation", vec![feature(&[], road)]), Z14, &style, MeshDetail::Regular);
        let Some(LayerMesh::Line(batches)) = &mesh.layers[0] else { panic!() };
        assert_eq!(batches[0].width, LineWidth::Layer);
        assert_eq!(batches[0].color, BatchColor::Layer);
    }

    #[test]
    fn line_vertices_carry_centreline_and_unit_normal() {
        let style = style(
            r##"{"id": "roads", "type": "line", "source": "osm", "source-layer": "transportation",
                 "paint": {"line-color": "#888", "line-width": ["get", "w"]}}"##,
        );
        let road = Geometry::LineString(LineString::from(vec![(100.0, 2048.0), (4000.0, 2048.0)]));
        let tile = tile("transportation", vec![feature(&[("w", Value::Double(4.0))], road)]);
        let mesh = build_tile_mesh(&tile, Z14, &style, MeshDetail::Regular);
        let Some(LayerMesh::Line(batches)) = &mesh.layers[0] else { panic!() };
        let batch = &batches[0];
        assert!(!batch.vertices.is_empty() && batch.vertices.len() % 3 == 0);
        for v in &batch.vertices {
            assert!((v.center[1] - 256.0).abs() < 1e-3, "centre on the line: {v:?}");
            assert!(v.normal[0].abs() < 1e-3 && (v.normal[1].abs() - 1.0).abs() < 1e-3, "{v:?}");
        }
        let LineWidth::PerVertex(widths) = &batch.width else { panic!("{:?}", batch.width) };
        assert_eq!(widths.len(), batch.vertices.len());
        assert!(widths.iter().all(|w| *w == [4.0, 4.0]));
    }

    #[test]
    fn line_layers_stroke_polygon_outlines() {
        let style = style(
            r##"{"id": "outline", "type": "line", "source": "osm", "source-layer": "building",
                 "paint": {"line-color": "#000"}}"##,
        );
        let mesh = build_tile_mesh(&tile("building", vec![feature(&[], square(0.0, 0.0, 512.0))]), Z14, &style, MeshDetail::Regular);
        assert!(matches!(&mesh.layers[0], Some(LayerMesh::Line(b)) if !b[0].vertices.is_empty()));
    }

    #[test]
    fn pattern_fills_are_not_drawn_as_solid_colour() {
        let style = style(
            r##"{"id": "garden pattern", "type": "fill", "source": "osm", "source-layer": "landcover",
                 "paint": {"fill-pattern": "wood-pattern", "fill-opacity": 0.6}}"##,
        );
        let mesh = build_tile_mesh(&tile("landcover", vec![feature(&[], square(0.0, 0.0, 512.0))]), Z14, &style, MeshDetail::Regular);
        assert!(mesh.layers[0].is_none());
    }

    #[test]
    fn malformed_layer_is_dropped_but_tile_survives() {
        use prost::Message;
        use crate::mvt::proto::tile;

        let square = tile::Feature {
            r#type: Some(tile::GeomType::Polygon as i32),
            geometry: vec![9, 0, 0, 26, 20, 0, 0, 20, 19, 0, 15],
            ..Default::default()
        };
        let good = tile::Layer { name: "water".into(), version: 2, features: vec![square], ..Default::default() };
        // A value with no populated variant is invalid per the spec.
        let bad = tile::Layer { name: "poi".into(), version: 2, values: vec![tile::Value::default()], ..Default::default() };
        let bytes = crate::mvt::proto::Tile { layers: vec![bad, good] }.encode_to_vec();

        let decoded = decode_tile(&bytes).expect("protobuf is well-formed");
        let names: Vec<_> = decoded.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["water"]);
        assert_eq!(decoded.layers[0].features.len(), 1);
    }

    #[test]
    fn deeper_overzoom_keeps_more_detail() {
        let style = style(
            r##"{"id": "roads", "type": "line", "source": "osm", "source-layer": "transportation",
                 "paint": {"line-color": "#888"}}"##,
        );
        // Wobbles of 1.5 MVT units: within the tolerance at the tile's own
        // zoom, but most of a pixel two zooms deeper.
        let wobble: Vec<_> = (0..100).map(|i| coord! { x: 40.0 * i as f64, y: 2048.0 + 1.5 * (i % 2) as f64 }).collect();
        let count = |detail| {
            let road = Geometry::LineString(LineString::new(wobble.clone()));
            let mesh = build_tile_mesh(&tile("transportation", vec![feature(&[], road)]), Z14, &style, detail);
            let Some(LayerMesh::Line(b)) = &mesh.layers[0] else { panic!() };
            b[0].vertices.len()
        };
        let shallow = count(MeshDetail::Overzoomed { levels: 0 });
        let deep = count(MeshDetail::Overzoomed { levels: MeshDetail::FINEST_OVERZOOM });
        assert!(deep > shallow * 10, "{shallow} vs {deep}");
        assert_eq!(shallow, count(MeshDetail::Regular));
    }

    #[test]
    fn detail_follows_the_camera_past_the_deepest_level() {
        assert_eq!(MeshDetail::for_camera(13, 14, 13.7), MeshDetail::Regular);
        assert_eq!(MeshDetail::for_camera(14, 14, 14.3), MeshDetail::Overzoomed { levels: 0 });
        assert_eq!(MeshDetail::for_camera(14, 14, 15.9), MeshDetail::Overzoomed { levels: 1 });
        assert_eq!(MeshDetail::for_camera(14, 14, 19.0), MeshDetail::Overzoomed { levels: MeshDetail::FINEST_OVERZOOM });
    }

    fn labels(mesh: &TileMesh, layer: usize) -> &[LabelSource] {
        match &mesh.layers[layer] {
            Some(LayerMesh::Symbol(labels)) => labels,
            other => panic!("expected labels, got {other:?}"),
        }
    }

    fn point(x: f64, y: f64) -> Geometry {
        Geometry::Point(geo::Point::new(x, y))
    }

    #[test]
    fn point_labels_resolve_text_and_transform() {
        let style = style(
            r##"{"id": "places", "type": "symbol", "source": "osm", "source-layer": "place",
                 "layout": {"text-field": "{name}", "text-transform": "uppercase", "text-size": 14}}"##,
        );
        let tile = tile("place", vec![feature(&[("name", Value::String("Vaduz".into()))], point(1024.0, 2048.0))]);
        let mesh = build_tile_mesh(&tile, Z14, &style, MeshDetail::Regular);
        let [label] = labels(&mesh, 0) else { panic!() };
        assert_eq!(label.text.as_ref(), "VADUZ");
        assert!(matches!(label.anchor, LabelAnchor::Point(p) if p == [128.0, 256.0]), "{label:?}");
        assert_eq!((label.size, label.color), (None, None));
    }

    #[test]
    fn labels_in_the_neighbouring_tiles_buffer_are_left_to_that_tile() {
        let style = style(
            r##"{"id": "places", "type": "symbol", "source": "osm", "source-layer": "place",
                 "layout": {"text-field": ["get", "name"]}}"##,
        );
        let name = || [("name", Value::String("Schaan".into()))];
        let tile = tile(
            "place",
            vec![feature(&name(), point(-20.0, 100.0)), feature(&name(), point(4100.0, 100.0)), feature(&name(), point(5.0, 5.0))],
        );
        let mesh = build_tile_mesh(&tile, Z14, &style, MeshDetail::Regular);
        assert_eq!(labels(&mesh, 0).len(), 1);
    }

    #[test]
    fn empty_text_is_not_a_label() {
        let style = style(
            r##"{"id": "places", "type": "symbol", "source": "osm", "source-layer": "place",
                 "layout": {"text-field": ["get", "name"]}}"##,
        );
        let mesh = build_tile_mesh(&tile("place", vec![feature(&[], point(100.0, 100.0))]), Z14, &style, MeshDetail::Regular);
        assert!(mesh.layers[0].is_none());
    }

    #[test]
    fn line_labels_are_anchored_along_the_line_at_symbol_spacing() {
        let style = style(
            r##"{"id": "streets", "type": "symbol", "source": "osm", "source-layer": "transportation_name",
                 "layout": {"text-field": ["get", "name"], "symbol-placement": "line",
                            "symbol-spacing": 100, "text-size": 10}}"##,
        );
        // 4000 MVT units = 500 mesh px along y = 256 px.
        let road = Geometry::LineString(LineString::from(vec![(48.0, 2048.0), (4048.0, 2048.0)]));
        let tile = tile("transportation_name", vec![feature(&[("name", Value::String("Ring".into()))], road)]);
        let mesh = build_tile_mesh(&tile, Z14, &style, MeshDetail::Regular);
        let anchors: Vec<f32> = labels(&mesh, 0)
            .iter()
            .map(|l| match &l.anchor {
                LabelAnchor::Line { path, at } => {
                    assert_eq!(path.len(), 2);
                    *at
                }
                LabelAnchor::Point(_) => panic!("line placement gives line anchors"),
            })
            .collect();
        // Every 100 px from 50 along the 500 px line, which starts at
        // x = 6, so all five anchors are inside the tile.
        assert_eq!(anchors, vec![50.0, 150.0, 250.0, 350.0, 450.0]);
    }

    #[test]
    fn placement_zoom_functions_resolve_at_the_tiles_zoom() {
        let style = style(
            r##"{"id": "streets", "type": "symbol", "source": "osm", "source-layer": "transportation_name",
                 "layout": {"text-field": ["get", "name"], "text-size": 10,
                            "symbol-placement": {"base": 1, "stops": [[10, "point"], [11, "line"]]}}}"##,
        );
        let road = Geometry::LineString(LineString::from(vec![(48.0, 2048.0), (4048.0, 2048.0)]));
        let tile = || tile("transportation_name", vec![feature(&[("name", Value::String("Ring".into()))], road.clone())]);
        let at_z10 = build_tile_mesh(&tile(), TileId::new(10, 538, 359), &style, MeshDetail::Regular);
        assert!(matches!(labels(&at_z10, 0), [LabelSource { anchor: LabelAnchor::Point(_), .. }]));
        let at_z14 = build_tile_mesh(&tile(), Z14, &style, MeshDetail::Regular);
        assert!(labels(&at_z14, 0).iter().all(|l| matches!(l.anchor, LabelAnchor::Line { .. })));
    }

    #[test]
    fn data_driven_text_size_is_baked_per_feature() {
        let style = style(
            r##"{"id": "places", "type": "symbol", "source": "osm", "source-layer": "place",
                 "layout": {"text-field": ["get", "name"], "text-size": ["get", "rank"]}}"##,
        );
        let tile = tile(
            "place",
            vec![feature(&[("name", Value::String("Wien".into())), ("rank", Value::Double(20.0))], point(10.0, 10.0))],
        );
        let mesh = build_tile_mesh(&tile, Z14, &style, MeshDetail::Regular);
        assert_eq!(labels(&mesh, 0)[0].size, Some(20.0));
    }

    #[test]
    fn sub_pixel_polygons_are_dropped() {
        let style = style(
            r##"{"id": "b", "type": "fill", "source": "osm", "source-layer": "building",
                 "paint": {"fill-color": "#000"}}"##,
        );
        // 3 MVT units is under half a mesh pixel (4096 / 512 / 2 = 4 units).
        let mesh = build_tile_mesh(&tile("building", vec![feature(&[], square(10.0, 10.0, 3.0))]), Z14, &style, MeshDetail::Regular);
        assert!(mesh.layers[0].is_none());
    }

    #[test]
    fn simplification_removes_near_collinear_vertices() {
        let style = style(
            r##"{"id": "roads", "type": "line", "source": "osm", "source-layer": "transportation",
                 "paint": {"line-color": "#888"}}"##,
        );
        // 100 vertices wobbling by one MVT unit: far below half a pixel.
        let wobble: Vec<_> = (0..100).map(|i| coord! { x: 40.0 * i as f64, y: 2048.0 + (i % 2) as f64 }).collect();
        let wobbly = Geometry::LineString(LineString::new(wobble));
        let straight = Geometry::LineString(LineString::from(vec![(0.0, 2048.0), (3960.0, 2048.0)]));
        let count = |g| {
            let mesh = build_tile_mesh(&tile("transportation", vec![feature(&[], g)]), Z14, &style, MeshDetail::Regular);
            let Some(LayerMesh::Line(b)) = &mesh.layers[0] else { panic!() };
            b[0].vertices.len()
        };
        assert_eq!(count(wobbly), count(straight));
    }
}
