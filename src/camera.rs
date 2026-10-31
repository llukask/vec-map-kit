// ==============================================================================
// Camera and Web Mercator math
// ==============================================================================
//
// The world is the Web Mercator unit square: `(0, 0)` is the north-west
// corner, `(1, 1)` the south-east. All camera state lives in f64 world
// units so deep zoom keeps full precision; only per-tile offsets are cast
// to f32 at paint time.
//
// Zoom follows the MapLibre convention: at zoom `z` the world is
// `512 · 2^z` logical pixels wide, so a vector tile at integer zoom `z`
// is drawn at exactly 512 px and style expressions see the same `zoom`
// a MapLibre map would show.
//
// Nothing here knows about GPUI; screen positions are `[f64; 2]` offsets
// from the viewport's top-left corner.

use crate::tile::TileId;

/// Logical pixels per tile at integer zoom.
pub const TILE_SIZE: f64 = 512.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldPoint {
    pub x: f64,
    pub y: f64,
}

impl WorldPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn from_lon_lat(lon: f64, lat: f64) -> Self {
        let lat = lat.to_radians();
        Self {
            x: (lon + 180.0) / 360.0,
            y: (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / std::f64::consts::PI) / 2.0,
        }
    }

    pub fn to_lon_lat(self) -> (f64, f64) {
        let lon = self.x * 360.0 - 180.0;
        let lat = (std::f64::consts::PI * (1.0 - 2.0 * self.y)).sinh().atan().to_degrees();
        (lon, lat)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub center: WorldPoint,
    pub zoom: f64,
}

impl Camera {
    /// Zoom bounds every camera stays within. Beyond ~24 the f64 world
    /// coordinates still hold, but no tile source has data that deep.
    pub const MIN_ZOOM: f64 = 0.0;
    pub const MAX_ZOOM: f64 = 24.0;

    pub fn new(center: WorldPoint, zoom: f64) -> Self {
        Self { center, zoom: zoom.clamp(Self::MIN_ZOOM, Self::MAX_ZOOM) }
    }

    /// Logical pixels per world unit.
    pub fn world_size(&self) -> f64 {
        TILE_SIZE * self.zoom.exp2()
    }

    pub fn world_to_screen(&self, p: WorldPoint, viewport: [f64; 2]) -> [f64; 2] {
        let s = self.world_size();
        [
            (p.x - self.center.x) * s + viewport[0] / 2.0,
            (p.y - self.center.y) * s + viewport[1] / 2.0,
        ]
    }

    pub fn screen_to_world(&self, p: [f64; 2], viewport: [f64; 2]) -> WorldPoint {
        let s = self.world_size();
        WorldPoint::new(
            self.center.x + (p[0] - viewport[0] / 2.0) / s,
            self.center.y + (p[1] - viewport[1] / 2.0) / s,
        )
    }

    /// Move the map by a screen-space drag: content follows the pointer.
    pub fn pan(&mut self, delta: [f64; 2]) {
        let s = self.world_size();
        self.center.x -= delta[0] / s;
        self.center.y = (self.center.y - delta[1] / s).clamp(0.0, 1.0);
        // TODO: wrap the world horizontally instead of letting `x` run off.
    }

    /// Change zoom by `delta` while keeping the world point under `anchor`
    /// (a screen position) fixed, as scroll-to-zoom expects.
    pub fn zoom_at(&mut self, anchor: [f64; 2], viewport: [f64; 2], delta: f64) {
        let before = self.screen_to_world(anchor, viewport);
        self.zoom = (self.zoom + delta).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let after = self.screen_to_world(anchor, viewport);
        self.center.x += before.x - after.x;
        self.center.y += before.y - after.y;
    }

    /// The integer zoom whose tiles should be drawn, given the zoom range a
    /// source actually has data for. Above `max_z` the deepest tiles are
    /// overzoomed; below `min_z` nothing is drawn by the source.
    pub fn tile_zoom(&self, min_z: u8, max_z: u8) -> u8 {
        (self.zoom.floor().max(0.0) as u8).clamp(min_z, max_z)
    }

    /// Tiles at `tile_zoom` that intersect the viewport, nearest to the
    /// viewport centre first so loading fills in from the middle.
    pub fn tile_cover(&self, viewport: [f64; 2], min_z: u8, max_z: u8) -> Vec<TileId> {
        let z = self.tile_zoom(min_z, max_z);
        let n = 1u32 << z;
        let top_left = self.screen_to_world([0.0, 0.0], viewport);
        let bottom_right = self.screen_to_world(viewport, viewport);
        let to_index = |w: f64| ((w * n as f64).floor().max(0.0) as u32).min(n - 1);
        let (x0, x1) = (to_index(top_left.x.max(0.0)), to_index(bottom_right.x.min(1.0)));
        let (y0, y1) = (to_index(top_left.y.max(0.0)), to_index(bottom_right.y.min(1.0)));
        if top_left.x >= 1.0 || bottom_right.x <= 0.0 {
            return Vec::new();
        }

        let mut tiles: Vec<TileId> = (y0..=y1)
            .flat_map(|y| (x0..=x1).map(move |x| TileId::new(z, x, y)))
            .collect();
        let centre = [self.center.x * n as f64, self.center.y * n as f64];
        tiles.sort_by(|a, b| {
            let d = |t: &TileId| {
                let dx = t.x as f64 + 0.5 - centre[0];
                let dy = t.y as f64 + 0.5 - centre[1];
                dx * dx + dy * dy
            };
            d(a).total_cmp(&d(b))
        });
        tiles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn lon_lat_origin_is_world_centre_and_round_trips() {
        let p = WorldPoint::from_lon_lat(0.0, 0.0);
        assert!(close(p.x, 0.5) && close(p.y, 0.5));

        let vaduz = WorldPoint::from_lon_lat(9.5209, 47.1410);
        let (lon, lat) = vaduz.to_lon_lat();
        assert!(close(lon, 9.5209) && close(lat, 47.1410), "{lon} {lat}");
    }

    #[test]
    fn centre_projects_to_viewport_middle() {
        let cam = Camera::new(WorldPoint::new(0.3, 0.7), 5.5);
        let s = cam.world_to_screen(cam.center, [800.0, 600.0]);
        assert!(close(s[0], 400.0) && close(s[1], 300.0));
    }

    #[test]
    fn zoom_at_keeps_anchor_fixed() {
        let viewport = [1000.0, 700.0];
        let anchor = [120.0, 640.0];
        let mut cam = Camera::new(WorldPoint::new(0.53, 0.35), 12.3);
        let under = cam.screen_to_world(anchor, viewport);
        cam.zoom_at(anchor, viewport, 0.75);
        let after = cam.world_to_screen(under, viewport);
        assert!(close(cam.zoom, 13.05));
        assert!((after[0] - anchor[0]).abs() < 1e-6 && (after[1] - anchor[1]).abs() < 1e-6);
    }

    #[test]
    fn pan_moves_content_with_pointer() {
        let viewport = [800.0, 600.0];
        let mut cam = Camera::new(WorldPoint::new(0.5, 0.5), 10.0);
        let p = WorldPoint::new(0.5001, 0.4999);
        let before = cam.world_to_screen(p, viewport);
        cam.pan([30.0, -12.0]);
        let after = cam.world_to_screen(p, viewport);
        assert!((after[0] - before[0] - 30.0).abs() < 1e-6);
        assert!((after[1] - before[1] + 12.0).abs() < 1e-6);
    }

    #[test]
    fn whole_world_at_zoom_one_covers_four_tiles() {
        let cam = Camera::new(WorldPoint::new(0.5, 0.5), 1.0);
        let mut cover = cam.tile_cover([1024.0, 1024.0], 0, 14);
        cover.sort_by_key(|t| (t.y, t.x));
        assert_eq!(
            cover,
            vec![TileId::new(1, 0, 0), TileId::new(1, 1, 0), TileId::new(1, 0, 1), TileId::new(1, 1, 1)]
        );
    }

    #[test]
    fn deep_zoom_overzooms_the_deepest_source_level() {
        let cam = Camera::new(WorldPoint::from_lon_lat(9.52, 47.14), 16.4);
        assert_eq!(cam.tile_zoom(0, 14), 14);
        let cover = cam.tile_cover([1600.0, 1000.0], 0, 14);
        // 1600 px at 2^2.4 × 512 px per z14 tile is under one tile wide.
        assert!(cover.len() <= 4 && !cover.is_empty(), "{cover:?}");
        assert!(cover.iter().all(|t| t.z == 14));
    }

    #[test]
    fn cover_is_sorted_centre_first() {
        let cam = Camera::new(WorldPoint::new(0.5004, 0.5003), 12.0);
        let cover = cam.tile_cover([3000.0, 2000.0], 0, 14);
        let n = 4096.0;
        let centre = TileId::new(12, (0.5004 * n) as u32, (0.5003 * n) as u32);
        assert_eq!(cover[0], centre);
    }
}
