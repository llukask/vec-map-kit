// A tile address in the XYZ scheme (y grows southwards), plus the bits of
// quadtree arithmetic the renderer needs to stand an ancestor in for a
// tile that hasn't loaded yet.

use crate::camera::WorldPoint;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl TileId {
    pub fn new(z: u8, x: u32, y: u32) -> Self {
        Self { z, x, y }
    }

    pub fn parent(self) -> Option<Self> {
        (self.z > 0).then(|| Self::new(self.z - 1, self.x / 2, self.y / 2))
    }

    /// The four tiles one level down, in reading order.
    pub fn children(self) -> [Self; 4] {
        let (z, x, y) = (self.z + 1, self.x * 2, self.y * 2);
        [Self::new(z, x, y), Self::new(z, x + 1, y), Self::new(z, x, y + 1), Self::new(z, x + 1, y + 1)]
    }

    /// North-west corner in world units.
    pub fn origin(self) -> WorldPoint {
        let n = (1u64 << self.z) as f64;
        WorldPoint::new(self.x as f64 / n, self.y as f64 / n)
    }

    /// Edge length in world units.
    pub fn world_extent(self) -> f64 {
        1.0 / (1u64 << self.z) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_are_the_four_quadrants() {
        let parent = TileId::new(13, 4468, 2840);
        let children = parent.children();
        assert_eq!(children[3], TileId::new(14, 8937, 5681));
        assert!(children.iter().all(|c| c.parent() == Some(parent)));
    }

    #[test]
    fn parent_halves_coordinates() {
        assert_eq!(TileId::new(14, 8613, 5754).parent(), Some(TileId::new(13, 4306, 2877)));
        assert_eq!(TileId::new(0, 0, 0).parent(), None);
    }
}
