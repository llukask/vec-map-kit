// Where do a tile's triangles come from? Builds the mesh for the tile
// under a point and prints triangle counts per style layer, largest first.
//
//     cargo run --release --example mesh_stats -- SOURCE STYLE Z LON LAT
//
// SOURCE is a TileJSON URL and STYLE a style file or URL, as for the `map`
// example.

use futures::executor::block_on;
use vec_map_kit::mesh::{LayerMesh, MeshDetail, build_tile_mesh, decode_tile};
use vec_map_kit::source::fetch_text;
use vec_map_kit::{Style, TileId, TileJsonSource, WorldPoint};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [source, style, z, lon, lat] = args.as_slice() else {
        eprintln!("usage: mesh_stats SOURCE STYLE Z LON LAT");
        std::process::exit(2);
    };
    let is_url = |s: &str| s.starts_with("http://") || s.starts_with("https://");
    let source = TileJsonSource::fetch(source).expect("TileJSON loads");
    let style_json = if is_url(style) { fetch_text(style).expect("style downloads") } else { std::fs::read_to_string(style).expect("style is readable") };
    let style = Style::from_json_str(&style_json).expect("style parses");

    let z: u8 = z.parse().expect("Z is a zoom level");
    let point = WorldPoint::from_lon_lat(lon.parse().expect("LON is a number"), lat.parse().expect("LAT is a number"));
    let n = (1u64 << z) as f64;
    let id = TileId::new(z, (point.x * n) as u32, (point.y * n) as u32);
    let bytes = block_on(source.tile(id)).expect("tile loads").expect("source has this tile");

    let started = std::time::Instant::now();
    let tile = decode_tile(&bytes).expect("tile decodes");
    // The detail the map would use with the camera at the tile's own zoom.
    let mesh = build_tile_mesh(&tile, id, &style, MeshDetail::for_camera(z, source.zoom_range().1, z as f64));
    let build_time = started.elapsed();

    let mut rows: Vec<(usize, &str, &str)> = mesh
        .layers
        .iter()
        .zip(&style.layers)
        .filter_map(|(mesh, layer)| {
            let (count, kind) = match mesh.as_ref()? {
                LayerMesh::Fill(b) => (b.iter().map(|b| b.triangles.len() / 3).sum(), "fill"),
                LayerMesh::Line(b) => (b.iter().map(|b| b.vertices.len() / 3).sum(), "line"),
                LayerMesh::Symbol(labels) => (labels.len(), "labels"),
            };
            Some((count, kind, layer.id.as_str()))
        })
        .collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.0));
    let total: usize = rows.iter().map(|r| r.0).sum();
    let lines: usize = rows.iter().filter(|r| r.1 == "line").map(|r| r.0).sum();
    println!("{id:?}: {total} triangles ({lines} in lines) from {} layers, built in {build_time:?}", rows.len());
    for (count, kind, layer) in rows.iter().take(15) {
        println!("  {count:7}  {kind}  {layer}");
    }
}
