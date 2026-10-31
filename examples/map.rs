// A window with one `MapView`: drag to pan, scroll to zoom.
//
//     cargo run --release --example map -- SOURCE STYLE [ZOOM [LON LAT]]
//
// SOURCE is an http(s) TileJSON URL, e.g.
// `https://api.maptiler.com/tiles/v3-openmaptiles/tiles.json?key=KEY`.
// STYLE is a local style JSON file or an http(s) URL to one. The style's
// own `sources` are ignored: every layer draws from SOURCE, so the two
// must use the same tile schema.
//
// With `VEC_MAP_KIT_BENCH=zoomout` it instead zooms out steadily from 15.2
// to 11.5 while tiles load, and counts flicker (see `ZoomOutBench`).
//
// With `VEC_MAP_KIT_BENCH=1` the example visits a fixed set of zooms over
// the start position, waits at each until every visible tile has loaded,
// times 60 redraws, prints main-thread CPU per frame, and exits. Waiting
// for tiles keeps network latency out of the numbers. Run it inside
// `script/headless-sway.sh` to keep it off the desktop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::*;
use vec_map_kit::source::fetch_text;
use vec_map_kit::{Camera, MapView, Style, TileJsonSource, WorldPoint};

/// Zooms the bench visits, around the start position. Each is a
/// different tile level or scale within one, where triangle counts differ.
const BENCH_ZOOMS: [f64; 5] = [12.6, 13.4, 14.25, 15.1, 16.0];
/// Frames timed at each zoom once its tiles are loaded.
const BENCH_FRAMES: u32 = 60;
/// Give up waiting for tiles after this long.
const BENCH_LOAD_TIMEOUT: Duration = Duration::from_secs(60);

struct Bench {
    origin: WorldPoint,
    stop: usize,
    phase: BenchPhase,
    /// Per stop: main-thread CPU per frame in ms.
    results: Vec<f32>,
    /// Logical pixels the camera is nudged by; flips every frame.
    nudge: f64,
}

enum BenchPhase {
    Loading { since: Instant, settled_frames: u32 },
    Measuring { frames: u32, cpu_at_start: Duration, paint: Duration, triangles: usize },
}

impl Bench {
    /// Advance one frame: returns the camera to show, or exits the
    /// process when every stop is done.
    fn step(&mut self, stats: vec_map_kit::FrameStats) -> Camera {
        match &mut self.phase {
            BenchPhase::Loading { since, settled_frames } => {
                // A few quiet frames in a row, so tiles arriving between
                // frames have been drawn too.
                *settled_frames = if stats.loading == 0 && stats.tiles > 0 { *settled_frames + 1 } else { 0 };
                if *settled_frames >= 10 {
                    self.phase = BenchPhase::Measuring {
                        frames: 0,
                        cpu_at_start: thread_cpu(),
                        paint: Duration::ZERO,
                        triangles: 0,
                    };
                } else if since.elapsed() > BENCH_LOAD_TIMEOUT {
                    eprintln!("zoom {:5.2}: tiles did not finish loading", BENCH_ZOOMS[self.stop]);
                    std::process::exit(1);
                }
            }
            BenchPhase::Measuring { frames, cpu_at_start, paint, triangles } => {
                *frames += 1;
                *paint += stats.paint;
                *triangles += stats.triangles;
                if *frames == BENCH_FRAMES {
                    let n = BENCH_FRAMES as f32;
                    let cpu = (thread_cpu() - *cpu_at_start).as_secs_f32() * 1000.0 / n;
                    println!(
                        "zoom {:5.2}  tiles {:2}  tris {:7}  labels {:4}/{:4}  sprites {:6}  label layout {:5.2} ms  paint {:6.2} ms  main-thread {:6.2} ms/frame",
                        BENCH_ZOOMS[self.stop],
                        stats.tiles,
                        *triangles / BENCH_FRAMES as usize,
                        stats.labels,
                        stats.label_candidates,
                        stats.glyph_sprites,
                        stats.label_layout.as_secs_f32() * 1000.0,
                        paint.as_secs_f32() * 1000.0 / n,
                        cpu,
                    );
                    self.results.push(cpu);
                    self.stop += 1;
                    if self.stop == BENCH_ZOOMS.len() {
                        let worst = self.results.iter().copied().fold(0.0, f32::max);
                        let mean = self.results.iter().sum::<f32>() / self.results.len() as f32;
                        println!("main-thread ms/frame: mean {mean:.2}  worst {worst:.2}");
                        std::process::exit(0);
                    }
                    self.phase = BenchPhase::Loading { since: Instant::now(), settled_frames: 0 };
                }
            }
        }
        // Alternate a quarter-pixel nudge so every frame is a real redraw
        // without changing which tiles are visible.
        self.nudge = -self.nudge;
        let zoom = BENCH_ZOOMS[self.stop];
        Camera::new(WorldPoint::new(self.origin.x + self.nudge / (512.0 * zoom.exp2()), self.origin.y), zoom)
    }
}

/// `VEC_MAP_KIT_BENCH=zoomout`: once tiles at the start zoom have loaded,
/// zoom out at a steady rate (tiles for each new level load meanwhile, as
/// when a user zooms out) and count flicker: frames with holes in the map
/// and labels that vanish instead of fading.
struct ZoomOutBench {
    origin: WorldPoint,
    phase: ZoomOutPhase,
    frames: u32,
    frames_with_holes: u32,
    uncovered_tiles: usize,
    label_pops: usize,
}

enum ZoomOutPhase {
    Loading { settled_frames: u32 },
    Sweeping { started: Instant },
}

impl ZoomOutBench {
    const FROM: f64 = 15.2;
    const TO: f64 = 11.5;
    /// Zoom levels per second.
    const RATE: f64 = 1.0;

    fn step(&mut self, stats: vec_map_kit::FrameStats) -> Camera {
        let zoom = match &mut self.phase {
            ZoomOutPhase::Loading { settled_frames } => {
                *settled_frames = if stats.loading == 0 && stats.tiles > 0 { *settled_frames + 1 } else { 0 };
                if *settled_frames >= 30 {
                    self.phase = ZoomOutPhase::Sweeping { started: Instant::now() };
                }
                Self::FROM
            }
            ZoomOutPhase::Sweeping { started } => {
                self.frames += 1;
                self.frames_with_holes += u32::from(stats.uncovered > 0);
                self.uncovered_tiles += stats.uncovered;
                self.label_pops += stats.label_pops;
                let zoom = Self::FROM - Self::RATE * started.elapsed().as_secs_f64();
                if zoom <= Self::TO {
                    println!(
                        "zoom-out {:.1} -> {:.1}: {} frames, {} with holes ({} uncovered tile-frames), {} label pops",
                        Self::FROM,
                        Self::TO,
                        self.frames,
                        self.frames_with_holes,
                        self.uncovered_tiles,
                        self.label_pops
                    );
                    std::process::exit(0);
                }
                zoom
            }
        };
        Camera::new(self.origin, zoom)
    }
}

enum BenchMode {
    Frames(Bench),
    ZoomOut(ZoomOutBench),
}

struct MapWindow {
    map: Entity<MapView>,
    bench: Option<BenchMode>,
}

impl Render for MapWindow {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if let Some(bench) = &mut self.bench {
            let stats = self.map.read(cx).last_frame_stats();
            let camera = match bench {
                BenchMode::Frames(bench) => bench.step(stats),
                BenchMode::ZoomOut(bench) => bench.step(stats),
            };
            self.map.update(cx, |map, cx| map.set_camera(camera, cx));
            window.request_animation_frame();
        }

        let zoom = self.map.read(cx).camera().zoom;
        div()
            .size_full()
            .relative()
            .child(self.map.clone())
            .child(
                div()
                    .absolute()
                    .top_2()
                    .left_2()
                    .px_2()
                    .py_1()
                    .rounded_md()
                    .bg(rgba(0xffffffcc))
                    .text_sm()
                    .text_color(rgb(0x333333))
                    .child(format!("z {zoom:.2}")),
            )
    }
}

fn thread_cpu() -> Duration {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `ts` is a valid out-pointer for the duration of the call.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

fn is_url(location: &str) -> bool {
    location.starts_with("http://") || location.starts_with("https://")
}

/// Print an error with its whole `source()` chain and exit.
fn fail(doing: &str, error: impl std::error::Error) -> ! {
    eprintln!("error while {doing}: {error}");
    let mut cause = error.source();
    while let Some(c) = cause {
        eprintln!("  caused by: {c}");
        cause = c.source();
    }
    std::process::exit(1);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(source), Some(style)) = (args.next(), args.next()) else {
        eprintln!("usage: map SOURCE STYLE [ZOOM [LON LAT]]");
        eprintln!("  SOURCE  an http(s) TileJSON URL");
        eprintln!("  STYLE   a style JSON file or an http(s) URL");
        std::process::exit(2);
    };
    let source = Arc::new(TileJsonSource::fetch(&source).unwrap_or_else(|e| fail("loading the TileJSON", e)));
    let mut camera = source.initial_camera();
    let style_json = if is_url(&style) {
        fetch_text(&style).unwrap_or_else(|e| fail("downloading the style", e))
    } else {
        std::fs::read_to_string(&style).unwrap_or_else(|e| fail("reading the style", e))
    };
    let style = Arc::new(Style::from_json_str(&style_json).unwrap_or_else(|e| fail("parsing the style", e)));
    if let Some(zoom) = args.next() {
        camera = Camera::new(camera.center, zoom.parse().expect("ZOOM is a number"));
    }
    if let (Some(lon), Some(lat)) = (args.next(), args.next()) {
        let lon_lat = (lon.parse().expect("LON is a number"), lat.parse().expect("LAT is a number"));
        camera = Camera::new(WorldPoint::from_lon_lat(lon_lat.0, lon_lat.1), camera.zoom);
    }
    let bench = std::env::var("VEC_MAP_KIT_BENCH").ok();

    gpui_kit::application().run(move |cx| {
        gpui_kit::init(cx);
        let options = WindowOptions { app_id: Some("vec-map-kit".into()), ..Default::default() };
        gpui_kit::open_window(options, cx, |window, cx| {
            window.set_window_title("vec-map-kit");
            let map = cx.new(|_| MapView::new(source, style, camera));
            cx.new(|cx| {
                cx.observe(&map, |_, _, cx| cx.notify()).detach();
                MapWindow {
                    map,
                    bench: bench.map(|mode| match mode.as_str() {
                        "zoomout" => BenchMode::ZoomOut(ZoomOutBench {
                            origin: camera.center,
                            phase: ZoomOutPhase::Loading { settled_frames: 0 },
                            frames: 0,
                            frames_with_holes: 0,
                            uncovered_tiles: 0,
                            label_pops: 0,
                        }),
                        _ => BenchMode::Frames(Bench {
                            origin: camera.center,
                            stop: 0,
                            phase: BenchPhase::Loading { since: Instant::now(), settled_frames: 0 },
                            results: Vec::new(),
                            nudge: 0.25,
                        }),
                    }),
                }
            })
        })
        .expect("window opens");
    });
}
