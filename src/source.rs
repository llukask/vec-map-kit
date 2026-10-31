// Where tile bytes come from: a TileJSON document names URL templates for
// a tile pyramid, and `TileJsonSource` downloads tiles from them on a
// small pool of fetch threads. `tile` returns a future so a GPUI
// background task awaiting the network is parked rather than occupying
// an executor thread.
//
// TODO: multiple sources per style.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use futures::future::BoxFuture;

use crate::camera::{Camera, WorldPoint};
use crate::tile::TileId;
use crate::tilejson::{TileJson, TileJsonError};

pub type SourceError = Box<dyn std::error::Error + Send + Sync>;
pub type TileFuture = BoxFuture<'static, Result<Option<Vec<u8>>, SourceError>>;

#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error("could not fetch {url}")]
    Http { url: String, source: Box<ureq::Error> },
    #[error("invalid TileJSON")]
    TileJson(#[from] TileJsonError),
}

pub struct TileJsonSource {
    tilejson: TileJson,
    fetcher: Fetcher,
}

/// Concurrent tile downloads, matching the per-host connection limit
/// browsers (and so tile servers' expectations) use.
const FETCH_THREADS: usize = 6;

impl TileJsonSource {
    /// Download and parse a TileJSON document, e.g. MapTiler's
    /// `https://api.maptiler.com/tiles/v3-openmaptiles/tiles.json?key=…`.
    /// Blocks the calling thread for one request.
    pub fn fetch(url: &str) -> Result<Self, OpenError> {
        Ok(Self::new(TileJson::from_json_str(&fetch_text(url)?)?))
    }

    pub fn new(tilejson: TileJson) -> Self {
        Self { tilejson, fetcher: Fetcher::spawn(FETCH_THREADS) }
    }

    /// The document's `center` if it has one, else the middle of its
    /// `bounds`, else the whole world.
    pub fn initial_camera(&self) -> Camera {
        let tj = &self.tilejson;
        let min = tj.minzoom as f64;
        let (lon, lat, zoom) = match (tj.center.as_deref(), tj.bounds) {
            (Some(&[lon, lat, zoom, ..]), _) => (lon, lat, zoom),
            (Some(&[lon, lat]), _) => (lon, lat, min),
            (_, Some([west, south, east, north])) => ((west + east) / 2.0, (south + north) / 2.0, min),
            _ => (0.0, 0.0, min),
        };
        Camera::new(WorldPoint::from_lon_lat(lon, lat), zoom)
    }

    /// Inclusive range of zoom levels the source has tiles for.
    pub fn zoom_range(&self) -> (u8, u8) {
        (self.tilejson.minzoom, self.tilejson.maxzoom)
    }

    /// Decompressed MVT bytes, or `None` where the server has no tile.
    /// The download runs on a fetch thread; the future only waits for it.
    //
    // TODO: TileJSON allows several `tiles` templates for load
    // balancing; only the first is used.
    pub fn tile(&self, id: TileId) -> TileFuture {
        self.fetcher.get(self.tilejson.tile_url(id.z, id.x, id.y))
    }
}

/// GET a text document (TileJSON, a style). Non-2xx statuses are errors.
/// Blocks the calling thread.
pub fn fetch_text(url: &str) -> Result<String, OpenError> {
    let error = |source| OpenError::Http { url: redact(url).to_string(), source: Box::new(source) };
    ureq::get(url).call().map_err(error)?.body_mut().read_to_string().map_err(error)
}

/// Drop the query string, which for hosted tiles usually carries an API
/// key that shouldn't end up in logs or error messages.
fn redact(url: &str) -> &str {
    url.split_once('?').map_or(url, |(path, _)| path)
}

// ------------------------------------------------------------------------------
// Fetch threads
// ------------------------------------------------------------------------------
//
// A fixed pool of blocking HTTP workers fed by a queue. Each request
// carries a oneshot sender; the future handed to the view just waits on
// the matching receiver, so a GPUI background task awaiting a tile is
// parked rather than occupying an executor thread. Dropping the source
// closes the queue and the workers exit.

struct Fetcher {
    jobs: mpsc::Sender<FetchJob>,
}

struct FetchJob {
    url: String,
    reply: oneshot::Sender<Result<Option<Vec<u8>>, SourceError>>,
}

impl Fetcher {
    fn spawn(threads: usize) -> Self {
        let (jobs, queue) = mpsc::channel::<FetchJob>();
        let queue = Arc::new(Mutex::new(queue));
        // 404 / 204 mean "no tile here", so statuses are inspected rather
        // than turned into errors by ureq.
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .into();
        for index in 0..threads {
            let queue = queue.clone();
            let agent = agent.clone();
            std::thread::Builder::new()
                .name(format!("vec-map-kit-fetch-{index}"))
                .spawn(move || {
                    loop {
                        // The lock is held only while waiting for a job.
                        let job = queue.lock().expect("no fetch thread panics while holding the queue").recv();
                        let Ok(job) = job else { return };
                        // The view may have dropped the request already.
                        if job.reply.is_canceled() {
                            continue;
                        }
                        let _ = job.reply.send(get_tile(&agent, &job.url));
                    }
                })
                .expect("the OS can start a handful of threads");
        }
        Self { jobs }
    }

    fn get(&self, url: String) -> TileFuture {
        let (reply, response) = oneshot::channel();
        let queued = self.jobs.send(FetchJob { url, reply }).is_ok();
        Box::pin(async move {
            if !queued {
                return Err("fetch threads have stopped".into());
            }
            match response.await {
                Ok(result) => result,
                Err(_) => Err("fetch thread dropped the request".into()),
            }
        })
    }
}

fn get_tile(agent: &ureq::Agent, url: &str) -> Result<Option<Vec<u8>>, SourceError> {
    let mut response = agent.get(url).call()?;
    let status = response.status();
    if status == 404 || status == 204 {
        return Ok(None);
    }
    if !status.is_success() {
        return Err(format!("HTTP {status} for {}", redact(url)).into());
    }
    let body = response.body_mut().read_to_vec()?;
    // ureq undoes `Content-Encoding: gzip`, but some servers send gzipped
    // tiles without that header. Sniff the gzip magic and unpack those too.
    if body.starts_with(&[0x1f, 0x8b]) {
        let mut unpacked = Vec::with_capacity(body.len() * 4);
        std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(body.as_slice()), &mut unpacked)?;
        return Ok(Some(unpacked));
    }
    Ok(Some(body))
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    use futures::executor::block_on;

    use super::*;

    /// A one-file HTTP/1.1 server on a free localhost port. `routes` gets
    /// the server's base URL so documents can point back at it.
    fn serve(routes: impl FnOnce(&str) -> Vec<(String, u16, Vec<u8>)>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("localhost has a free port");
        let base = format!("http://{}", listener.local_addr().expect("bound socket has an address"));
        let routes = routes(&base);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().expect("socket clones"));
                let mut request_line = String::new();
                reader.read_line(&mut request_line).expect("request line");
                let mut header = String::new();
                while reader.read_line(&mut header).is_ok_and(|n| n > 2) {
                    header.clear();
                }
                let path = request_line.split_whitespace().nth(1).unwrap_or("").to_string();
                let (status, body) = routes
                    .iter()
                    .find(|(p, _, _)| *p == path)
                    .map(|(_, s, b)| (*s, b.clone()))
                    .unwrap_or((404, Vec::new()));
                let head = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        base
    }

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes).expect("in-memory write");
        encoder.finish().expect("in-memory write")
    }

    fn source_with_tiles(tiles: Vec<(&str, u16, Vec<u8>)>) -> TileJsonSource {
        let tiles: Vec<_> = tiles.into_iter().map(|(p, s, b)| (p.to_string(), s, b)).collect();
        let base = serve(move |base| {
            let doc = format!(
                r#"{{"tilejson": "3.0.0", "tiles": ["{base}/tiles/{{z}}/{{x}}/{{y}}.pbf?key=k"],
                    "minzoom": 2, "maxzoom": 14, "center": [9.52, 47.14, 12]}}"#
            );
            let mut routes = vec![("/tiles.json".to_string(), 200, doc.into_bytes())];
            routes.extend(tiles);
            routes
        });
        TileJsonSource::fetch(&format!("{base}/tiles.json")).expect("TileJSON loads")
    }

    #[test]
    fn fetch_reads_zoom_range_and_centre() {
        let source = source_with_tiles(vec![]);
        assert_eq!(source.zoom_range(), (2, 14));
        let camera = source.initial_camera();
        assert_eq!(camera.zoom, 12.0);
        let (lon, lat) = camera.center.to_lon_lat();
        assert!((lon - 9.52).abs() < 1e-9 && (lat - 47.14).abs() < 1e-9);
    }

    #[test]
    fn tiles_come_from_the_template_and_raw_gzip_is_unpacked() {
        let source = source_with_tiles(vec![
            ("/tiles/3/4/5.pbf?key=k", 200, gzip(b"zipped mvt")),
            ("/tiles/3/4/6.pbf?key=k", 200, b"plain mvt".to_vec()),
        ]);
        let zipped = block_on(source.tile(TileId::new(3, 4, 5))).expect("tile loads");
        let plain = block_on(source.tile(TileId::new(3, 4, 6))).expect("tile loads");
        assert_eq!(zipped.as_deref(), Some(&b"zipped mvt"[..]));
        assert_eq!(plain.as_deref(), Some(&b"plain mvt"[..]));
    }

    #[test]
    fn missing_tiles_are_none() {
        let source = source_with_tiles(vec![("/tiles/3/4/6.pbf?key=k", 204, Vec::new())]);
        assert!(block_on(source.tile(TileId::new(3, 4, 5))).expect("404 is not an error").is_none());
        assert!(block_on(source.tile(TileId::new(3, 4, 6))).expect("204 is not an error").is_none());
    }

    #[test]
    fn server_errors_are_reported() {
        let source = source_with_tiles(vec![("/tiles/3/4/5.pbf?key=k", 500, Vec::new())]);
        assert!(block_on(source.tile(TileId::new(3, 4, 5))).is_err());
    }

    #[test]
    fn invalid_tilejson_is_an_error() {
        let base = serve(|_| vec![("/tiles.json".to_string(), 200, br#"{"tiles": []}"#.to_vec())]);
        assert!(matches!(TileJsonSource::fetch(&format!("{base}/tiles.json")), Err(OpenError::TileJson(_))));
    }

    #[test]
    fn unreachable_tilejson_is_an_error() {
        let base = serve(|_| vec![]);
        assert!(matches!(TileJsonSource::fetch(&format!("{base}/tiles.json")), Err(OpenError::Http { .. })));
    }
}
