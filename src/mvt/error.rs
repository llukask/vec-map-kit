// Errors raised while lifting a `proto::Tile` into the high-level model.
//
// The decoder is strict about every constraint the MVT spec calls out
// (section 4.2 for tags, section 4.3 for geometry, section 4.3.4.4 for
// polygon ring orientation). We surface each violation as a distinct
// variant so callers can react programmatically — e.g. skip a single
// malformed feature rather than discarding the whole tile.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    // ----- geometry-stream framing -----
    #[error("geometry stream truncated after {consumed} words")]
    Truncated { consumed: usize },

    #[error("invalid command id {id} (expected 1, 2, or 7)")]
    InvalidCommandId { id: u32 },

    #[error("unexpected command {cmd} in {geom_type} sequence")]
    UnexpectedCommand {
        cmd: &'static str,
        geom_type: &'static str,
    },

    #[error("invalid count {count} for command {cmd}")]
    InvalidCount { cmd: &'static str, count: u32 },

    #[error("coordinate overflow while applying command at word {cmd_index}")]
    CoordOverflow { cmd_index: usize },

    #[error("empty geometry for {geom_type}")]
    EmptyGeometry { geom_type: &'static str },

    // ----- tag / property decoding -----
    #[error("feature has odd tag count ({count})")]
    OddTags { count: usize },

    #[error("tag {kind} index {idx} out of range (len {len})")]
    TagOutOfRange {
        kind: &'static str,
        idx: u32,
        len: usize,
    },

    #[error("MVT Value has {populated} variants set, expected exactly 1")]
    AmbiguousValue { populated: usize },
}

pub type Result<T> = std::result::Result<T, Error>;
