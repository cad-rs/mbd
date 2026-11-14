//! Structured errors: `{code, detail, hint}` data, never a panic at the
//! crate boundary (the tool-surface convention).

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("[{code}] {detail}")]
pub struct MbdError {
    /// Stable machine-readable code.
    pub code: &'static str,
    pub detail: String,
    pub hint: &'static str,
}

impl MbdError {
    pub fn new(code: &'static str, detail: impl Into<String>, hint: &'static str) -> Self {
        Self {
            code,
            detail: detail.into(),
            hint,
        }
    }
}

/// The error codes this crate emits.
pub mod codes {
    /// A marker references a body that is not in the system.
    pub const DANGLING_BODY: &str = "dangling_body";
    /// A joint or force element references a marker that is not in the system.
    pub const DANGLING_MARKER: &str = "dangling_marker";
    /// A driver references a joint that is not in the system.
    pub const DANGLING_JOINT: &str = "dangling_joint";
    /// A driver slot is not a degree of freedom of its joint kind.
    pub const SLOT_UNSUPPORTED: &str = "slot_unsupported";
    /// A body has a non-positive mass or a non-finite property.
    pub const BAD_MASS_PROPERTIES: &str = "bad_mass_properties";
    /// Two entities share one id in a table.
    pub const DUPLICATE_ID: &str = "duplicate_id";
}
