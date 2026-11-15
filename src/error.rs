//! Structured errors: `{code, detail, hint}` data, never a panic at the
//! crate boundary.

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
    /// The kinematic position solve did not converge below the residual
    /// tolerance (over-constrained layout or a bad seed).
    pub const KINEMATICS_NOT_CONVERGED: &str = "kinematics_not_converged";
    /// Two drivers prescribe the same slot of one joint.
    pub const DUPLICATE_DRIVER: &str = "duplicate_driver";
    /// A driver profile's data is unusable (non-finite, unsorted, too few
    /// points).
    pub const BAD_DRIVER_PROFILE: &str = "bad_driver_profile";
    /// A seed state does not match the system's body table.
    pub const STATE_MISMATCH: &str = "state_mismatch";
    /// The velocity/acceleration levels need a fully prescribed motion: the
    /// mechanism still has free degrees of freedom (add drivers).
    pub const UNDERDETERMINED_MOTION: &str = "underdetermined_motion";
    /// The velocity or acceleration equations are inconsistent: the
    /// mechanism is at a lock-up / toggle position, or the drivers
    /// over-prescribe a degree of freedom.
    pub const KINEMATIC_SINGULAR: &str = "kinematic_singular";
    /// A time grid is not finite and non-decreasing.
    pub const BAD_TIME_GRID: &str = "bad_time_grid";
    /// A coupler has no terms, a zero or non-finite coefficient, or a term on
    /// a coordinate its joint kind does not have.
    pub const BAD_COUPLER: &str = "bad_coupler";
    /// Fixed joints that cannot all hold: a weld loop inside a rigid group
    /// disagrees with the tree placement, or two grounded bodies are welded
    /// off their authored poses.
    pub const INCONSISTENT_WELD: &str = "inconsistent_weld";
    /// The system uses something the `.asmt` export does not cover.
    pub const ASMT_UNSUPPORTED: &str = "asmt_unsupported";
    /// The dynamics saddle-point equations are inconsistent: a lock-up /
    /// toggle position, drivers over-prescribing a degree of freedom, or a
    /// load along a direction nothing resists (no mass, stiffness or
    /// constraint).
    pub const DYNAMICS_SINGULAR: &str = "dynamics_singular";
    /// The static-equilibrium Newton iteration did not converge.
    pub const STATIC_NOT_CONVERGED: &str = "static_not_converged";
    /// A force element is geometrically degenerate (a spring with coincident
    /// markers but a nonzero free length).
    pub const DEGENERATE_FORCE: &str = "degenerate_force";
    /// Integrator settings out of range (α outside [−1/3, 0], a
    /// non-positive step or tolerance).
    pub const BAD_INTEGRATOR_SETTINGS: &str = "bad_integrator_settings";
    /// The integrator could not complete a step even at its minimum step.
    pub const INTEGRATION_FAILED: &str = "integration_failed";
    /// A force element's numbers are unusable (non-finite parameters, a
    /// zero direction, an invalid curve).
    pub const BAD_FORCE_ELEMENT: &str = "bad_force_element";
    /// A study is unusable (an empty or non-finite output grid, a measure
    /// on an entity the system does not have).
    pub const BAD_STUDY: &str = "bad_study";
    /// The system uses something the MJCF export does not cover.
    pub const MJCF_UNSUPPORTED: &str = "mjcf_unsupported";
}
