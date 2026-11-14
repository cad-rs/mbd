//! Couplers: linear relations between joint coordinates (Adams `COUPLER`)
//! — gears, rack-and-pinion, screws, and any other fixed-ratio
//! transmission a CAD assembly carries as a "mechanical mate".
//!
//! A coupler holds `Σ cᵢ·Δqᵢ = 0`, where `Δqᵢ` is the change of joint
//! coordinate `qᵢ` (a rotation angle or an axial offset, the same
//! coordinates a driver prescribes) from the authored configuration. The
//! authored assembly therefore always satisfies its couplers, the way a
//! CAD gear mate holds the teeth where they were placed.
//!
//! Rotation coordinates are unwrapped across turns: the solver tracks each
//! rotation coordinate continuously from instant to instant (a sweep), so a
//! gear may run any number of revolutions as long as one solve step turns it
//! by less than half a revolution.

use serde::{Deserialize, Serialize};

use crate::driver::DriverSlot;
use crate::joint::JointId;

/// Stable coupler identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CouplerId(pub u64);

/// One term `c·Δq` of a coupler: the coordinate `slot` of `joint`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CouplerTerm {
    pub joint: JointId,
    pub slot: DriverSlot,
    pub coefficient: f64,
}

/// A linear relation between joint coordinates (one equation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Coupler {
    pub id: CouplerId,
    pub name: String,
    pub terms: Vec<CouplerTerm>,
}

impl Coupler {
    /// A gear pair: `Δθ_output = ratio · Δθ_input` (both rotation
    /// coordinates). The sign is the caller's: with both joint axes pointing
    /// the same way, an external mesh turns the other way (`ratio < 0`,
    /// `ratio = −r_in / r_out`), an internal mesh the same way.
    pub fn gear(id: CouplerId, name: &str, input: JointId, output: JointId, ratio: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            terms: vec![
                term(input, DriverSlot::Rotation, ratio),
                term(output, DriverSlot::Rotation, -1.0),
            ],
        }
    }

    /// Rack and pinion: `Δx_rack = pitch_radius · Δθ_pinion` (the rack's
    /// translation coordinate against the pinion's rotation).
    pub fn rack_pinion(
        id: CouplerId,
        name: &str,
        pinion: JointId,
        rack: JointId,
        pitch_radius: f64,
    ) -> Self {
        Self {
            id,
            name: name.to_string(),
            terms: vec![
                term(pinion, DriverSlot::Rotation, pitch_radius),
                term(rack, DriverSlot::Translation, -1.0),
            ],
        }
    }

    /// A screw on one cylindrical joint: `Δz = lead / 2π · Δθ` (`lead` the
    /// advance per revolution; negative for a left-hand thread).
    pub fn screw(id: CouplerId, name: &str, joint: JointId, lead: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            terms: vec![
                term(joint, DriverSlot::Rotation, lead / std::f64::consts::TAU),
                term(joint, DriverSlot::Translation, -1.0),
            ],
        }
    }
}

fn term(joint: JointId, slot: DriverSlot, coefficient: f64) -> CouplerTerm {
    CouplerTerm {
        joint,
        slot,
        coefficient,
    }
}
