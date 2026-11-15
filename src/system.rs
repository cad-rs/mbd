//! The system: every table the solver reads, keyed by stable ids in sorted
//! order so equation numbering and serialized bytes are reproducible.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::body::{Body, BodyId, Marker, MarkerId};
use crate::coupler::{Coupler, CouplerId};
use crate::error::{codes, MbdError};
use crate::force::{ForceElement, ForceId};
use crate::driver::Driver;
use crate::joint::{Joint, JointId};

/// Standard gravity, SI (m/s²).
pub const STANDARD_GRAVITY: f64 = 9.806_65;

/// A multibody system: bodies, markers, joints, drivers, force elements and
/// the gravity vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct System {
    pub name: String,
    pub bodies: BTreeMap<BodyId, Body>,
    pub markers: BTreeMap<MarkerId, Marker>,
    pub joints: BTreeMap<JointId, Joint>,
    /// Drivers in authoring order (a joint slot carries at most one; the
    /// validation gate enforces it).
    pub drivers: Vec<Driver>,
    pub forces: BTreeMap<ForceId, ForceElement>,
    /// Gravity in the inertial frame (m/s²); `None` = no gravity.
    pub gravity: Option<DVec3>,
    /// Joint-coordinate couplers (gears, rack-and-pinion, screws). Absent
    /// in documents written before couplers existed.
    #[serde(default)]
    pub couplers: BTreeMap<CouplerId, Coupler>,
}

impl System {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            bodies: BTreeMap::new(),
            markers: BTreeMap::new(),
            joints: BTreeMap::new(),
            drivers: Vec::new(),
            forces: BTreeMap::new(),
            gravity: None,
            couplers: BTreeMap::new(),
        }
    }

    /// Standard gravity along -Z.
    pub fn with_standard_gravity(mut self) -> Self {
        self.gravity = Some(DVec3::new(0.0, 0.0, -STANDARD_GRAVITY));
        self
    }

    pub fn add_body(&mut self, body: Body) -> Result<BodyId, MbdError> {
        insert_new(&mut self.bodies, body.id, body.id.0, "body", body)
    }

    pub fn add_marker(&mut self, marker: Marker) -> Result<MarkerId, MbdError> {
        insert_new(&mut self.markers, marker.id, marker.id.0, "marker", marker)
    }

    pub fn add_joint(&mut self, joint: Joint) -> Result<JointId, MbdError> {
        insert_new(&mut self.joints, joint.id, joint.id.0, "joint", joint)
    }

    pub fn add_driver(&mut self, driver: Driver) {
        self.drivers.push(driver);
    }

    pub fn add_force(&mut self, force: ForceElement) -> Result<ForceId, MbdError> {
        let id = force.id();
        insert_new(&mut self.forces, id, id.0, "force", force)
    }

    pub fn add_coupler(&mut self, coupler: Coupler) -> Result<CouplerId, MbdError> {
        insert_new(&mut self.couplers, coupler.id, coupler.id.0, "coupler", coupler)
    }

    /// Referential and physical integrity: every reference resolves; every
    /// free mass is positive and finite; force elements, driver profiles and
    /// couplers carry usable data; every driver, coupler term and measured
    /// coordinate addresses a degree of freedom of its joint kind; a joint
    /// slot takes one driver. The solver entry points run this first.
    pub fn validate(&self) -> Result<(), MbdError> {
        for body in self.bodies.values() {
            let finite = body.mass_kg.is_finite()
                && body.com.is_finite()
                && body.inertia_com.to_cols_array().iter().all(|v| v.is_finite());
            if !finite || !(body.grounded || body.mass_kg > 0.0) {
                return Err(MbdError::new(
                    codes::BAD_MASS_PROPERTIES,
                    format!("body {} ({}) has mass {} or non-finite properties", body.id.0, body.name, body.mass_kg),
                    "every free body needs a positive finite mass; grounded bodies may carry 0",
                ));
            }
        }
        for marker in self.markers.values() {
            if !self.bodies.contains_key(&marker.body) {
                return Err(MbdError::new(
                    codes::DANGLING_BODY,
                    format!("marker {} ({}) references missing body {}", marker.id.0, marker.name, marker.body.0),
                    "add the body before its markers",
                ));
            }
        }
        for joint in self.joints.values() {
            for marker in [joint.marker1, joint.marker2] {
                if !self.markers.contains_key(&marker) {
                    return Err(MbdError::new(
                        codes::DANGLING_MARKER,
                        format!("joint {} ({}) references missing marker {}", joint.id.0, joint.name, marker.0),
                        "add both markers before the joint",
                    ));
                }
            }
        }
        for force in self.forces.values() {
            for marker in force.markers() {
                if !self.markers.contains_key(&marker) {
                    return Err(MbdError::new(
                        codes::DANGLING_MARKER,
                        format!("force element {} references missing marker {}", force.id().0, marker.0),
                        "add the markers before the force element",
                    ));
                }
            }
            if let Err(why) = force.check() {
                return Err(MbdError::new(
                    codes::BAD_FORCE_ELEMENT,
                    format!("force element {} ({}): {why}", force.id().0, force.name()),
                    "give finite parameters, a non-zero direction and a valid curve",
                ));
            }
            if let Some((joint, slot)) = force.measured_joint() {
                let Some(j) = self.joints.get(&joint) else {
                    return Err(MbdError::new(
                        codes::DANGLING_JOINT,
                        format!("force element {} measures missing joint {}", force.id().0, joint.0),
                        "add the joint before measuring it",
                    ));
                };
                if !j.kind.drives(slot) {
                    return Err(MbdError::new(
                        codes::SLOT_UNSUPPORTED,
                        format!(
                            "force element {} measures {slot:?}, not a coordinate of {:?} joint {}",
                            force.id().0, j.kind, joint.0
                        ),
                        "measure a rotation (revolute, cylindrical, planar) or translation (prismatic, cylindrical) coordinate",
                    ));
                }
            }
        }
        let mut driven = BTreeSet::new();
        for driver in &self.drivers {
            let Some(joint) = self.joints.get(&driver.joint) else {
                return Err(MbdError::new(
                    codes::DANGLING_JOINT,
                    format!("driver references missing joint {}", driver.joint.0),
                    "add the joint before driving it",
                ));
            };
            if !joint.kind.drives(driver.slot) {
                return Err(MbdError::new(
                    codes::SLOT_UNSUPPORTED,
                    format!(
                        "driver slot {:?} is not a drivable degree of freedom of {:?} joint {} ({})",
                        driver.slot, joint.kind, joint.id.0, joint.name
                    ),
                    "rotation drives revolute/cylindrical/planar; translation drives prismatic/cylindrical",
                ));
            }
            if !driven.insert((driver.joint, driver.slot)) {
                return Err(MbdError::new(
                    codes::DUPLICATE_DRIVER,
                    format!(
                        "joint {} ({}) slot {:?} carries two drivers",
                        joint.id.0, joint.name, driver.slot
                    ),
                    "a joint slot takes at most one driver; remove or merge the extra one",
                ));
            }
            if let Err(reason) = driver.profile.check() {
                return Err(MbdError::new(
                    codes::BAD_DRIVER_PROFILE,
                    format!("driver on joint {} ({}): {reason}", joint.id.0, joint.name),
                    "fix the profile data (finite numbers, sorted times, enough points)",
                ));
            }
        }
        for coupler in self.couplers.values() {
            let bad = |detail: String| {
                Err(MbdError::new(
                    codes::BAD_COUPLER,
                    format!("coupler {} ({}): {detail}", coupler.id.0, coupler.name),
                    "a coupler needs finite non-zero coefficients on measurable coordinates of existing joints",
                ))
            };
            if coupler.terms.is_empty() {
                return bad("no terms".into());
            }
            for term in &coupler.terms {
                let Some(joint) = self.joints.get(&term.joint) else {
                    return Err(MbdError::new(
                        codes::DANGLING_JOINT,
                        format!(
                            "coupler {} ({}) references missing joint {}",
                            coupler.id.0, coupler.name, term.joint.0
                        ),
                        "add the joint before coupling it",
                    ));
                };
                if !joint.kind.drives(term.slot) {
                    return bad(format!(
                        "{:?} is not a coordinate of {:?} joint {}",
                        term.slot, joint.kind, joint.id.0
                    ));
                }
                if !term.coefficient.is_finite() || term.coefficient == 0.0 {
                    return bad(format!("coefficient {} on joint {}", term.coefficient, joint.id.0));
                }
            }
        }
        Ok(())
    }
}

/// Insert under a fresh id; a taken id is [`codes::DUPLICATE_ID`].
fn insert_new<K: Ord + Copy, V>(
    table: &mut BTreeMap<K, V>,
    id: K,
    raw: u64,
    kind: &str,
    value: V,
) -> Result<K, MbdError> {
    match table.entry(id) {
        Entry::Occupied(_) => Err(MbdError::new(
            codes::DUPLICATE_ID,
            format!("{kind} id {raw} is already in the system"),
            "ids are stable keys; allocate a fresh one",
        )),
        Entry::Vacant(slot) => {
            slot.insert(value);
            Ok(id)
        }
    }
}
