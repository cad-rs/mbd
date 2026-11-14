//! The system: every table the solver reads, keyed by stable ids in sorted
//! order so equation numbering and serialized bytes are reproducible
//! (OndselSolver's `createMbD` name-sort lesson).

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::body::{Body, BodyId, Marker, MarkerId};
use crate::error::{codes, MbdError};
use crate::force::{ForceElement, ForceId};
use crate::joint::{Driver, DriverSlot, Joint, JointId};

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
        }
    }

    /// Standard gravity along -Z.
    pub fn with_standard_gravity(mut self) -> Self {
        self.gravity = Some(DVec3::new(0.0, 0.0, -STANDARD_GRAVITY));
        self
    }

    pub fn add_body(&mut self, body: Body) -> Result<BodyId, MbdError> {
        let id = body.id;
        if self.bodies.contains_key(&id) {
            return Err(duplicate("body", id.0));
        }
        self.bodies.insert(id, body);
        Ok(id)
    }

    pub fn add_marker(&mut self, marker: Marker) -> Result<MarkerId, MbdError> {
        let id = marker.id;
        if self.markers.contains_key(&id) {
            return Err(duplicate("marker", id.0));
        }
        self.markers.insert(id, marker);
        Ok(id)
    }

    pub fn add_joint(&mut self, joint: Joint) -> Result<JointId, MbdError> {
        let id = joint.id;
        if self.joints.contains_key(&id) {
            return Err(duplicate("joint", id.0));
        }
        self.joints.insert(id, joint);
        Ok(id)
    }

    pub fn add_driver(&mut self, driver: Driver) {
        self.drivers.push(driver);
    }

    pub fn add_force(&mut self, force: ForceElement) -> Result<ForceId, MbdError> {
        let id = force.id();
        if self.forces.contains_key(&id) {
            return Err(duplicate("force", id.0));
        }
        self.forces.insert(id, force);
        Ok(id)
    }

    /// Referential and physical integrity: every reference resolves, every
    /// mass is positive and finite, every driver slot is a degree of freedom
    /// of its joint kind. The solver entry points (M1+) run this first.
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
            if !driven.insert((driver.joint, driver.slot == DriverSlot::Rotation)) {
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
        Ok(())
    }
}

fn duplicate(table: &str, id: u64) -> MbdError {
    MbdError::new(
        codes::DUPLICATE_ID,
        format!("{table} id {id} is already in the system"),
        "ids are stable keys; allocate a fresh one",
    )
}
