//! Measures: what a CAD motion study plots — trace paths, point and marker
//! kinematics, relative motion between markers, joint coordinates — read off
//! a solved [`KinematicFrame`].
//!
//! Conventions (the crate's): world frame, SI. A point's velocity and
//! acceleration are the time derivatives of its world position (`v + ω×r`,
//! `a + α×r + ω×(ω×r)`). Relative measures follow the Adams `DX/VX/ACCX`
//! convention: world-frame differences of marker J against marker I,
//! **expressed** in marker I's axes (not differentiated in I's rotating
//! frame).

use glam::{DQuat, DVec3};

use crate::body::{BodyId, MarkerId};
use crate::driver::DriverSlot;
use crate::joint::JointId;
use crate::kinematics::{check_state, coordinate_jet, KinematicFrame, Twist};
use crate::system::System;

/// A point's world position and its first two time derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct PointMotion {
    pub position: DVec3,
    pub velocity: DVec3,
    pub acceleration: DVec3,
}

/// A marker's world frame and its rates: the origin's velocity /
/// acceleration with the carrying body's angular velocity / acceleration.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MarkerMotion {
    pub position: DVec3,
    pub rotation: DQuat,
    pub velocity: Twist,
    pub acceleration: Twist,
}

/// Marker J relative to marker I: J's origin minus I's (position, velocity,
/// acceleration — world-frame differences expressed in I's axes), the
/// relative angular velocity / acceleration (world differences expressed in
/// I's axes) and the relative rotation `R_I⁻¹·R_J`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RelativeMotion {
    pub displacement: DVec3,
    pub velocity: DVec3,
    pub acceleration: DVec3,
    pub angular_velocity: DVec3,
    pub angular_acceleration: DVec3,
    pub rotation: DQuat,
}

/// A joint coordinate (the slot a driver or coupler addresses) and its
/// first two time derivatives; rotation values are unwrapped across turns.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct JointMotion {
    pub value: f64,
    pub velocity: f64,
    pub acceleration: f64,
}

impl KinematicFrame {
    /// A body-fixed point (body-frame coordinates `local`). `None` for a
    /// body not in the frame.
    pub fn point(&self, body: BodyId, local: DVec3) -> Option<PointMotion> {
        let (t, r) = self.state.body_pose(body)?;
        let v = self.velocity.get(&body)?;
        let a = self.acceleration.get(&body)?;
        let arm = r * local;
        let w_arm = v.angular.cross(arm);
        Some(PointMotion {
            position: t + arm,
            velocity: v.linear + w_arm,
            acceleration: a.linear + a.angular.cross(arm) + v.angular.cross(w_arm),
        })
    }

    /// A marker's world frame and rates. `None` for a marker not in `sys`.
    pub fn marker(&self, sys: &System, marker: MarkerId) -> Option<MarkerMotion> {
        let m = sys.markers.get(&marker)?;
        let p = self.point(m.body, m.translation)?;
        let (_, r) = self.state.body_pose(m.body)?;
        Some(MarkerMotion {
            position: p.position,
            rotation: r * m.rotation,
            velocity: Twist {
                linear: p.velocity,
                angular: self.velocity[&m.body].angular,
            },
            acceleration: Twist {
                linear: p.acceleration,
                angular: self.acceleration[&m.body].angular,
            },
        })
    }

    /// Marker `j` relative to marker `i` (Adams `DX/VX/ACCX/WX/WDTX` with
    /// `i` as both the reference and the expression frame).
    pub fn relative(&self, sys: &System, i: MarkerId, j: MarkerId) -> Option<RelativeMotion> {
        let mi = self.marker(sys, i)?;
        let mj = self.marker(sys, j)?;
        let into_i = mi.rotation.inverse();
        Some(RelativeMotion {
            displacement: into_i * (mj.position - mi.position),
            velocity: into_i * (mj.velocity.linear - mi.velocity.linear),
            acceleration: into_i * (mj.acceleration.linear - mi.acceleration.linear),
            angular_velocity: into_i * (mj.velocity.angular - mi.velocity.angular),
            angular_acceleration: into_i * (mj.acceleration.angular - mi.acceleration.angular),
            rotation: (into_i * mj.rotation).normalize(),
        })
    }

    /// A joint coordinate with its rate and acceleration (rotation
    /// unwrapped from the frame's tracked angle). `None` when the joint is
    /// missing, suppressed, or has no such coordinate.
    pub fn joint_coordinate(
        &self,
        sys: &System,
        joint: JointId,
        slot: DriverSlot,
    ) -> Option<JointMotion> {
        let j = sys.joints.get(&joint)?;
        if j.suppressed || !j.kind.drives(slot) {
            return None;
        }
        check_state(sys, &self.state).ok()?;
        let state = &self.state;
        // The jet along the velocities gives the rate and the
        // velocity-quadratic part, the jet along the accelerations the
        // linear part `J·q̈`.
        let vel: Vec<Twist> = state.order.iter().map(|id| self.velocity[id]).collect();
        let acc: Vec<Twist> = state.order.iter().map(|id| self.acceleration[id]).collect();
        let jv = coordinate_jet(sys, &state.slot_of, &state.params, &vel, joint, slot);
        let ja = coordinate_jet(sys, &state.slot_of, &state.params, &acc, joint, slot);
        let value = match slot {
            DriverSlot::Rotation => state.joint_angle(joint).unwrap_or(jv.v),
            DriverSlot::Translation => jv.v,
        };
        Some(JointMotion {
            value,
            velocity: jv.d,
            acceleration: jv.dd + ja.d,
        })
    }
}
