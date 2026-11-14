//! `rcad-mbd`: the multibody-dynamics backend (route card: `ROADMAP.md`).
//!
//! A standalone library in the `rcad-gcs` mould: pure math and data, no OCCT,
//! no B-rep. Mass properties arrive as numbers (the kernel's GProp output,
//! converted to SI by the caller); the crate consumes rigid bodies, markers,
//! joints, drivers and force elements and — from M1 on — produces state
//! trajectories and joint reactions.
//!
//! Architecture decisions (fixed, see the roadmap §3):
//! - **Maximal coordinates**: every body carries 6 DOF; joints are constraint
//!   equations (Lagrange multipliers from M2). Closed loops cost nothing.
//! - **SI units** inside the crate: kg, m, s, rad, N. Callers convert.
//! - **Deterministic**: tables are `BTreeMap`s keyed by stable ids, so
//!   equation numbering and serialized bytes are reproducible.
//! - **Failures are data**: [`MbdError`] `{code, detail, hint}`.
//!
//! - **Velocities in the world frame**: per free body, the linear velocity of
//!   the body-frame origin and the angular velocity, both inertial
//!   ([`Twist`]).
//!
//! Milestones: M0 data model + golden (done); M1 kinematic three levels
//! (position via `rcad-gcs`, velocity/acceleration linear solves against
//! the analytic Jacobian, redundancy diagnosis, time sweep; done); M2 forward
//! dynamics (equations of motion, dynamic IC, static equilibrium), index-3
//! HHT-α integration with error control, joint reactions (done); M3 force library
//! (function forces, follower loads, joint springs, bushings; done);
//! M4 assembly wiring. Nothing here pre-seeds M5 (contact/friction).

pub mod asmt;
pub mod body;
pub mod coupler;
pub mod driver;
pub mod dynamics;
pub mod error;
pub mod force;
pub mod integrate;
mod jet;
pub mod joint;
pub mod kinematics;
pub mod measure;
pub mod merge;
pub mod reaction;
mod linalg;
pub mod system;

pub use asmt::{write_asmt, AsmtSimulation};
pub use body::{Body, BodyId, Marker, MarkerId};
pub use coupler::{Coupler, CouplerId, CouplerTerm};
pub use driver::{Driver, DriverProfile, DriverSample, DriverSlot};
pub use dynamics::{
    dynamic_ic, energy, forward_dynamics, mass_matrix, momentum, power, static_equilibrium, velocity_ic,
    DynamicFrame,
    Energy, Momentum, StaticEquilibrium,
};
pub use error::{codes, MbdError};
pub use force::{Argument, ForceElement, ForceId, Function};
pub use integrate::{simulate, HhtSettings, IntegrationStats, Simulation, DEFAULT_ALPHA};
pub use joint::{Joint, JointId, JointKind};
pub use kinematics::{
    diagnose, equation_count, kinematic_sweep, position_ic, position_ic_from, solve_frame,
    EquationOwner, EquationRef, KinematicDiagnosis, KinematicFrame, KinematicState, Twist,
    BODY_PARAMS, BODY_TWIST,
};
pub use measure::{JointMotion, MarkerMotion, PointMotion, RelativeMotion};
pub use merge::{merge_fixed_groups, MergedSystem, Placement};
pub use reaction::{reactions, Reaction};
pub use system::{System, STANDARD_GRAVITY};

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat3, DQuat, DVec3};

    /// A grounded base, a 2 kg link on a revolute hinge, a constant-rate
    /// driver and a spring back to the base — one of each table.
    fn pendulum() -> System {
        let mut sys = System::new("pendulum").with_standard_gravity();
        sys.add_body(Body {
            id: BodyId(1),
            name: "base".into(),
            mass_kg: 0.0,
            com: DVec3::ZERO,
            inertia_com: DMat3::ZERO,
            translation: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            grounded: true,
            velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
        })
        .unwrap();
        sys.add_body(Body {
            id: BodyId(2),
            name: "link".into(),
            mass_kg: 2.0,
            com: DVec3::new(0.0, 0.0, -0.5),
            inertia_com: DMat3::from_diagonal(DVec3::new(0.17, 0.17, 0.01)),
            translation: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            grounded: false,
            velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
        })
        .unwrap();
        for (id, body) in [(1, BodyId(1)), (2, BodyId(2))] {
            sys.add_marker(Marker {
                id: MarkerId(id),
                name: format!("hinge{id}"),
                body,
                translation: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
            })
            .unwrap();
        }
        sys.add_joint(Joint {
            id: JointId(1),
            name: "hinge".into(),
            kind: JointKind::Revolute,
            marker1: MarkerId(1),
            marker2: MarkerId(2),
            suppressed: false,
        })
        .unwrap();
        sys.add_driver(Driver {
            joint: JointId(1),
            slot: DriverSlot::Rotation,
            profile: DriverProfile::Constant { start: 0.0, rate: 1.0 },
        });
        sys.add_force(ForceElement::SpringDamper {
            id: ForceId(1),
            name: "return".into(),
            marker1: MarkerId(1),
            marker2: MarkerId(2),
            stiffness: 100.0,
            damping: 1.0,
            free_length: 0.0,
            preload: 0.0,
        })
        .unwrap();
        sys
    }

    /// The same system serializes to identical bytes every time, and the
    /// bytes round-trip to an equal system (the golden contract).
    #[test]
    fn serialization_is_deterministic_and_round_trips() {
        let sys = pendulum();
        let a = serde_json::to_string(&sys).unwrap();
        let b = serde_json::to_string(&pendulum()).unwrap();
        assert_eq!(a, b, "identical systems serialize to identical bytes");
        let back: System = serde_json::from_str(&a).unwrap();
        assert_eq!(back, sys, "round trip is lossless");
        assert_eq!(serde_json::to_string(&back).unwrap(), a);
    }

    /// Insertion order never leaks: bodies added in reverse id order
    /// serialize in sorted order (the equation-numbering guarantee).
    #[test]
    fn tables_are_id_sorted_regardless_of_insertion_order() {
        let forward = pendulum();
        let mut reversed = System::new("pendulum").with_standard_gravity();
        let mut bodies: Vec<_> = forward.bodies.values().cloned().collect();
        bodies.reverse();
        for b in bodies {
            reversed.add_body(b).unwrap();
        }
        let mut markers: Vec<_> = forward.markers.values().cloned().collect();
        markers.reverse();
        for m in markers {
            reversed.add_marker(m).unwrap();
        }
        for j in forward.joints.values() {
            reversed.add_joint(j.clone()).unwrap();
        }
        for d in &forward.drivers {
            reversed.add_driver(d.clone());
        }
        for f in forward.forces.values() {
            reversed.add_force(f.clone()).unwrap();
        }
        assert_eq!(
            serde_json::to_string(&reversed).unwrap(),
            serde_json::to_string(&forward).unwrap()
        );
    }

    /// The integrity gates are data: dangling references, bad masses,
    /// unsupported driver slots and duplicate ids all name their code.
    #[test]
    fn validation_gates_name_their_codes() {
        assert!(pendulum().validate().is_ok());

        let mut dangling = pendulum();
        dangling.joints.get_mut(&JointId(1)).unwrap().marker2 = MarkerId(99);
        assert_eq!(dangling.validate().unwrap_err().code, codes::DANGLING_MARKER);

        let mut massless = pendulum();
        massless.bodies.get_mut(&BodyId(2)).unwrap().mass_kg = 0.0;
        assert_eq!(massless.validate().unwrap_err().code, codes::BAD_MASS_PROPERTIES);

        let mut bad_slot = pendulum();
        bad_slot.drivers[0].slot = DriverSlot::Translation;
        assert_eq!(bad_slot.validate().unwrap_err().code, codes::SLOT_UNSUPPORTED);

        let mut orphan_driver = pendulum();
        orphan_driver.drivers[0].joint = JointId(7);
        assert_eq!(orphan_driver.validate().unwrap_err().code, codes::DANGLING_JOINT);

        let mut dup = pendulum();
        let again = dup.bodies[&BodyId(2)].clone();
        assert_eq!(dup.add_body(again).unwrap_err().code, codes::DUPLICATE_ID);
    }

    /// Driver profiles: constant rate and clamped keyframe interpolation.
    #[test]
    fn driver_profiles_evaluate() {
        let constant = Driver {
            joint: JointId(1),
            slot: DriverSlot::Rotation,
            profile: DriverProfile::Constant { start: 0.5, rate: 2.0 },
        };
        assert!((constant.value_at(1.5) - 3.5).abs() < 1e-12);
        let keyed = Driver {
            joint: JointId(1),
            slot: DriverSlot::Translation,
            profile: DriverProfile::Keyframes {
                points: vec![(0.0, 0.0), (1.0, 10.0), (2.0, 10.0)],
            },
        };
        assert!((keyed.value_at(-1.0) - 0.0).abs() < 1e-12);
        assert!((keyed.value_at(0.5) - 5.0).abs() < 1e-12);
        assert!((keyed.value_at(9.0) - 10.0).abs() < 1e-12);
        assert!((keyed.sample(0.5).velocity - 10.0).abs() < 1e-12);
        assert_eq!(JointKind::Revolute.remaining_dof(), 1);
        assert_eq!(JointKind::Fixed.remaining_dof(), 0);
    }

    /// Every driver profile survives the serde round trip (the internally
    /// tagged enum needs struct variants: a tagged newtype holding a
    /// sequence does not serialize).
    #[test]
    fn every_driver_profile_round_trips() {
        let mut sys = pendulum();
        let profiles = [
            DriverProfile::Constant { start: 0.0, rate: 1.0 },
            DriverProfile::Keyframes { points: vec![(0.0, 0.0), (1.0, 2.0)] },
            DriverProfile::Harmonic { offset: 0.0, amplitude: 1.0, frequency_hz: 2.0, phase: 0.5 },
            DriverProfile::Step5 { t0: 0.0, v0: 0.0, t1: 1.0, v1: 1.0 },
            DriverProfile::Spline { points: vec![(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)] },
        ];
        for profile in profiles {
            sys.drivers[0].profile = profile;
            let text = serde_json::to_string(&sys).expect("serializes");
            let back: System = serde_json::from_str(&text).expect("deserializes");
            assert_eq!(back, sys);
        }
    }
}
