//! Export a [`System`] to OndselSolver's `.asmt` text format (the format
//! FreeCAD's Assembly workbench feeds its solver).
//!
//! Two uses: the Tier B reference pipeline (roadmap §6 — an independent
//! OndselSolver build runs the exported file and its time series becomes the
//! reference), and interchange with the FreeCAD ecosystem.
//!
//! The mapping follows what FreeCAD itself writes:
//! - every body becomes a `Part` (frame = the body's authored pose, a
//!   `PrincipalMassMarker` at the center of mass);
//! - a grounded body is a `Part` held by a `FixedJoint` to an assembly-level
//!   marker at its pose;
//! - joints map one to one (`Prismatic` → `TranslationalJoint`), marker 1 →
//!   `MarkerI`, marker 2 → `MarkerJ` — the MbD formulations are the same
//!   direction-cosine sets this crate uses;
//! - drivers become `RotationalMotion` / `TranslationalMotion` on their joint
//!   with a `time` expression (constant-rate and harmonic profiles; the
//!   sampled profiles have no expression and are refused). A joint driven
//!   in both rotation and translation gives the rotation motion its own
//!   alias I marker (see the comment in [`write_asmt`]).
//!
//! Verified against an OndselSolver build (Tier B, `tests/tier_b.rs`): the
//! exported cases agree with this crate's kinematics to ≤ 4e-10.
//!
//! Export only: the kinematic subset plus constant gravity (the joint and
//! motion reaction series OndselSolver writes then carry the gravity load —
//! the Tier B reference for reaction recovery). No force elements.


use glam::{DMat3, DQuat, DVec3};

use crate::body::{Body, BodyId, MarkerId};
use crate::driver::{DriverProfile, DriverSlot};
use crate::error::{codes, MbdError};
use crate::joint::JointKind;
use crate::system::System;

/// The analysis window written to `SimulationParameters`: output every
/// `hout` seconds from `tstart` to `tend`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsmtSimulation {
    pub tstart: f64,
    pub tend: f64,
    pub hout: f64,
}

const ASSEMBLY: &str = "Assembly";

fn part_name(body: BodyId) -> String {
    format!("b{}", body.0)
}

/// Export `sys` as `.asmt` text.
///
/// Refused as [`codes::ASMT_UNSUPPORTED`]: a body inertia that is not
/// diagonal in the body axes (principal axes are not derived here), a
/// driver profile without a closed-form expression (keyframes, step,
/// spline), force elements, and couplers (OndselSolver's gear /
/// rack-pinion / screw joints relate two markers directly, not joint
/// coordinates, so there is no faithful one-to-one mapping).
pub fn write_asmt(sys: &System, sim: &AsmtSimulation) -> Result<String, MbdError> {
    sys.validate()?;
    if !sys.forces.is_empty() {
        return Err(unsupported(
            "force elements are not exported (kinematic subset plus gravity)",
        ));
    }
    if !sys.couplers.is_empty() {
        return Err(unsupported(
            "couplers are not exported (OndselSolver couples markers, not joint coordinates)",
        ));
    }
    // OndselSolver realizes a motion by turning its I marker into a
    // time-dependent frame; two motions on one joint share that marker and
    // the later one overwrites the earlier one's prescription (observed: a
    // cylindrical joint with both drivers did not rotate). The rotation
    // driver of a doubly-driven joint therefore gets its own I marker, an
    // alias of marker 1 at the same pose, and names its markers explicitly.
    let aliased: Vec<usize> = sys
        .drivers
        .iter()
        .enumerate()
        .filter(|(_, d)| {
            d.slot == DriverSlot::Rotation
                && sys
                    .drivers
                    .iter()
                    .any(|o| o.joint == d.joint && o.slot == DriverSlot::Translation)
        })
        .map(|(i, _)| i)
        .collect();
    let mut out = String::new();
    let w = &mut out;
    line(w, 0, "OndselSolver");
    line(w, 0, "Assembly");
    line(w, 1, "Notes");
    line(
        w,
        2,
        &format!(
            "(Text string: 'rcad-mbd export: {}' runs: (Core.RunArray runs: #() values: #()))",
            sys.name.replace('\'', "")
        ),
    );
    line(w, 1, "Name");
    line(w, 2, ASSEMBLY);
    frame(w, 1, DVec3::ZERO, DQuat::IDENTITY);
    rates(w, 1);

    line(w, 1, "RefPoints");
    for body in sys.bodies.values().filter(|b| b.grounded) {
        let name = format!("ground{}", body.id.0);
        ref_point(w, 2, &name, body.translation, body.rotation);
    }
    line(w, 1, "RefCurves");
    line(w, 1, "RefSurfaces");

    line(w, 1, "Parts");
    for body in sys.bodies.values() {
        line(w, 2, "Part");
        line(w, 3, "Name");
        line(w, 4, &part_name(body.id));
        frame(w, 3, body.translation, body.rotation);
        rates(w, 3);
        line(w, 3, "FeatureOrder");
        mass_marker(w, 3, body)?;
        line(w, 3, "RefPoints");
        if body.grounded {
            ref_point(w, 4, "FixingMarker", DVec3::ZERO, DQuat::IDENTITY);
        }
        for marker in sys.markers.values().filter(|m| m.body == body.id) {
            let name = format!("m{}", marker.id.0);
            ref_point(w, 4, &name, marker.translation, marker.rotation);
            for &index in &aliased {
                if sys.joints[&sys.drivers[index].joint].marker1 == marker.id {
                    let alias = alias_name(marker.id, index);
                    ref_point(w, 4, &alias, marker.translation, marker.rotation);
                }
            }
        }
        line(w, 3, "RefCurves");
        line(w, 3, "RefSurfaces");
    }

    line(w, 1, "KinematicIJs");
    line(w, 1, "ConstraintSets");
    line(w, 2, "Joints");
    for body in sys.bodies.values().filter(|b| b.grounded) {
        line(w, 3, "FixedJoint");
        item_ij(
            w,
            &format!("grounded{}", body.id.0),
            &format!("/{ASSEMBLY}/ground{}", body.id.0),
            &format!("/{ASSEMBLY}/{}/FixingMarker", part_name(body.id)),
        );
    }
    for joint in sys.joints.values().filter(|j| !j.suppressed) {
        line(w, 3, joint_keyword(joint.kind));
        item_ij(
            w,
            &format!("j{}", joint.id.0),
            &marker_path(sys, joint.marker1),
            &marker_path(sys, joint.marker2),
        );
    }
    line(w, 2, "Motions");
    for (index, driver) in sys.drivers.iter().enumerate() {
        let Some(joint) = sys.joints.get(&driver.joint).filter(|j| !j.suppressed) else {
            continue;
        };
        let expr = expression(&driver.profile)?;
        let (keyword, slot) = match driver.slot {
            DriverSlot::Rotation => ("RotationalMotion", "RotationZ"),
            DriverSlot::Translation => ("TranslationalMotion", "TranslationZ"),
        };
        line(w, 3, keyword);
        line(w, 4, "Name");
        line(w, 5, &format!("d{index}"));
        if aliased.contains(&index) {
            // Own I frame (an alias of marker 1), no motion joint.
            line(w, 4, "MarkerI");
            line(w, 5, &alias_path(sys, joint.marker1, index));
            line(w, 4, "MarkerJ");
            line(w, 5, &marker_path(sys, joint.marker2));
            line(w, 4, "MotionJoint");
            line(w, 5, "");
        } else {
            line(w, 4, "MotionJoint");
            line(w, 5, &format!("/{ASSEMBLY}/j{}", joint.id.0));
        }
        line(w, 4, slot);
        line(w, 5, &expr);
    }
    line(w, 2, "GeneralConstraintSets");
    line(w, 1, "ForceTorques");
    line(w, 1, "ConstantGravity");
    line(w, 2, &vector(sys.gravity.unwrap_or(DVec3::ZERO)));

    line(w, 1, "SimulationParameters");
    for (key, value) in [
        ("tstart", sim.tstart),
        ("tend", sim.tend),
        ("hmin", 1.0e-9),
        ("hmax", 1.0),
        ("hout", sim.hout),
        ("errorTol", 1.0e-6),
    ] {
        line(w, 2, key);
        line(w, 3, &number(value));
    }
    let frames = ((sim.tend - sim.tstart) / sim.hout).round().max(0.0) as u64 + 1;
    line(w, 1, "AnimationParameters");
    for (key, value) in [
        ("nframe", frames.to_string()),
        ("icurrent", "1".into()),
        ("istart", "1".into()),
        ("iend", frames.to_string()),
        ("isForward", "true".into()),
        ("framesPerSecond", "30".into()),
    ] {
        line(w, 2, key);
        line(w, 3, &value);
    }
    Ok(out)
}

fn unsupported(detail: &str) -> MbdError {
    MbdError::new(
        codes::ASMT_UNSUPPORTED,
        detail,
        "the .asmt export covers the kinematic subset: lower-pair joints, constant-rate and harmonic drivers, diagonal inertia",
    )
}

fn joint_keyword(kind: JointKind) -> &'static str {
    match kind {
        JointKind::Revolute => "RevoluteJoint",
        JointKind::Prismatic => "TranslationalJoint",
        JointKind::Cylindrical => "CylindricalJoint",
        JointKind::Spherical => "SphericalJoint",
        JointKind::Planar => "PlanarJoint",
        JointKind::Universal => "UniversalJoint",
        JointKind::Fixed => "FixedJoint",
        JointKind::InLine => "PointInLineJoint",
        JointKind::InPlane => "PointInPlaneJoint",
        JointKind::Orientation => "NoRotationJoint",
        JointKind::ParallelAxes => "ParallelAxesJoint",
        JointKind::Perpendicular => "PerpendicularJoint",
    }
}

/// The path of a marker-like name on the part carrying `marker`.
fn path_on_part(sys: &System, marker: MarkerId, name: &str) -> String {
    format!("/{ASSEMBLY}/{}/{name}", part_name(sys.markers[&marker].body))
}

fn marker_path(sys: &System, marker: MarkerId) -> String {
    path_on_part(sys, marker, &format!("m{}", marker.0))
}

fn alias_name(marker: MarkerId, driver: usize) -> String {
    format!("m{}d{driver}", marker.0)
}

fn alias_path(sys: &System, marker: MarkerId, driver: usize) -> String {
    path_on_part(sys, marker, &alias_name(marker, driver))
}

/// The driver profile as an OndselSolver `time` expression.
fn expression(profile: &DriverProfile) -> Result<String, MbdError> {
    match profile {
        DriverProfile::Constant { start, rate } => Ok(format!(
            "{} {} {}*time",
            number(*start),
            if *rate < 0.0 { "-" } else { "+" },
            number(rate.abs())
        )),
        DriverProfile::Harmonic {
            offset,
            amplitude,
            frequency_hz,
            phase,
        } => Ok(format!(
            "{} + {}*sin({}*time {} {})",
            number(*offset),
            number(*amplitude),
            number(std::f64::consts::TAU * frequency_hz),
            if *phase < 0.0 { "-" } else { "+" },
            number(phase.abs())
        )),
        _ => Err(unsupported(
            "only constant-rate and harmonic drivers have an .asmt expression",
        )),
    }
}

fn mass_marker(w: &mut String, level: usize, body: &Body) -> Result<(), MbdError> {
    let (mass, moments) = if body.grounded {
        (1.0, DVec3::ONE)
    } else {
        let i = body.inertia_com;
        let off = [
            i.x_axis.y, i.x_axis.z, i.y_axis.x, i.y_axis.z, i.z_axis.x, i.z_axis.y,
        ];
        let scale = i.x_axis.x.abs().max(i.y_axis.y.abs()).max(i.z_axis.z.abs());
        if off.iter().any(|v| v.abs() > 1e-12 * scale.max(1e-300)) {
            return Err(unsupported(
                "body inertia must be diagonal in the body axes for the .asmt export",
            ));
        }
        (body.mass_kg, DVec3::new(i.x_axis.x, i.y_axis.y, i.z_axis.z))
    };
    line(w, level, "PrincipalMassMarker");
    line(w, level + 1, "Name");
    line(w, level + 2, "MassMarker");
    frame(w, level + 1, body.com, DQuat::IDENTITY);
    line(w, level + 1, "Mass");
    line(w, level + 2, &number(mass));
    line(w, level + 1, "MomentOfInertias");
    line(w, level + 2, &vector(moments));
    line(w, level + 1, "Density");
    line(w, level + 2, "1");
    Ok(())
}

fn item_ij(w: &mut String, name: &str, marker_i: &str, marker_j: &str) {
    line(w, 4, "Name");
    line(w, 5, name);
    line(w, 4, "MarkerI");
    line(w, 5, marker_i);
    line(w, 4, "MarkerJ");
    line(w, 5, marker_j);
}

/// A reference point at the container origin carrying one marker.
fn ref_point(w: &mut String, level: usize, name: &str, t: DVec3, r: DQuat) {
    line(w, level, "RefPoint");
    frame(w, level + 1, DVec3::ZERO, DQuat::IDENTITY);
    line(w, level + 1, "Markers");
    line(w, level + 2, "Marker");
    line(w, level + 3, "Name");
    line(w, level + 4, name);
    frame(w, level + 3, t, r);
}

fn frame(w: &mut String, level: usize, t: DVec3, r: DQuat) {
    line(w, level, "Position3D");
    line(w, level + 1, &vector(t));
    line(w, level, "RotationMatrix");
    // Row-major: row i holds the world components (i) of the three axes.
    let m = DMat3::from_quat(r.normalize());
    for row in [m.row(0), m.row(1), m.row(2)] {
        line(w, level + 1, &vector(row));
    }
}

fn rates(w: &mut String, level: usize) {
    line(w, level, "Velocity3D");
    line(w, level + 1, &vector(DVec3::ZERO));
    line(w, level, "Omega3D");
    line(w, level + 1, &vector(DVec3::ZERO));
}

fn line(w: &mut String, level: usize, text: &str) {
    for _ in 0..level {
        w.push('\t');
    }
    w.push_str(text);
    w.push('\n');
}

/// Shortest round-trip decimal (Rust's `Display` never uses exponents).
fn number(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else {
        format!("{v}")
    }
}

fn vector(v: DVec3) -> String {
    let mut s = String::new();
    for c in [v.x, v.y, v.z] {
        s.push_str(&number(c));
        s.push('\t');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{BodyId, Marker, MarkerId};
    use crate::driver::Driver;
    use crate::joint::{Joint, JointId};

    fn hinge() -> System {
        let mut sys = System::new("hinge");
        for (id, grounded) in [(1, true), (2, false)] {
            sys.add_body(Body {
                id: BodyId(id),
                name: format!("b{id}"),
                mass_kg: if grounded { 0.0 } else { 2.0 },
                com: DVec3::new(0.5, 0.0, 0.0),
                inertia_com: DMat3::from_diagonal(DVec3::new(0.1, 0.2, 0.3)),
                translation: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
                grounded,
                velocity: DVec3::ZERO,
                angular_velocity: DVec3::ZERO,
            })
            .unwrap();
            sys.add_marker(Marker {
                id: MarkerId(id),
                name: format!("m{id}"),
                body: BodyId(id),
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
            profile: DriverProfile::Constant {
                start: 0.25,
                rate: -2.0,
            },
        });
        sys
    }

    /// The export carries the section skeleton OndselSolver's parser walks,
    /// the grounding fixed joint, the joint and the motion expression.
    #[test]
    fn hinge_exports_the_expected_sections() {
        let sim = AsmtSimulation {
            tstart: 0.0,
            tend: 1.0,
            hout: 0.1,
        };
        let text = write_asmt(&hinge(), &sim).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(&lines[..2], &["OndselSolver", "Assembly"]);
        for section in [
            "\tRefPoints",
            "\tParts",
            "\tKinematicIJs",
            "\tConstraintSets",
            "\t\tJoints",
            "\t\tMotions",
            "\t\tGeneralConstraintSets",
            "\tForceTorques",
            "\tConstantGravity",
            "\tSimulationParameters",
            "\tAnimationParameters",
        ] {
            assert!(lines.contains(&section), "missing {section:?}");
        }
        assert!(text.contains("\t\t\tFixedJoint\n\t\t\t\tName\n\t\t\t\t\tgrounded1\n"));
        assert!(text.contains("\t\t\tRevoluteJoint\n\t\t\t\tName\n\t\t\t\t\tj1\n\t\t\t\tMarkerI\n\t\t\t\t\t/Assembly/b1/m1\n\t\t\t\tMarkerJ\n\t\t\t\t\t/Assembly/b2/m2\n"));
        assert!(text.contains("\t\t\t\tRotationZ\n\t\t\t\t\t0.25 - 2*time\n"));
        assert!(text.contains("\t\tnframe\n\t\t\t11\n"));
        assert_eq!(text, write_asmt(&hinge(), &sim).unwrap(), "deterministic");
    }

    /// A joint driven in rotation and translation gives the rotation motion
    /// its own alias I marker (OndselSolver lets the later motion on a shared
    /// I marker overwrite the earlier one); the translation keeps the joint.
    #[test]
    fn doubly_driven_joint_aliases_the_rotation_marker() {
        let sim = AsmtSimulation {
            tstart: 0.0,
            tend: 1.0,
            hout: 0.1,
        };
        let mut sys = hinge();
        sys.joints.get_mut(&JointId(1)).unwrap().kind = JointKind::Cylindrical;
        sys.add_driver(Driver {
            joint: JointId(1),
            slot: DriverSlot::Translation,
            profile: DriverProfile::Constant {
                start: 0.0,
                rate: 0.5,
            },
        });
        let text = write_asmt(&sys, &sim).unwrap();
        assert!(
            text.contains("\t\t\t\t\t\t\t\tm1d0\n"),
            "alias marker on the ground part"
        );
        assert!(text.contains(
            "\t\t\tRotationalMotion\n\t\t\t\tName\n\t\t\t\t\td0\n\t\t\t\tMarkerI\n\t\t\t\t\t/Assembly/b1/m1d0\n\t\t\t\tMarkerJ\n\t\t\t\t\t/Assembly/b2/m2\n\t\t\t\tMotionJoint\n\t\t\t\t\t\n"
        ));
        assert!(text.contains(
            "\t\t\tTranslationalMotion\n\t\t\t\tName\n\t\t\t\t\td1\n\t\t\t\tMotionJoint\n\t\t\t\t\t/Assembly/j1\n"
        ));
        assert!(
            !write_asmt(&hinge(), &sim).unwrap().contains("m1d0"),
            "single driver: no alias"
        );
    }

    /// Profiles without an expression and non-diagonal inertia are refused.
    #[test]
    fn unsupported_inputs_are_refused() {
        let sim = AsmtSimulation {
            tstart: 0.0,
            tend: 1.0,
            hout: 0.1,
        };
        let mut sys = hinge();
        sys.drivers[0].profile = DriverProfile::Step5 {
            t0: 0.0,
            v0: 0.0,
            t1: 1.0,
            v1: 1.0,
        };
        assert_eq!(
            write_asmt(&sys, &sim).unwrap_err().code,
            codes::ASMT_UNSUPPORTED
        );
        let mut sys = hinge();
        sys.bodies.get_mut(&BodyId(2)).unwrap().inertia_com.x_axis.y = 0.01;
        assert_eq!(
            write_asmt(&sys, &sim).unwrap_err().code,
            codes::ASMT_UNSUPPORTED
        );
    }
}
