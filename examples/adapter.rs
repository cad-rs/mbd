//! A neutral sample adapter (M4, `docs/INTEGRATING.md`): how a consumer —
//! any CAD kernel or application — turns *its own* assembly description
//! into a [`System`], runs a [`Study`] and reads the tables.
//!
//! The "CAD side" here is a made-up but typical export: millimetres,
//! kilograms, kg·mm² inertia, parts placed in the assembly, mates given as a
//! world-frame origin and axis at the assembly pose, motors in rpm. Nothing
//! in it is specific to any kernel — replace it with your own model.
//!
//! ```text
//! cargo run --example adapter            # prints the results CSV
//! ```

#![allow(dead_code)]

use std::collections::BTreeMap;

use glam::{DMat3, DQuat, DVec3};
use rcad_mbd::*;

// ── the consumer's own model (stands in for a CAD kernel's export) ──────

/// A part as a CAD tool reports it (millimetre units).
pub struct Part {
    pub name: String,
    pub fixed: bool,
    pub mass_kg: f64,
    /// Center of mass in the part frame (mm).
    pub com_mm: DVec3,
    /// Inertia about the center of mass, part axes (kg·mm²).
    pub inertia_kg_mm2: DMat3,
    /// Part frame in the assembly (mm) at the assembly pose.
    pub position_mm: DVec3,
    pub rotation: DQuat,
}

/// The mate vocabulary of the made-up CAD side.
#[derive(Clone, Copy)]
pub enum MateKind {
    /// Concentric + coincident: rotation about the axis.
    Hinge,
    /// Sliding along the axis, no rotation.
    Slider,
    /// Point coincidence.
    Ball,
    /// Rigid.
    Weld,
}

/// A mate between two parts: a world-frame origin and axis at the
/// assembly pose (mm).
pub struct Mate {
    pub name: String,
    pub kind: MateKind,
    pub part_a: usize,
    pub part_b: usize,
    pub origin_mm: DVec3,
    pub axis: DVec3,
}

/// A motor on a mate.
pub enum Motor {
    /// Constant speed (rpm) on a hinge.
    Rotary { mate: usize, rpm: f64 },
    /// Constant speed (mm/s) on a slider.
    Linear { mate: usize, mm_per_s: f64 },
}

pub struct Assembly {
    pub parts: Vec<Part>,
    pub mates: Vec<Mate>,
    pub motors: Vec<Motor>,
    /// Gravity (mm/s²), if any.
    pub gravity_mm_s2: Option<DVec3>,
}

// ── the adapter ─────────────────────────────────────────────────────────

const MM: f64 = 1e-3;

/// Where each CAD entity went in the system (for reading results back).
pub struct Lookup {
    pub body_of_part: Vec<BodyId>,
    pub joint_of_mate: Vec<JointId>,
}

/// The CAD model as a `System`: SI units, one body per part, two markers
/// per mate (the mate frame, Z along the mate axis, expressed in each part's
/// frame), the joint the mate means, a driver per motor.
pub fn to_system(asm: &Assembly) -> Result<(System, Lookup), MbdError> {
    let mut sys = System::new("adapted assembly");
    sys.gravity = asm.gravity_mm_s2.map(|g| g * MM);
    let mut body_of_part = Vec::new();
    for (i, p) in asm.parts.iter().enumerate() {
        let id = BodyId(i as u64 + 1);
        sys.add_body(Body {
            id,
            name: p.name.clone(),
            mass_kg: if p.fixed { 0.0 } else { p.mass_kg },
            com: p.com_mm * MM,
            inertia_com: p.inertia_kg_mm2 * (MM * MM),
            translation: p.position_mm * MM,
            rotation: p.rotation.normalize(),
            grounded: p.fixed,
            velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
        })?;
        body_of_part.push(id);
    }
    let mut joint_of_mate = Vec::new();
    for (k, mate) in asm.mates.iter().enumerate() {
        // The mate frame in the world: origin, Z along the axis.
        let world = (
            mate.origin_mm * MM,
            DQuat::from_rotation_arc(DVec3::Z, mate.axis.normalize()),
        );
        let mut markers = [MarkerId(0); 2];
        for (side, part) in [mate.part_a, mate.part_b].into_iter().enumerate() {
            let p = &asm.parts[part];
            let (pt, pr) = (p.position_mm * MM, p.rotation.normalize());
            let inv = pr.inverse();
            let id = MarkerId(2 * k as u64 + side as u64 + 1);
            sys.add_marker(Marker {
                id,
                name: format!("{}:{}", mate.name, if side == 0 { "I" } else { "J" }),
                body: body_of_part[part],
                translation: inv * (world.0 - pt),
                rotation: inv * world.1,
            })?;
            markers[side] = id;
        }
        let id = JointId(k as u64 + 1);
        sys.add_joint(Joint {
            id,
            name: mate.name.clone(),
            kind: match mate.kind {
                MateKind::Hinge => JointKind::Revolute,
                MateKind::Slider => JointKind::Prismatic,
                MateKind::Ball => JointKind::Spherical,
                MateKind::Weld => JointKind::Fixed,
            },
            marker1: markers[0],
            marker2: markers[1],
            suppressed: false,
        })?;
        joint_of_mate.push(id);
    }
    for motor in &asm.motors {
        let (mate, slot, rate) = match motor {
            Motor::Rotary { mate, rpm } => (
                *mate,
                DriverSlot::Rotation,
                rpm * std::f64::consts::TAU / 60.0,
            ),
            Motor::Linear { mate, mm_per_s } => (*mate, DriverSlot::Translation, mm_per_s * MM),
        };
        sys.add_driver(Driver {
            joint: joint_of_mate[mate],
            slot,
            profile: DriverProfile::Constant { start: 0.0, rate },
        });
    }
    sys.validate()?;
    Ok((
        sys,
        Lookup {
            body_of_part,
            joint_of_mate,
        },
    ))
}

// ── a sample model: a crank-slider, as a CAD tool would export it ───────

/// Crank 50 mm, rod 200 mm, slider on world X; the crank motor at 60 rpm.
pub fn crank_slider() -> Assembly {
    let (r, l) = (50.0, 200.0);
    let block = |name: &str, mass: f64, at: DVec3, com: DVec3| Part {
        name: name.into(),
        fixed: false,
        mass_kg: mass,
        com_mm: com,
        inertia_kg_mm2: DMat3::from_diagonal(DVec3::new(120.0, 900.0, 950.0)),
        position_mm: at,
        rotation: DQuat::IDENTITY,
    };
    Assembly {
        parts: vec![
            Part {
                name: "frame".into(),
                fixed: true,
                mass_kg: 0.0,
                com_mm: DVec3::ZERO,
                inertia_kg_mm2: DMat3::ZERO,
                position_mm: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
            },
            block("crank", 0.4, DVec3::ZERO, DVec3::new(r / 2.0, 0.0, 0.0)),
            block(
                "rod",
                0.9,
                DVec3::new(r, 0.0, 0.0),
                DVec3::new(l / 2.0, 0.0, 0.0),
            ),
            block("slider", 1.5, DVec3::new(r + l, 0.0, 0.0), DVec3::ZERO),
        ],
        mates: vec![
            Mate {
                name: "crank pivot".into(),
                kind: MateKind::Hinge,
                part_a: 0,
                part_b: 1,
                origin_mm: DVec3::ZERO,
                axis: DVec3::Z,
            },
            Mate {
                name: "crank pin".into(),
                kind: MateKind::Hinge,
                part_a: 1,
                part_b: 2,
                origin_mm: DVec3::new(r, 0.0, 0.0),
                axis: DVec3::Z,
            },
            Mate {
                name: "wrist pin".into(),
                kind: MateKind::Hinge,
                part_a: 2,
                part_b: 3,
                origin_mm: DVec3::new(r + l, 0.0, 0.0),
                axis: DVec3::Z,
            },
            Mate {
                name: "guide".into(),
                kind: MateKind::Slider,
                part_a: 0,
                part_b: 3,
                origin_mm: DVec3::new(r + l, 0.0, 0.0),
                axis: DVec3::X,
            },
        ],
        motors: vec![Motor::Rotary { mate: 0, rpm: 60.0 }],
        gravity_mm_s2: Some(DVec3::new(0.0, -9806.65, 0.0)),
    }
}

/// Adapt, run a kinematic study with reactions and a slider measure.
pub fn run() -> Result<(System, Lookup, StudyResult), MbdError> {
    let (sys, lookup) = to_system(&crank_slider())?;
    let study = Study {
        name: "one revolution".into(),
        analysis: Analysis::Kinematic,
        output: TimeGrid::Uniform {
            start: 0.0,
            end: 1.0,
            step: 0.05,
        },
        measures: vec![MeasureRequest::JointCoordinate {
            name: "stroke".into(),
            joint: lookup.joint_of_mate[3],
            slot: DriverSlot::Translation,
        }],
        reactions: true,
    };
    let result = run_study(&sys, &study)?;
    Ok((sys, lookup, result))
}

fn main() {
    match run() {
        Ok((_, lookup, result)) => {
            let names: BTreeMap<_, _> = lookup
                .joint_of_mate
                .iter()
                .enumerate()
                .map(|(i, j)| (*j, i))
                .collect();
            eprintln!(
                "{} frames, {} redundant equations; joints: {:?}",
                result.frames.len(),
                result.redundant.len(),
                names
            );
            print!("{}", result.to_csv());
        }
        Err(e) => {
            eprintln!("{e} — {}", e.hint);
            std::process::exit(1);
        }
    }
}
