//! Studies (M4, the kernel-neutral integration surface): an analysis
//! described as data, one entry point, one result type.
//!
//! A consumer — any CAD kernel or application — builds a [`System`] from its
//! own assembly model (its adapter), describes what it wants as a [`Study`]
//! (which analysis, at which output times, which measures) and calls
//! [`run_study`]. The [`StudyResult`] carries, per output time, every body's
//! pose / velocity / acceleration, the joint and driver reactions, the
//! requested measures and the energy / momentum audit; it serializes
//! (serde, stable field names) and flattens to a wide CSV table
//! ([`StudyResult::to_csv`]) — the "results table" of the commercial tools.
//!
//! The three analyses follow the commercial vocabulary:
//! - **kinematic** — a fully driven mechanism swept over the output times
//!   (with `reactions`, the inverse-dynamics loads and motor efforts on
//!   every frame, as SolidWorks Motion / Inventor report them);
//! - **dynamic** — the index-3 HHT-α simulation ([`simulate`]);
//! - **static** — static equilibrium at every output time (drivers held at
//!   their values: a quasi-static sweep, each solve seeded from the last).

use std::collections::BTreeMap;

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::body::{BodyId, MarkerId};
use crate::driver::DriverSlot;
use crate::dynamics::{energy, forward_dynamics, momentum, static_equilibrium, Energy, Momentum};
use crate::error::{codes, MbdError};
use crate::integrate::{simulate, HhtSettings, IntegrationStats, DEFAULT_ALPHA, DEFAULT_ERROR_TOL};
use crate::joint::JointId;
use crate::kinematics::{kinematic_sweep, EquationOwner, EquationRef, KinematicFrame, Twist};
use crate::measure::{JointMotion, MarkerMotion, PointMotion, RelativeMotion};
use crate::reaction::Reaction;
use crate::system::System;

/// Output-grid size cap (a guard against a mistyped step).
const MAX_OUTPUT_TIMES: usize = 10_000_000;

/// An analysis to run on a system, as data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Study {
    pub name: String,
    pub analysis: Analysis,
    /// The output times (the first is the start of a dynamic run).
    pub output: TimeGrid,
    /// Measures to report at every output time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub measures: Vec<MeasureRequest>,
    /// Report joint and driver reactions (kinematic: inverse dynamics).
    #[serde(default = "default_true")]
    pub reactions: bool,
}

fn default_true() -> bool {
    true
}

/// Which analysis a study runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Analysis {
    /// Kinematic sweep of a fully driven mechanism.
    Kinematic,
    /// Forward dynamics, index-3 HHT-α.
    Dynamic {
        #[serde(default)]
        integrator: Integrator,
    },
    /// Static (quasi-static) equilibrium at every output time.
    Static,
}

/// Integrator settings as data (the serializable face of [`HhtSettings`];
/// every field optional in a document, defaulting as [`HhtSettings`] does).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Integrator {
    pub alpha: f64,
    /// Initial step (variable step) or the step (fixed), s.
    pub step: f64,
    /// Local error tolerance (m / rad); `None` = fixed step.
    pub error_tol: Option<f64>,
    pub min_step: f64,
    /// `None` = unbounded (output times still bound the steps).
    pub max_step: Option<f64>,
    pub max_newton: usize,
    pub newton_tol: f64,
}

impl Default for Integrator {
    fn default() -> Self {
        let d = HhtSettings::default();
        Self {
            alpha: DEFAULT_ALPHA,
            step: d.step,
            error_tol: Some(DEFAULT_ERROR_TOL),
            min_step: d.min_step,
            max_step: None,
            max_newton: d.max_newton,
            newton_tol: d.newton_tol,
        }
    }
}

impl From<&Integrator> for HhtSettings {
    fn from(i: &Integrator) -> Self {
        HhtSettings {
            alpha: i.alpha,
            step: i.step,
            error_tol: i.error_tol,
            min_step: i.min_step,
            max_step: i.max_step.unwrap_or(f64::INFINITY),
            max_newton: i.max_newton,
            newton_tol: i.newton_tol,
        }
    }
}

/// Output times.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "grid", rename_all = "snake_case")]
pub enum TimeGrid {
    /// `start, start + step, …` up to `end` (inclusive, to rounding).
    Uniform { start: f64, end: f64, step: f64 },
    /// Explicit times (finite, non-decreasing).
    Times { times: Vec<f64> },
}

impl TimeGrid {
    /// The output times.
    pub fn times(&self) -> Result<Vec<f64>, MbdError> {
        let bad = |detail: String| {
            MbdError::new(
                codes::BAD_STUDY,
                detail,
                "give finite times, end ≥ start and a positive step",
            )
        };
        match self {
            TimeGrid::Uniform { start, end, step } => {
                if !(start.is_finite() && end.is_finite() && step.is_finite())
                    || *step <= 0.0
                    || end < start
                {
                    return Err(bad(format!(
                        "unusable uniform grid {start}..{end} by {step}"
                    )));
                }
                let n = ((end - start) / step + 1e-9).floor() as usize;
                if n >= MAX_OUTPUT_TIMES {
                    return Err(bad(format!(
                        "{n} output times exceed the cap {MAX_OUTPUT_TIMES}"
                    )));
                }
                Ok((0..=n).map(|k| start + k as f64 * step).collect())
            }
            TimeGrid::Times { times } => {
                if times.is_empty()
                    || times.iter().any(|t| !t.is_finite())
                    || times.windows(2).any(|w| w[1] < w[0])
                {
                    return Err(bad(
                        "the explicit times must be non-empty, finite and non-decreasing".into(),
                    ));
                }
                Ok(times.clone())
            }
        }
    }
}

/// A measure to report, with the label it carries in the results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "measure", rename_all = "snake_case")]
pub enum MeasureRequest {
    /// A body-fixed point (body-frame coordinates).
    Point {
        name: String,
        body: BodyId,
        local: DVec3,
    },
    /// A marker's frame and rates.
    Marker { name: String, marker: MarkerId },
    /// Marker `j` relative to marker `i`, expressed in `i`.
    Relative {
        name: String,
        i: MarkerId,
        j: MarkerId,
    },
    /// A joint coordinate and its rates.
    JointCoordinate {
        name: String,
        joint: JointId,
        slot: DriverSlot,
    },
}

impl MeasureRequest {
    pub fn name(&self) -> &str {
        match self {
            MeasureRequest::Point { name, .. }
            | MeasureRequest::Marker { name, .. }
            | MeasureRequest::Relative { name, .. }
            | MeasureRequest::JointCoordinate { name, .. } => name,
        }
    }

    fn check(&self, sys: &System) -> Result<(), MbdError> {
        let missing = |what: String| {
            MbdError::new(
                codes::BAD_STUDY,
                format!("measure {:?}: {what}", self.name()),
                "measure entities of the system the study runs on",
            )
        };
        match self {
            MeasureRequest::Point { body, local, .. } => {
                if !sys.bodies.contains_key(body) {
                    return Err(missing(format!("no body {}", body.0)));
                }
                if !local.is_finite() {
                    return Err(missing("non-finite point".into()));
                }
            }
            MeasureRequest::Marker { marker, .. } => {
                if !sys.markers.contains_key(marker) {
                    return Err(missing(format!("no marker {}", marker.0)));
                }
            }
            MeasureRequest::Relative { i, j, .. } => {
                for m in [i, j] {
                    if !sys.markers.contains_key(m) {
                        return Err(missing(format!("no marker {}", m.0)));
                    }
                }
            }
            MeasureRequest::JointCoordinate { joint, slot, .. } => match sys.joints.get(joint) {
                None => return Err(missing(format!("no joint {}", joint.0))),
                Some(j) if !j.kind.drives(*slot) => {
                    return Err(missing(format!(
                        "{slot:?} is not a coordinate of {:?} joint {}",
                        j.kind, joint.0
                    )))
                }
                Some(_) => {}
            },
        }
        Ok(())
    }

    fn evaluate(&self, sys: &System, frame: &KinematicFrame) -> Option<MeasureValue> {
        Some(match self {
            MeasureRequest::Point { name, body, local } => MeasureValue::Point {
                name: name.clone(),
                motion: frame.point(*body, *local)?,
            },
            MeasureRequest::Marker { name, marker } => MeasureValue::Marker {
                name: name.clone(),
                motion: frame.marker(sys, *marker)?,
            },
            MeasureRequest::Relative { name, i, j } => MeasureValue::Relative {
                name: name.clone(),
                motion: frame.relative(sys, *i, *j)?,
            },
            MeasureRequest::JointCoordinate { name, joint, slot } => {
                MeasureValue::JointCoordinate {
                    name: name.clone(),
                    motion: frame.joint_coordinate(sys, *joint, *slot)?,
                }
            }
        })
    }
}

/// A measured value at one output time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "measure", rename_all = "snake_case")]
pub enum MeasureValue {
    Point {
        name: String,
        motion: PointMotion,
    },
    Marker {
        name: String,
        motion: MarkerMotion,
    },
    Relative {
        name: String,
        motion: RelativeMotion,
    },
    JointCoordinate {
        name: String,
        motion: JointMotion,
    },
}

/// A body's state at one output time (world frame, body-frame origin).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyResult {
    pub position: DVec3,
    pub rotation: DQuat,
    pub velocity: Twist,
    pub acceleration: Twist,
}

/// A joint's or driver's load at one output time (see [`Reaction`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactionResult {
    pub owner: EquationOwner,
    pub force_on_i: DVec3,
    pub torque_on_i: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<f64>,
    /// Whether the model determines the load completely (no redundancy).
    pub unique: bool,
}

impl From<&Reaction> for ReactionResult {
    fn from(r: &Reaction) -> Self {
        Self {
            owner: r.owner,
            force_on_i: r.force_on_i,
            torque_on_i: r.torque_on_i,
            effort: r.effort,
            unique: r.is_unique(),
        }
    }
}

/// Everything reported at one output time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultFrame {
    pub time: f64,
    pub bodies: BTreeMap<BodyId, BodyResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reactions: Vec<ReactionResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub measures: Vec<MeasureValue>,
    pub energy: Energy,
    pub momentum: Momentum,
}

/// A study's results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyResult {
    pub study: String,
    pub analysis: Analysis,
    pub frames: Vec<ResultFrame>,
    /// Redundant constraint equations at the first frame.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redundant: Vec<EquationRef>,
    /// Integrator work (dynamic studies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<IntegrationStats>,
}

/// Run a study on a system.
pub fn run_study(sys: &System, study: &Study) -> Result<StudyResult, MbdError> {
    sys.validate()?;
    for m in &study.measures {
        m.check(sys)?;
    }
    let times = study.output.times()?;
    let mut stats = None;
    // (kinematic frame, reactions) per output time.
    let mut solved: Vec<(KinematicFrame, Vec<Reaction>)> = Vec::with_capacity(times.len());
    match &study.analysis {
        Analysis::Kinematic => {
            for frame in kinematic_sweep(sys, &times)? {
                let reactions = if study.reactions {
                    forward_dynamics(sys, frame.time, &frame.state, &frame.velocity)?
                        .reactions(sys)?
                } else {
                    Vec::new()
                };
                solved.push((frame, reactions));
            }
        }
        Analysis::Dynamic { integrator } => {
            let sim = simulate(sys, &times, &HhtSettings::from(integrator))?;
            stats = Some(sim.stats);
            for frame in sim.frames {
                let reactions = if study.reactions {
                    frame.reactions(sys)?
                } else {
                    Vec::new()
                };
                solved.push((
                    KinematicFrame {
                        time: frame.time,
                        state: frame.state,
                        velocity: frame.velocity,
                        acceleration: frame.acceleration,
                        redundant: frame.redundant,
                    },
                    reactions,
                ));
            }
        }
        Analysis::Static => {
            let mut seed = None;
            for &time in &times {
                let eq = static_equilibrium(sys, time, seed.as_ref())?;
                let reactions = if study.reactions {
                    eq.reactions(sys)?
                } else {
                    Vec::new()
                };
                let rest: BTreeMap<BodyId, Twist> = eq
                    .state
                    .order()
                    .iter()
                    .map(|id| (*id, Twist::default()))
                    .collect();
                seed = Some(eq.state.clone());
                solved.push((
                    KinematicFrame {
                        time,
                        state: eq.state,
                        velocity: rest.clone(),
                        acceleration: rest,
                        redundant: eq.redundant,
                    },
                    reactions,
                ));
            }
        }
    }
    let redundant = solved
        .first()
        .map(|(f, _)| f.redundant.clone())
        .unwrap_or_default();
    let mut frames = Vec::with_capacity(solved.len());
    for (frame, reactions) in &solved {
        let bodies = frame
            .state
            .order()
            .iter()
            .map(|id| {
                let (position, rotation) = frame.state.body_pose(*id).expect("own body");
                (
                    *id,
                    BodyResult {
                        position,
                        rotation,
                        velocity: frame.velocity[id],
                        acceleration: frame.acceleration[id],
                    },
                )
            })
            .collect();
        frames.push(ResultFrame {
            time: frame.time,
            bodies,
            reactions: reactions.iter().map(ReactionResult::from).collect(),
            measures: study
                .measures
                .iter()
                .filter_map(|m| m.evaluate(sys, frame))
                .collect(),
            energy: energy(sys, &frame.state, &frame.velocity)?,
            momentum: momentum(sys, &frame.state, &frame.velocity)?,
        });
    }
    Ok(StudyResult {
        study: study.name.clone(),
        analysis: study.analysis.clone(),
        frames,
        redundant,
        stats,
    })
}

fn owner_label(owner: &EquationOwner) -> String {
    match owner {
        EquationOwner::Joint(id) => format!("j{}", id.0),
        EquationOwner::Coupler(id) => format!("c{}", id.0),
        EquationOwner::Driver(i) => format!("d{i}"),
    }
}

fn v3(v: DVec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

impl StudyResult {
    /// The results as one wide CSV table: a row per output time; columns
    /// `time`, per body `b{id}.{x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz,ax,ay,az,alx,aly,alz}`,
    /// per reaction `j{id}` / `d{index}` `.{fx,fy,fz,tx,ty,tz}` (+ `.effort`
    /// for drivers), per measure its name with its components, then the
    /// energy (`kinetic`, `gravity`, `elastic`) and momentum
    /// (`px..pz`, `hx..hz`). World frame, SI; values in Rust's shortest
    /// round-trip decimal form.
    pub fn to_csv(&self) -> String {
        let mut header: Vec<String> = vec!["time".into()];
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(self.frames.len());
        for (k, frame) in self.frames.iter().enumerate() {
            let mut row = vec![frame.time];
            let mut push =
                |label: String, values: &[f64], names: &[&str], header: &mut Vec<String>| {
                    if k == 0 {
                        header.extend(names.iter().map(|n| format!("{label}.{n}")));
                    }
                    row.extend_from_slice(values);
                };
            for (id, b) in &frame.bodies {
                let mut v = Vec::with_capacity(19);
                v.extend(v3(b.position));
                v.extend([b.rotation.x, b.rotation.y, b.rotation.z, b.rotation.w]);
                v.extend(v3(b.velocity.linear));
                v.extend(v3(b.velocity.angular));
                v.extend(v3(b.acceleration.linear));
                v.extend(v3(b.acceleration.angular));
                push(
                    format!("b{}", id.0),
                    &v,
                    &[
                        "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz",
                        "ax", "ay", "az", "alx", "aly", "alz",
                    ],
                    &mut header,
                );
            }
            for r in &frame.reactions {
                let mut v: Vec<f64> = v3(r.force_on_i)
                    .into_iter()
                    .chain(v3(r.torque_on_i))
                    .collect();
                let mut names = vec!["fx", "fy", "fz", "tx", "ty", "tz"];
                if let Some(e) = r.effort {
                    v.push(e);
                    names.push("effort");
                }
                push(owner_label(&r.owner), &v, &names, &mut header);
            }
            for m in &frame.measures {
                let (name, v, names): (&str, Vec<f64>, Vec<&str>) = match m {
                    MeasureValue::Point { name, motion } => (
                        name,
                        [
                            v3(motion.position),
                            v3(motion.velocity),
                            v3(motion.acceleration),
                        ]
                        .concat(),
                        vec!["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"],
                    ),
                    MeasureValue::Marker { name, motion } => {
                        let q = motion.rotation;
                        let mut v = v3(motion.position).to_vec();
                        v.extend([q.x, q.y, q.z, q.w]);
                        v.extend(v3(motion.velocity.linear));
                        v.extend(v3(motion.velocity.angular));
                        (
                            name,
                            v,
                            vec![
                                "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx",
                                "wy", "wz",
                            ],
                        )
                    }
                    MeasureValue::Relative { name, motion } => (
                        name,
                        [
                            v3(motion.displacement),
                            v3(motion.velocity),
                            v3(motion.acceleration),
                            v3(motion.angular_velocity),
                        ]
                        .concat(),
                        vec![
                            "dx", "dy", "dz", "vx", "vy", "vz", "ax", "ay", "az", "wx", "wy", "wz",
                        ],
                    ),
                    MeasureValue::JointCoordinate { name, motion } => (
                        name,
                        vec![motion.value, motion.velocity, motion.acceleration],
                        vec!["q", "qd", "qdd"],
                    ),
                };
                push(name.to_string(), &v, &names, &mut header);
            }
            let e = frame.energy;
            let p = frame.momentum;
            push(
                "energy".into(),
                &[e.kinetic, e.gravity, e.elastic],
                &["kinetic", "gravity", "elastic"],
                &mut header,
            );
            push(
                "momentum".into(),
                &[v3(p.linear), v3(p.angular)].concat(),
                &["px", "py", "pz", "hx", "hy", "hz"],
                &mut header,
            );
            rows.push(row);
        }
        let mut out = header.join(",");
        out.push('\n');
        for row in rows {
            let cells: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
            out.push_str(&cells.join(","));
            out.push('\n');
        }
        out
    }
}
