//! MuJoCo MJCF export — the Tier C cross-check (roadmap §6): the same
//! system run by an independent engine with a different formulation
//! (minimal coordinates, recursive dynamics) and a different integrator.
//! Tolerance-level sanity, not alignment.
//!
//! Supported subset (anything else is [`codes::MJCF_UNSUPPORTED`]):
//! - **topology**: grounded bodies merge into the world body; the joints
//!   must form a spanning tree from the ground (each free body reached
//!   once) — `Revolute` → `hinge`, `Prismatic` → `slide`, `Spherical` →
//!   `ball`, `Cylindrical` → `hinge` + `slide`, `Fixed` → the child nested
//!   rigidly; a free body without a tree joint gets a `freejoint`;
//!   further loop-closing `Revolute` / `Spherical` joints become
//!   `<equality connect>` (point coincidence; MuJoCo's equalities are soft,
//!   so loops agree to the constraint stiffness only);
//! - **initial state**: the position IC of the system; initial velocities
//!   only on free (freejoint) bodies, exported as the `qvel` of a keyframe;
//! - **forces**: gravity; `SpringDamper` (zero preload) as a spatial tendon;
//!   `JointSpringDamper` (zero preload) on a tree joint as the MuJoCo
//!   joint's `stiffness` / `damping` / `springref`;
//! - no drivers, couplers or other force elements. Contacts are disabled
//!   and the energy computation enabled.
//!
//! Bodies are named `b{id}`, so a MuJoCo `xpos` / `xquat` per body maps back
//! to this crate's body poses.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use glam::{DMat3, DQuat, DVec3};

use crate::body::{Body, BodyId};
use crate::driver::DriverSlot;
use crate::error::{codes, MbdError};
use crate::force::ForceElement;
use crate::joint::{Joint, JointId, JointKind};
use crate::kinematics::{position_ic, KinematicState};
use crate::system::System;

/// Export settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MjcfOptions {
    /// MuJoCo's fixed time step (s).
    pub timestep: f64,
    /// MuJoCo integrator name (`"RK4"`, `"implicit"`, `"Euler"`).
    pub integrator: &'static str,
}

impl Default for MjcfOptions {
    fn default() -> Self {
        Self {
            timestep: 1e-4,
            integrator: "RK4",
        }
    }
}

fn unsupported(detail: impl Into<String>) -> MbdError {
    MbdError::new(
        codes::MJCF_UNSUPPORTED,
        detail,
        "the MJCF export covers tree mechanisms (plus revolute/spherical loop closures) without drivers, couplers or general force elements",
    )
}

fn v(x: DVec3) -> String {
    format!("{} {} {}", x.x, x.y, x.z)
}

/// MuJoCo quaternions are `w x y z`.
fn q(r: DQuat) -> String {
    let r = r.normalize();
    format!("{} {} {} {}", r.w, r.x, r.y, r.z)
}

/// The tree: per free body, the joint reaching it from its parent (or
/// none: a free-floating body), the parent body (`None` = world).
struct Tree {
    parent: BTreeMap<BodyId, (Option<BodyId>, Option<JointId>)>,
    loops: Vec<JointId>,
}

fn build_tree(sys: &System) -> Result<Tree, MbdError> {
    let body_of = |m| sys.markers[&m].body;
    let grounded = |b: BodyId| sys.bodies[&b].grounded;
    let mut parent: BTreeMap<BodyId, (Option<BodyId>, Option<JointId>)> = BTreeMap::new();
    let mut reached: BTreeSet<BodyId> = sys
        .bodies
        .values()
        .filter(|b| b.grounded)
        .map(|b| b.id)
        .collect();
    let mut used: BTreeSet<JointId> = BTreeSet::new();
    let active: Vec<&Joint> = sys.joints.values().filter(|j| !j.suppressed).collect();
    // Breadth-first from the ground, joints in id order (deterministic).
    loop {
        let mut grew = false;
        for j in &active {
            if used.contains(&j.id) {
                continue;
            }
            let (a, b) = (body_of(j.marker1), body_of(j.marker2));
            let (from, to) = match (reached.contains(&a), reached.contains(&b)) {
                (true, false) => (a, b),
                (false, true) => (b, a),
                _ => continue,
            };
            used.insert(j.id);
            reached.insert(to);
            parent.insert(to, ((!grounded(from)).then_some(from), Some(j.id)));
            grew = true;
        }
        if !grew {
            break;
        }
    }
    for body in sys.bodies.values().filter(|b| !b.grounded) {
        if !reached.contains(&body.id) {
            if active
                .iter()
                .any(|j| body_of(j.marker1) == body.id || body_of(j.marker2) == body.id)
            {
                return Err(unsupported(format!(
                    "body {} is jointed but not connected to the ground (sub-mechanisms floating free are not exported)",
                    body.id.0
                )));
            }
            parent.insert(body.id, (None, None));
        }
    }
    let loops: Vec<JointId> = active
        .iter()
        .map(|j| j.id)
        .filter(|id| !used.contains(id))
        .collect();
    for id in &loops {
        let j = &sys.joints[id];
        if !matches!(j.kind, JointKind::Revolute | JointKind::Spherical) {
            return Err(unsupported(format!(
                "loop-closing joint {} is {:?}; only revolute / spherical loop closures are exported (as connect equalities)",
                id.0, j.kind
            )));
        }
    }
    Ok(Tree { parent, loops })
}

/// Export `sys` as MJCF text.
pub fn write_mjcf(sys: &System, options: &MjcfOptions) -> Result<String, MbdError> {
    sys.validate()?;
    if !sys.drivers.is_empty() {
        return Err(unsupported(
            "drivers are not exported (the cross-check is forward dynamics)",
        ));
    }
    if !sys.couplers.is_empty() {
        return Err(unsupported("couplers are not exported"));
    }
    let tree = build_tree(sys)?;
    let state = position_ic(sys, 0.0)?;
    let pose = |b: BodyId| state.body_pose(b).expect("own body");
    // Joint springs by tree joint.
    let mut joint_spring: BTreeMap<(JointId, DriverSlot), (f64, f64, f64)> = BTreeMap::new();
    let mut tendons = Vec::new();
    for f in sys.forces.values() {
        match f {
            ForceElement::JointSpringDamper {
                joint,
                slot,
                stiffness,
                damping,
                free_value,
                preload,
                ..
            } if *preload == 0.0 && !tree.loops.contains(joint) => {
                joint_spring.insert((*joint, *slot), (*stiffness, *damping, *free_value));
            }
            ForceElement::SpringDamper {
                id,
                marker1,
                marker2,
                stiffness,
                damping,
                free_length,
                preload,
                ..
            } if *preload == 0.0 => {
                tendons.push((*id, *marker1, *marker2, *stiffness, *damping, *free_length))
            }
            other => {
                return Err(unsupported(format!(
                    "force element {} ({}) has no MJCF counterpart in this export",
                    other.id().0,
                    other.name()
                )))
            }
        }
    }
    for b in sys.bodies.values() {
        if !b.grounded
            && tree.parent.get(&b.id).is_some_and(|p| p.1.is_some())
            && (b.velocity != DVec3::ZERO || b.angular_velocity != DVec3::ZERO)
        {
            return Err(unsupported(format!(
                "body {} has an initial velocity but is jointed; only free bodies carry initial velocities in the export",
                b.id.0
            )));
        }
    }

    let mut out = String::new();
    let w = &mut out;
    let _ = writeln!(
        w,
        "<mujoco model=\"{}\">",
        sys.name.replace(['"', '<', '>', '&'], "")
    );
    let _ = writeln!(
        w,
        "  <compiler angle=\"radian\" inertiafromgeom=\"false\" autolimits=\"true\"/>"
    );
    let g = sys.gravity.unwrap_or(DVec3::ZERO);
    let _ = writeln!(
        w,
        "  <option timestep=\"{}\" integrator=\"{}\" gravity=\"{}\">",
        options.timestep,
        options.integrator,
        v(g)
    );
    let _ = writeln!(w, "    <flag contact=\"disable\" energy=\"enable\"/>");
    let _ = writeln!(w, "  </option>");
    let _ = writeln!(w, "  <worldbody>");
    // Sites for tendons, by body.
    let mut sites: BTreeMap<BodyId, Vec<(String, DVec3)>> = BTreeMap::new();
    for (id, m1, m2, ..) in &tendons {
        for (k, m) in [m1, m2].into_iter().enumerate() {
            let marker = &sys.markers[m];
            sites
                .entry(marker.body)
                .or_default()
                .push((format!("s{}_{}", id.0, k), marker.translation));
        }
    }
    // World sites: grounded bodies' sites in world coordinates.
    for (body, list) in &sites {
        if sys.bodies[body].grounded {
            let (t, r) = pose(*body);
            for (name, local) in list {
                let _ = writeln!(
                    w,
                    "    <site name=\"{name}\" pos=\"{}\"/>",
                    v(t + r * *local)
                );
            }
        }
    }
    let children = |p: Option<BodyId>| -> Vec<BodyId> {
        tree.parent
            .iter()
            .filter(|(_, (par, _))| *par == p)
            .map(|(c, _)| *c)
            .collect()
    };
    let mut qvel: Vec<f64> = Vec::new();
    let mut has_free = false;
    // Depth-first emission.
    #[derive(Clone, Copy)]
    struct Ctx<'c> {
        sys: &'c System,
        tree: &'c Tree,
        state: &'c KinematicState,
        sites: &'c BTreeMap<BodyId, Vec<(String, DVec3)>>,
        joint_spring: &'c BTreeMap<(JointId, DriverSlot), (f64, f64, f64)>,
        children: &'c dyn Fn(Option<BodyId>) -> Vec<BodyId>,
    }
    fn emit(
        ctx: Ctx,
        w: &mut String,
        body: BodyId,
        depth: usize,
        qvel: &mut Vec<f64>,
        has_free: &mut bool,
    ) {
        let Ctx {
            sys,
            tree,
            state,
            sites,
            joint_spring,
            children,
        } = ctx;
        let indent = "  ".repeat(depth + 2);
        let b: &Body = &sys.bodies[&body];
        let (t, r) = state.body_pose(body).expect("own body");
        let (parent, joint) = tree.parent[&body];
        let (pt, pr) = match parent {
            Some(p) => state.body_pose(p).expect("own body"),
            None => (DVec3::ZERO, DQuat::IDENTITY),
        };
        let rel_t = pr.inverse() * (t - pt);
        let rel_r = pr.inverse() * r;
        let _ = writeln!(
            w,
            "{indent}<body name=\"b{}\" pos=\"{}\" quat=\"{}\">",
            body.0,
            v(rel_t),
            q(rel_r)
        );
        let i: DMat3 = b.inertia_com;
        let _ = writeln!(
            w,
            "{indent}  <inertial pos=\"{}\" mass=\"{}\" fullinertia=\"{} {} {} {} {} {}\"/>",
            v(b.com),
            b.mass_kg,
            i.x_axis.x,
            i.y_axis.y,
            i.z_axis.z,
            i.y_axis.x,
            i.z_axis.x,
            i.z_axis.y
        );
        match joint {
            None => {
                *has_free = true;
                let _ = writeln!(w, "{indent}  <freejoint name=\"free{}\"/>", body.0);
                // MuJoCo free-joint velocity: world linear velocity of the
                // body origin, angular velocity in the body frame.
                let wl = r.inverse() * b.angular_velocity;
                qvel.extend([b.velocity.x, b.velocity.y, b.velocity.z, wl.x, wl.y, wl.z]);
            }
            Some(jid) => {
                let j = &sys.joints[&jid];
                // The marker on this (child) body defines the joint frame.
                let m = if sys.markers[&j.marker2].body == body {
                    &sys.markers[&j.marker2]
                } else {
                    &sys.markers[&j.marker1]
                };
                let axis = m.rotation * DVec3::Z;
                let spring = |slot| {
                    joint_spring
                        .get(&(jid, slot))
                        .map(|(k, c, r0)| {
                            format!(" stiffness=\"{k}\" damping=\"{c}\" springref=\"{r0}\"")
                        })
                        .unwrap_or_default()
                };
                let pos = v(m.translation);
                let ax = v(axis);
                let joint_line = |w: &mut String,
                                  qvel: &mut Vec<f64>,
                                  kind: &str,
                                  slot: DriverSlot,
                                  suffix: &str| {
                    let extra = spring(slot);
                    let _ = writeln!(
                        w,
                        "{indent}  <joint name=\"j{}{suffix}\" type=\"{kind}\" pos=\"{pos}\" axis=\"{ax}\"{extra}/>",
                        jid.0
                    );
                    qvel.push(0.0);
                };
                match j.kind {
                    JointKind::Revolute => joint_line(w, qvel, "hinge", DriverSlot::Rotation, ""),
                    JointKind::Prismatic => {
                        joint_line(w, qvel, "slide", DriverSlot::Translation, "")
                    }
                    JointKind::Cylindrical => {
                        joint_line(w, qvel, "hinge", DriverSlot::Rotation, "r");
                        joint_line(w, qvel, "slide", DriverSlot::Translation, "t");
                    }
                    JointKind::Spherical => {
                        let _ = writeln!(
                            w,
                            "{indent}  <joint name=\"j{}\" type=\"ball\" pos=\"{pos}\"/>",
                            jid.0
                        );
                        qvel.extend([0.0; 3]);
                    }
                    JointKind::Fixed => {}
                    _ => unreachable!("checked before emission"),
                }
            }
        }
        if let Some(list) = sites.get(&body) {
            for (name, local) in list {
                let _ = writeln!(w, "{indent}  <site name=\"{name}\" pos=\"{}\"/>", v(*local));
            }
        }
        for c in children(Some(body)) {
            emit(ctx, w, c, depth + 1, qvel, has_free);
        }
        let _ = writeln!(w, "{indent}</body>");
    }
    for (_, joint) in tree.parent.values() {
        if let Some(j) = joint {
            let kind = sys.joints[j].kind;
            if !matches!(
                kind,
                JointKind::Revolute
                    | JointKind::Prismatic
                    | JointKind::Cylindrical
                    | JointKind::Spherical
                    | JointKind::Fixed
            ) {
                return Err(unsupported(format!(
                    "tree joint {} is {kind:?}, which has no MJCF joint type here",
                    j.0
                )));
            }
        }
    }
    let ctx = Ctx {
        sys,
        tree: &tree,
        state: &state,
        sites: &sites,
        joint_spring: &joint_spring,
        children: &children,
    };
    for root in children(None) {
        emit(ctx, w, root, 0, &mut qvel, &mut has_free);
    }
    let _ = writeln!(w, "  </worldbody>");
    if !tree.loops.is_empty() {
        let _ = writeln!(w, "  <equality>");
        for id in &tree.loops {
            let j = &sys.joints[id];
            let m1 = &sys.markers[&j.marker1];
            let m2 = &sys.markers[&j.marker2];
            let name = |b: BodyId| {
                if sys.bodies[&b].grounded {
                    "world".to_string()
                } else {
                    format!("b{}", b.0)
                }
            };
            // The anchor in body1's frame (world coordinates for the world).
            let anchor = if sys.bodies[&m1.body].grounded {
                let (t, r) = pose(m1.body);
                t + r * m1.translation
            } else {
                m1.translation
            };
            let _ = writeln!(
                w,
                "    <connect name=\"loop{}\" body1=\"{}\" body2=\"{}\" anchor=\"{}\" solref=\"0.0005 1\" solimp=\"0.9999 0.9999 0.0001\"/>",
                id.0,
                name(m1.body),
                name(m2.body),
                v(anchor)
            );
        }
        let _ = writeln!(w, "  </equality>");
    }
    if !tendons.is_empty() {
        let _ = writeln!(w, "  <tendon>");
        for (id, _, _, k, c, l0) in &tendons {
            let _ = writeln!(
                w,
                "    <spatial name=\"t{}\" stiffness=\"{k}\" damping=\"{c}\" springlength=\"{l0}\">",
                id.0
            );
            let _ = writeln!(w, "      <site site=\"s{}_0\"/>", id.0);
            let _ = writeln!(w, "      <site site=\"s{}_1\"/>", id.0);
            let _ = writeln!(w, "    </spatial>");
        }
        let _ = writeln!(w, "  </tendon>");
    }
    if has_free && qvel.iter().any(|x| *x != 0.0) {
        let qv: Vec<String> = qvel.iter().map(|x| x.to_string()).collect();
        let _ = writeln!(w, "  <keyframe>");
        let _ = writeln!(w, "    <key name=\"initial\" qvel=\"{}\"/>", qv.join(" "));
        let _ = writeln!(w, "  </keyframe>");
    }
    let _ = writeln!(w, "</mujoco>");
    Ok(out)
}
