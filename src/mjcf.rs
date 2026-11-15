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
//!   and the energy computation enabled; RK4 at a 0.1 ms step (the Tier C
//!   references are pinned to it).
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

/// MuJoCo's fixed step and integrator.
const TIMESTEP: f64 = 1e-4;
const INTEGRATOR: &str = "RK4";

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

/// The tree: per free body, its parent body (`None` = world) and the joint
/// reaching it (`None`: a free-floating body); the children per parent; the
/// loop-closing joints.
struct Tree {
    parent: BTreeMap<BodyId, (Option<BodyId>, Option<JointId>)>,
    children: BTreeMap<Option<BodyId>, Vec<BodyId>>,
    loops: Vec<JointId>,
}

impl Tree {
    fn children_of(&self, parent: Option<BodyId>) -> Vec<BodyId> {
        self.children.get(&parent).cloned().unwrap_or_default()
    }
}

fn build_tree(sys: &System) -> Result<Tree, MbdError> {
    let body_of = |m| sys.markers[&m].body;
    let grounded = |b: BodyId| sys.bodies[&b].grounded;
    let mut parent = BTreeMap::new();
    let mut reached: BTreeSet<BodyId> = sys
        .bodies
        .values()
        .filter(|b| b.grounded)
        .map(|b| b.id)
        .collect();
    let mut used: BTreeSet<JointId> = BTreeSet::new();
    let active: Vec<&Joint> = sys.joints.values().filter(|j| !j.suppressed).collect();
    // Repeated sweeps over the joints in id order until nothing new is
    // reached (deterministic).
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
            if !matches!(
                j.kind,
                JointKind::Revolute
                    | JointKind::Prismatic
                    | JointKind::Cylindrical
                    | JointKind::Spherical
                    | JointKind::Fixed
            ) {
                return Err(unsupported(format!(
                    "tree joint {} is {:?}, which has no MJCF joint type here",
                    j.id.0, j.kind
                )));
            }
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
    let mut children: BTreeMap<Option<BodyId>, Vec<BodyId>> = BTreeMap::new();
    for (child, (p, _)) in &parent {
        children.entry(*p).or_default().push(*child);
    }
    Ok(Tree {
        parent,
        children,
        loops,
    })
}

/// The depth-first body writer.
struct Emitter<'a> {
    sys: &'a System,
    tree: &'a Tree,
    state: &'a KinematicState,
    /// Tendon sites per body (body-frame positions).
    sites: &'a BTreeMap<BodyId, Vec<(String, DVec3)>>,
    /// Joint springs by tree joint: stiffness, damping, reference.
    joint_spring: &'a BTreeMap<(JointId, DriverSlot), (f64, f64, f64)>,
    /// Constant Coulomb friction per tree joint coordinate (`frictionloss`).
    friction_loss: &'a BTreeMap<(JointId, DriverSlot), f64>,
    out: String,
    /// Initial `qvel`, in MuJoCo's dof order.
    qvel: Vec<f64>,
}

impl Emitter<'_> {
    fn body(&mut self, body: BodyId, depth: usize) {
        let indent = "  ".repeat(depth + 2);
        let b: &Body = &self.sys.bodies[&body];
        let (t, r) = self.state.body_pose(body).expect("own body");
        let (parent, joint) = self.tree.parent[&body];
        let (pt, pr) = match parent {
            Some(p) => self.state.body_pose(p).expect("own body"),
            None => (DVec3::ZERO, DQuat::IDENTITY),
        };
        let _ = writeln!(
            self.out,
            "{indent}<body name=\"b{}\" pos=\"{}\" quat=\"{}\">",
            body.0,
            v(pr.inverse() * (t - pt)),
            q(pr.inverse() * r)
        );
        let i: DMat3 = b.inertia_com;
        let _ = writeln!(
            self.out,
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
                let _ = writeln!(self.out, "{indent}  <freejoint name=\"free{}\"/>", body.0);
                // MuJoCo free-joint velocity: world linear velocity of the
                // body origin, angular velocity in the body frame.
                let wl = r.inverse() * b.angular_velocity;
                self.qvel
                    .extend([b.velocity.x, b.velocity.y, b.velocity.z, wl.x, wl.y, wl.z]);
            }
            Some(jid) => self.joint(&indent, body, jid),
        }
        if let Some(list) = self.sites.get(&body) {
            for (name, local) in list {
                let _ = writeln!(
                    self.out,
                    "{indent}  <site name=\"{name}\" pos=\"{}\"/>",
                    v(*local)
                );
            }
        }
        for child in self.tree.children_of(Some(body)) {
            self.body(child, depth + 1);
        }
        let _ = writeln!(self.out, "{indent}</body>");
    }

    /// The MuJoCo joint(s) of tree joint `jid` reaching `body`.
    fn joint(&mut self, indent: &str, body: BodyId, jid: JointId) {
        let j = &self.sys.joints[&jid];
        // The marker on this (child) body defines the joint frame.
        let m = if self.sys.markers[&j.marker2].body == body {
            &self.sys.markers[&j.marker2]
        } else {
            &self.sys.markers[&j.marker1]
        };
        let pos = v(m.translation);
        let axis = v(m.rotation * DVec3::Z);
        let mut line = |kind: &str, slot: DriverSlot, suffix: &str| {
            let spring = self
                .joint_spring
                .get(&(jid, slot))
                .map(|(k, c, r0)| format!(" stiffness=\"{k}\" damping=\"{c}\" springref=\"{r0}\""))
                .unwrap_or_default();
            let friction = self
                .friction_loss
                .get(&(jid, slot))
                .map(|loss| format!(" frictionloss=\"{loss}\""))
                .unwrap_or_default();
            let _ = writeln!(
                self.out,
                "{indent}  <joint name=\"j{}{suffix}\" type=\"{kind}\" pos=\"{pos}\" axis=\"{axis}\"{spring}{friction}/>",
                jid.0
            );
            self.qvel.push(0.0);
        };
        match j.kind {
            JointKind::Revolute => line("hinge", DriverSlot::Rotation, ""),
            JointKind::Prismatic => line("slide", DriverSlot::Translation, ""),
            JointKind::Cylindrical => {
                line("hinge", DriverSlot::Rotation, "r");
                line("slide", DriverSlot::Translation, "t");
            }
            JointKind::Spherical => {
                let friction = self
                    .friction_loss
                    .get(&(jid, DriverSlot::Rotation))
                    .map(|loss| format!(" frictionloss=\"{loss}\""))
                    .unwrap_or_default();
                let _ = writeln!(
                    self.out,
                    "{indent}  <joint name=\"j{}\" type=\"ball\" pos=\"{pos}\"{friction}/>",
                    jid.0
                );
                self.qvel.extend([0.0; 3]);
            }
            JointKind::Fixed => {}
            _ => unreachable!("build_tree admits only exportable tree joints"),
        }
    }
}

/// Export `sys` as MJCF text.
pub fn write_mjcf(sys: &System) -> Result<String, MbdError> {
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
    let mut joint_spring = BTreeMap::new();
    let mut friction_loss: BTreeMap<(JointId, DriverSlot), f64> = BTreeMap::new();
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
            // MuJoCo's `frictionloss` is a constant Coulomb loss on the
            // joint's coordinate, so only load-independent friction (every
            // arm zero: `μ_d·preload`) has a counterpart. Its own
            // regularization is the solver's, not this crate's Stribeck
            // curve, so the cross-check holds away from zero slip.
            ForceElement::JointFriction {
                joint,
                mu_dynamic,
                preload,
                ..
            } if !f.reads_reactions() && !tree.loops.contains(joint) => {
                let loss = mu_dynamic * preload;
                // The slip the constant friction acts on: with a zero pin
                // radius a cylindrical joint's turning rubs nothing, so
                // only its sliding carries the loss.
                let slot = match sys.joints[joint].kind {
                    JointKind::Prismatic | JointKind::Cylindrical => DriverSlot::Translation,
                    _ => DriverSlot::Rotation,
                };
                friction_loss.insert((*joint, slot), loss);
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
    for (id, (_, joint)) in &tree.parent {
        let b = &sys.bodies[id];
        if joint.is_some() && (b.velocity != DVec3::ZERO || b.angular_velocity != DVec3::ZERO) {
            return Err(unsupported(format!(
                "body {} has an initial velocity but is jointed; only free bodies carry initial velocities in the export",
                id.0
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
        "  <option timestep=\"{TIMESTEP}\" integrator=\"{INTEGRATOR}\" gravity=\"{}\">",
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
    let mut emitter = Emitter {
        sys,
        tree: &tree,
        state: &state,
        sites: &sites,
        joint_spring: &joint_spring,
        friction_loss: &friction_loss,
        out,
        qvel: Vec::new(),
    };
    for root in tree.children_of(None) {
        emitter.body(root, 0);
    }
    let Emitter { mut out, qvel, .. } = emitter;
    let w = &mut out;
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
    // Only free bodies contribute non-zero velocities.
    if qvel.iter().any(|x| *x != 0.0) {
        let qv: Vec<String> = qvel.iter().map(|x| x.to_string()).collect();
        let _ = writeln!(w, "  <keyframe>");
        let _ = writeln!(w, "    <key name=\"initial\" qvel=\"{}\"/>", qv.join(" "));
        let _ = writeln!(w, "  </keyframe>");
    }
    let _ = writeln!(w, "</mujoco>");
    Ok(out)
}
