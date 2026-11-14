//! Rigid-group merging: bodies welded together by `Fixed` joints collapse to
//! one body before solving.
//!
//! CAD assemblies are dominated by parts that never move relative to each
//! other (fasteners, welded brackets, sub-assemblies marked rigid). In
//! maximal coordinates each such part costs 6 coordinates and a fixed joint
//! 6 equations; merging removes both, which is what the commercial tools do
//! (rigid sub-assemblies, Adams part merging). The merged system solves
//! like any other; [`MergedSystem::expand`] maps its frames back onto the
//! original bodies.
//!
//! Placement: a merged body's pose relative to its group root is the one
//! its fixed joints imply (`T_B = T_A · M1 · M2⁻¹`), not its authored pose,
//! so a slightly mis-authored weld is closed exactly. Mass properties are
//! combined (total mass, center of mass, inertia about it by the parallel
//! axis theorem) in the root's body axes.

use std::collections::BTreeMap;

use glam::{DMat3, DQuat, DVec3};

use crate::body::{Body, BodyId, MarkerId};
use crate::error::{codes, MbdError};
use crate::joint::{JointId, JointKind};
use crate::kinematics::{KinematicFrame, KinematicState, Twist};
use crate::system::System;

/// Tolerance for a redundant weld (a fixed joint closing a loop inside a
/// group, or a second grounded body) to count as consistent.
const WELD_TOL: f64 = 1e-9;

/// A rigid transform `(translation, rotation)`: `p ↦ t + q·p`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub translation: DVec3,
    pub rotation: DQuat,
}

impl Placement {
    const IDENTITY: Placement = Placement {
        translation: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
    };

    fn new(translation: DVec3, rotation: DQuat) -> Self {
        Self {
            translation,
            rotation: rotation.normalize(),
        }
    }

    fn then(self, o: Placement) -> Placement {
        Placement::new(
            self.translation + self.rotation * o.translation,
            self.rotation * o.rotation,
        )
    }

    fn inverse(self) -> Placement {
        let r = self.rotation.inverse();
        Placement::new(-(r * self.translation), r)
    }

    fn differs(self, o: Placement) -> bool {
        let turn = self.rotation.inverse() * o.rotation;
        let angle = 2.0 * turn.xyz().length().atan2(turn.w.abs());
        let scale = 1.0 + self.translation.length().max(o.translation.length());
        (self.translation - o.translation).length() > WELD_TOL * scale || angle > WELD_TOL
    }
}

/// A system with its fixed-joint groups collapsed, and the map back.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedSystem {
    /// The reduced system: one body per group (keeping the root's id), every
    /// marker re-parented onto its group's root, the group-internal fixed
    /// joints removed; drivers, couplers and force elements unchanged.
    pub system: System,
    /// Every original body's group root and its placement in the root's
    /// frame (`T_body = T_root · placement`). Roots map to themselves with
    /// the identity.
    pub placement: BTreeMap<BodyId, (BodyId, Placement)>,
    /// The fixed joints absorbed by merging.
    pub merged_joints: Vec<JointId>,
}

/// Collapse every group of bodies joined by (unsuppressed) fixed joints.
///
/// The root of a group is its grounded member with the smallest id, else its
/// smallest id. A fixed joint closing a loop inside a group, or a second
/// grounded body, must agree with the tree placement
/// ([`codes::INCONSISTENT_WELD`] otherwise).
pub fn merge_fixed_groups(sys: &System) -> Result<MergedSystem, MbdError> {
    sys.validate()?;
    let fixed: Vec<_> = sys
        .joints
        .values()
        .filter(|j| j.kind == JointKind::Fixed && !j.suppressed)
        .collect();
    let body_of = |m: MarkerId| sys.markers[&m].body;
    let local = |m: MarkerId| {
        let mk = &sys.markers[&m];
        Placement::new(mk.translation, mk.rotation)
    };

    // Adjacency over fixed joints, in id order (deterministic traversal).
    let mut edges: BTreeMap<BodyId, Vec<(JointId, BodyId)>> = BTreeMap::new();
    for j in &fixed {
        let (a, b) = (body_of(j.marker1), body_of(j.marker2));
        edges.entry(a).or_default().push((j.id, b));
        edges.entry(b).or_default().push((j.id, a));
    }

    // Groups by flood fill from each unvisited body in id order.
    let mut group_of: BTreeMap<BodyId, usize> = BTreeMap::new();
    let mut groups: Vec<Vec<BodyId>> = Vec::new();
    for &start in sys.bodies.keys() {
        if group_of.contains_key(&start) {
            continue;
        }
        let g = groups.len();
        let mut members = vec![start];
        group_of.insert(start, g);
        let mut k = 0;
        while k < members.len() {
            for &(_, next) in edges.get(&members[k]).map(Vec::as_slice).unwrap_or(&[]) {
                if let std::collections::btree_map::Entry::Vacant(slot) = group_of.entry(next) {
                    slot.insert(g);
                    members.push(next);
                }
            }
            k += 1;
        }
        members.sort();
        groups.push(members);
    }

    let mut placement: BTreeMap<BodyId, (BodyId, Placement)> = BTreeMap::new();
    let mut tree_joints = std::collections::BTreeSet::new();
    for members in &groups {
        let root = members
            .iter()
            .copied()
            .find(|b| sys.bodies[b].grounded)
            .unwrap_or(members[0]);
        placement.insert(root, (root, Placement::IDENTITY));
        // Breadth-first placement along fixed joints in id order.
        let mut queue = vec![root];
        let mut k = 0;
        while k < queue.len() {
            let here = queue[k];
            k += 1;
            let at = placement[&here].1;
            for &(jid, next) in edges.get(&here).map(Vec::as_slice).unwrap_or(&[]) {
                if placement.contains_key(&next) {
                    continue;
                }
                let j = &sys.joints[&jid];
                // world(m1) = world(m2): T_B = T_A · M1 · M2⁻¹.
                let (m_here, m_next) = if body_of(j.marker1) == here {
                    (local(j.marker1), local(j.marker2))
                } else {
                    (local(j.marker2), local(j.marker1))
                };
                placement.insert(next, (root, at.then(m_here).then(m_next.inverse())));
                tree_joints.insert(jid);
                queue.push(next);
            }
        }
        // Loop-closing welds and extra grounded bodies must agree.
        let root_pose = Placement::new(sys.bodies[&root].translation, sys.bodies[&root].rotation);
        for &b in members {
            let body = &sys.bodies[&b];
            if b != root && body.grounded {
                let implied = root_pose.then(placement[&b].1);
                if implied.differs(Placement::new(body.translation, body.rotation)) {
                    return Err(inconsistent(format!(
                        "grounded bodies {} and {} are welded at a pose that contradicts their authored poses",
                        root.0, b.0
                    )));
                }
            }
        }
    }
    for j in &fixed {
        if tree_joints.contains(&j.id) {
            continue;
        }
        let (a, b) = (body_of(j.marker1), body_of(j.marker2));
        let wa = placement[&a].1.then(local(j.marker1));
        let wb = placement[&b].1.then(local(j.marker2));
        if wa.differs(wb) {
            return Err(inconsistent(format!(
                "fixed joint {} ({}) closes a weld loop inconsistently",
                j.id.0, j.name
            )));
        }
    }

    // The reduced system.
    let mut out = System::new(&sys.name);
    out.gravity = sys.gravity;
    for members in &groups {
        let root = placement[&members[0]].0;
        let rb = &sys.bodies[&root];
        let (mass_kg, com, inertia_com) =
            combine(members.iter().map(|b| (&sys.bodies[b], placement[b].1)));
        out.bodies.insert(
            root,
            Body {
                id: root,
                name: rb.name.clone(),
                mass_kg,
                com,
                inertia_com,
                translation: rb.translation,
                rotation: rb.rotation,
                grounded: members.iter().any(|b| sys.bodies[b].grounded),
                velocity: rb.velocity,
                angular_velocity: rb.angular_velocity,
            },
        );
    }
    for (id, m) in &sys.markers {
        let (root, p) = placement[&m.body];
        let mut moved = m.clone();
        let world = p.then(Placement::new(m.translation, m.rotation));
        moved.body = root;
        moved.translation = world.translation;
        moved.rotation = world.rotation;
        out.markers.insert(*id, moved);
    }
    let merged_joints: Vec<JointId> = fixed.iter().map(|j| j.id).collect();
    for (id, j) in &sys.joints {
        if !merged_joints.contains(id) {
            out.joints.insert(*id, j.clone());
        }
    }
    out.drivers = sys.drivers.clone();
    out.forces = sys.forces.clone();
    out.couplers = sys.couplers.clone();
    Ok(MergedSystem {
        system: out,
        placement,
        merged_joints,
    })
}

fn inconsistent(detail: String) -> MbdError {
    MbdError::new(
        codes::INCONSISTENT_WELD,
        detail,
        "align the welded parts (or suppress the redundant fixed joint) before merging",
    )
}

/// Total mass, center of mass and inertia about it, in the root's axes.
fn combine<'a>(parts: impl Iterator<Item = (&'a Body, Placement)>) -> (f64, DVec3, DMat3) {
    let parts: Vec<_> = parts.collect();
    let mass: f64 = parts.iter().map(|(b, _)| b.mass_kg).sum();
    let com = if mass > 0.0 {
        parts
            .iter()
            .map(|(b, p)| (p.translation + p.rotation * b.com) * b.mass_kg)
            .sum::<DVec3>()
            / mass
    } else {
        DVec3::ZERO
    };
    let mut inertia = DMat3::ZERO;
    for (b, p) in &parts {
        let r = DMat3::from_quat(p.rotation);
        let d = p.translation + p.rotation * b.com - com;
        let shift = DMat3::from_diagonal(DVec3::splat(d.length_squared()))
            - DMat3::from_cols(d * d.x, d * d.y, d * d.z);
        inertia += r * b.inertia_com * r.transpose() + shift * b.mass_kg;
    }
    (mass, com, inertia)
}

impl MergedSystem {
    /// Map a frame solved on [`MergedSystem::system`] back onto the bodies of
    /// `original` (the system this was merged from): every merged body rides
    /// on its root (`v + ω×r`, `a + α×r + ω×(ω×r)`).
    pub fn expand(&self, original: &System, frame: &KinematicFrame) -> KinematicFrame {
        let mut state = KinematicState::from_system(original);
        let mut velocity = BTreeMap::new();
        let mut acceleration = BTreeMap::new();
        for (id, (root, p)) in &self.placement {
            let (t, r) = frame
                .state
                .body_pose(*root)
                .expect("root in the merged frame");
            let v = frame.velocity[root];
            let a = frame.acceleration[root];
            let arm = r * p.translation;
            let w_arm = v.angular.cross(arm);
            state.set_body_pose(*id, t + arm, r * p.rotation);
            velocity.insert(
                *id,
                Twist {
                    linear: v.linear + w_arm,
                    angular: v.angular,
                },
            );
            acceleration.insert(
                *id,
                Twist {
                    linear: a.linear + a.angular.cross(arm) + v.angular.cross(w_arm),
                    angular: a.angular,
                },
            );
        }
        state.copy_angles_from(&frame.state);
        KinematicFrame {
            time: frame.time,
            state,
            velocity,
            acceleration,
            redundant: frame.redundant.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Marker;
    use crate::joint::Joint;

    fn body(id: u64, mass: f64, at: DVec3, grounded: bool) -> Body {
        Body {
            id: BodyId(id),
            name: format!("b{id}"),
            mass_kg: mass,
            com: DVec3::ZERO,
            inertia_com: DMat3::from_diagonal(DVec3::splat(0.01 * mass)),
            translation: at,
            rotation: DQuat::IDENTITY,
            grounded,
            velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
        }
    }

    fn weld(sys: &mut System, id: u64, a: u64, at_a: DVec3, b: u64) {
        for (mid, body, at) in [(10 * id, a, at_a), (10 * id + 1, b, DVec3::ZERO)] {
            sys.add_marker(Marker {
                id: MarkerId(mid),
                name: format!("m{mid}"),
                body: BodyId(body),
                translation: at,
                rotation: DQuat::IDENTITY,
            })
            .unwrap();
        }
        sys.add_joint(Joint {
            id: JointId(id),
            name: format!("w{id}"),
            kind: JointKind::Fixed,
            marker1: MarkerId(10 * id),
            marker2: MarkerId(10 * id + 1),
            suppressed: false,
        })
        .unwrap();
    }

    /// Two 1 kg bodies welded 2 m apart merge into one 2 kg body with the
    /// center of mass midway and the parallel-axis inertia `2·m·1²` added
    /// about the transverse axes.
    #[test]
    fn mass_properties_combine() {
        let mut sys = System::new("pair");
        sys.add_body(body(1, 1.0, DVec3::ZERO, false)).unwrap();
        sys.add_body(body(2, 1.0, DVec3::new(2.0, 0.0, 0.0), false))
            .unwrap();
        weld(&mut sys, 1, 1, DVec3::new(2.0, 0.0, 0.0), 2);
        let merged = merge_fixed_groups(&sys).unwrap();
        assert_eq!(merged.system.bodies.len(), 1);
        assert_eq!(merged.merged_joints, vec![JointId(1)]);
        let b = &merged.system.bodies[&BodyId(1)];
        assert!((b.mass_kg - 2.0).abs() < 1e-15);
        assert!((b.com - DVec3::new(1.0, 0.0, 0.0)).length() < 1e-15);
        let want = DMat3::from_diagonal(DVec3::new(0.02, 0.02 + 2.0, 0.02 + 2.0));
        assert!(
            (b.inertia_com - want).abs_diff_eq(DMat3::ZERO, 1e-12),
            "{:?}",
            b.inertia_com
        );
        let (root, p) = merged.placement[&BodyId(2)];
        assert_eq!(root, BodyId(1));
        assert!((p.translation - DVec3::new(2.0, 0.0, 0.0)).length() < 1e-15);
    }

    /// A weld loop that disagrees with the tree, and a second grounded body
    /// welded off its authored pose, are refused.
    #[test]
    fn inconsistent_welds_are_refused() {
        let mut sys = System::new("loop");
        sys.add_body(body(1, 1.0, DVec3::ZERO, false)).unwrap();
        sys.add_body(body(2, 1.0, DVec3::X, false)).unwrap();
        weld(&mut sys, 1, 1, DVec3::X, 2);
        weld(&mut sys, 2, 1, DVec3::new(1.5, 0.0, 0.0), 2);
        assert_eq!(
            merge_fixed_groups(&sys).unwrap_err().code,
            codes::INCONSISTENT_WELD
        );
        sys.joints.get_mut(&JointId(2)).unwrap().suppressed = true;
        assert!(
            merge_fixed_groups(&sys).is_ok(),
            "a suppressed weld is ignored"
        );

        let mut grounded = System::new("grounds");
        grounded.add_body(body(1, 0.0, DVec3::ZERO, true)).unwrap();
        grounded
            .add_body(body(2, 0.0, DVec3::new(3.0, 0.0, 0.0), true))
            .unwrap();
        weld(&mut grounded, 1, 1, DVec3::X, 2);
        assert_eq!(
            merge_fixed_groups(&grounded).unwrap_err().code,
            codes::INCONSISTENT_WELD
        );
    }
}
