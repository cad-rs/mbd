//! Scaling benchmark (performance channel, roadmap §3 "Costs"): an N-link
//! pendulum chain of revolutes under gravity, timed for the kinematic
//! Jacobian, the dynamic IC and one simulated second.
//!
//! ```text
//! cargo run --release --example scaling -- [N ...]
//! ```

use std::time::Instant;

use glam::{DMat3, DQuat, DVec3};
use rcad_mbd::*;

/// N links of length 0.2 m hanging in a line along −Y from a grounded
/// pivot, each hinged about world Z, released from a 30° tilt.
fn chain(n: usize) -> System {
    let mut sys = System::new("chain");
    sys.gravity = Some(DVec3::new(0.0, -STANDARD_GRAVITY, 0.0));
    let len = 0.2;
    let tilt = DQuat::from_rotation_z(0.5);
    let body = |id: u64, at: DVec3, grounded: bool| Body {
        id: BodyId(id),
        name: format!("b{id}"),
        mass_kg: if grounded { 0.0 } else { 0.5 },
        com: if grounded {
            DVec3::ZERO
        } else {
            DVec3::new(0.0, -len / 2.0, 0.0)
        },
        inertia_com: if grounded {
            DMat3::ZERO
        } else {
            DMat3::from_diagonal(DVec3::new(2e-3, 1e-4, 2e-3))
        },
        translation: at,
        rotation: if grounded { DQuat::IDENTITY } else { tilt },
        grounded,
        velocity: DVec3::ZERO,
        angular_velocity: DVec3::ZERO,
    };
    sys.add_body(body(0, DVec3::ZERO, true)).unwrap();
    let mut marker_id = 0;
    let mut add_marker = |sys: &mut System, body: u64, at: DVec3| {
        marker_id += 1;
        sys.add_marker(Marker {
            id: MarkerId(marker_id),
            name: format!("m{marker_id}"),
            body: BodyId(body),
            translation: at,
            rotation: DQuat::IDENTITY,
        })
        .unwrap();
        MarkerId(marker_id)
    };
    for i in 1..=n as u64 {
        let at = tilt * DVec3::new(0.0, -len * (i - 1) as f64, 0.0);
        sys.add_body(body(i, at, false)).unwrap();
        let parent_point = if i == 1 {
            DVec3::ZERO
        } else {
            DVec3::new(0.0, -len, 0.0)
        };
        let m1 = add_marker(&mut sys, i - 1, parent_point);
        let m2 = add_marker(&mut sys, i, DVec3::ZERO);
        sys.add_joint(Joint {
            id: JointId(i),
            name: format!("j{i}"),
            kind: JointKind::Revolute,
            marker1: m1,
            marker2: m2,
            suppressed: false,
        })
        .unwrap();
    }
    sys
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("body count"))
        .collect();
    let sizes = if sizes.is_empty() {
        vec![10, 25, 50, 100]
    } else {
        sizes
    };
    println!("bodies  ic[ms]  simulate 1 s[ms]  steps  newton  matrices  energy drift");
    for n in sizes {
        let sys = chain(n);
        let t0 = Instant::now();
        let ic = dynamic_ic(&sys, 0.0).unwrap();
        let ic_ms = t0.elapsed().as_secs_f64() * 1e3;
        let times: Vec<f64> = (0..=20).map(|k| k as f64 * 0.05).collect();
        let t1 = Instant::now();
        let sim = simulate(&sys, &times, &HhtSettings::default()).unwrap();
        let sim_ms = t1.elapsed().as_secs_f64() * 1e3;
        let e0 = energy(&sys, &ic.state, &ic.velocity).unwrap().total();
        let last = sim.frames.last().unwrap();
        let e1 = energy(&sys, &last.state, &last.velocity).unwrap().total();
        println!(
            "{n:6}  {ic_ms:6.1}  {sim_ms:16.1}  {:5}  {:6}  {:8}  {:.2e}",
            sim.stats.steps,
            sim.stats.newton_iterations,
            sim.stats.jacobians,
            ((e1 - e0) / e0).abs()
        );
    }
}
