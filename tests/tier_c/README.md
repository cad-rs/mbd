# Tier C: MuJoCo cross-check

`tests/tier_c.rs` runs `simulate` on the cases in `cases.rs` and compares
body poses and energy against MuJoCo, an independent engine with a
different formulation (minimal coordinates, recursive dynamics) and
integrator (RK4, fixed 0.1 ms step). The systems reach MuJoCo through this
crate's MJCF export (`write_mjcf`). It is a tolerance-level sanity check,
not an alignment (roadmap §6): loops are closed in MuJoCo by soft `connect`
equalities, so the four-bar agrees to the equality stiffness only.

The references are the committed `<case>.json` files; `cargo test` never
needs Python or MuJoCo.

## Regenerating

Python via [uv](https://docs.astral.sh/uv/) (no global install needed):

```sh
cargo run --example tier_c -- export <dir>
uv run --with mujoco python tests/tier_c/run_mujoco.py <dir> tests/tier_c
cargo test --test tier_c -- --nocapture   # prints the worst deviations
```

## What is compared

Per body `b{id}` and output time: the body-frame origin (`xpos`) and
orientation (`xquat`, `w x y z`), world frame; per output time MuJoCo's
potential + kinetic energy (gravity `−m·g·x_com`, joint and tendon springs
`½·k·(q − q₀)²`) against `energy()`. The worst deviations and the MuJoCo
version are printed by the test.

## The friction case

`rubbing_pendulum` is the one case whose two engines model the same
physics differently: this crate's joint friction is a regularized Stribeck
curve of the slip, MuJoCo's `frictionloss` is a Coulomb constraint its
solver resolves. Only load-independent friction (every arm zero) maps onto
`frictionloss` at all, and the comparison checks the torque's size and
sign through a reversal — not the regularization: over one second the
friction takes 0.3 J out of the pendulum in both engines and the
trajectories part by 3.6e-3 rad (5.4e-4 J). The friction model itself is
checked against closed forms in `tests/friction_benchmarks.rs`, which is
where its accuracy claims live. Its case limits in `cases.rs` are set
from the observed deviation, not from integration error.

## Observed

MuJoCo 4.0.1 (RK4, 0.1 ms) against this crate at a 1e-10 local tolerance:
worst ≤ 2.5e-6 m and ≤ 8.9e-5 rad (the slender spatial-chain link spinning
about its own axis; everything else ≤ 4.5e-6 rad), energy ≤ 1.4e-6 J. MuJoCo's
RK4 holds the energy to 1e-14, so the remaining difference is this crate's
integration error at 1e-10 — the practical floor of its index-3 scheme
(tighter tolerances hit rounding, `ε/h` in the velocities). The limits in
`cases.rs` keep ≥ 4x headroom.
