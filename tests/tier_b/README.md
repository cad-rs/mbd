# Tier B: OndselSolver reference trajectories

`tests/tier_b.rs` compares `kinematic_sweep` against trajectories produced by
an independent OndselSolver build from the `.asmt` export of the same
systems (`cases.rs`). The references are the committed `<case>.json` files;
`cargo test` never needs OndselSolver. Regenerate them when a case changes.

## Regenerating

1. Get the solver (LGPL-2.1, not vendored here):

   ```sh
   git clone https://github.com/Ondsel-Development/OndselSolver.git <ondsel>
   ```

2. Build the runner in `ondsel_runner/` (it links the OndselSolver library,
   parses each `.asmt`, runs `runKINEMATIC` and writes the result with its
   time series). On Windows, from a VS 2022 "x64 Native Tools" prompt:

   ```sh
   cmake -S tests/tier_b/ondsel_runner -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DONDSEL_SRC=<ondsel>
   cmake --build <build>
   ```

3. Export, run, convert:

   ```sh
   cargo run --example tier_b -- export <dir>
   # one (in, out) pair per case in cases.rs::all()
   <build>/ondsel_run <dir>/offset_crank_slider.asmt    <dir>/offset_crank_slider.out.asmt \
                      <dir>/spatial_rsur.asmt           <dir>/spatial_rsur.out.asmt \
                      <dir>/nested_chain.asmt           <dir>/nested_chain.out.asmt \
                      <dir>/primitive_crank_slider.asmt <dir>/primitive_crank_slider.out.asmt \
                      <dir>/primitive_rsur.asmt         <dir>/primitive_rsur.out.asmt \
                      <dir>/gravity_rsur.asmt           <dir>/gravity_rsur.out.asmt \
                      <dir>/gravity_nested_chain.asmt   <dir>/gravity_nested_chain.out.asmt
   cargo run --example tier_b -- reference <dir> "OndselSolver <commit>"
   ```

4. `cargo test --test tier_b -- --nocapture` prints the worst deviation per
   case; tighten or justify the limits in `tests/tier_b.rs` accordingly.

The committed references come from OndselSolver `09d6175` (2025-01-21),
release build. Worst deviation from this crate: 3.8e-10 (spatial RSUR
angular acceleration); everything else ≤ 5e-11. Reactions: see below.
The gravity cases and the reaction series were added with M2 card 3 from
the same build (the earlier cases' part series reproduced byte for byte).

## OndselSolver quirks met on the way

- `ASMTAssembly::assemblyFromFile` pops the `Assembly` header inside an
  `assert()`, so a release (NDEBUG) build parses one line off and crashes —
  the runner reads the file itself.
- A motion turns its I marker into a time-dependent frame; two motions on
  one joint (cylindrical: rotation + translation) share it and the later
  one overwrites the earlier — the export gives the rotation motion its own
  alias I marker.

## What is compared

Per part (`b{id}`) and output time: the part-frame origin position, the
orientation (OndselSolver's Bryant x-y-z angles, rebuilt to a rotation), the
origin velocity, the angular velocity, the origin acceleration and the
angular acceleration — all world frame, which is also this crate's `Twist`
convention. Joint and driver conventions match one to one: both solvers use
the same direction-cosine joint equations and measure a rotation driver as
`atan2(xJ·yI, xJ·xI)` unwrapped across turns.

Per joint (`j{id}`) and motion (`d{index}`) and output time: the reaction
OndselSolver writes (`FXonI … TZonI` — force on marker I's part and torque
about marker I's origin, world frame; the motion's I marker is the alias at
the joint's marker 1) against this crate's `Reaction::wrench_on_i` from
`forward_dynamics` on the kinematic frame (inverse dynamics: the masses and
gravity are in the export). OndselSolver removes redundant equations, this
crate reports the minimum-norm multiplier split; the two differ only along
the reaction's `indeterminate` directions, so the determinate part is
compared. Worst deviation: 1.4e-10 N force (RSUR, peak 28 N with gravity),
1.7e-11 N·m torque; the redundant crank-sliders agree to 3.5e-11 N on
their determinate part.
