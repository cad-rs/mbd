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
   <build>/ondsel_run <dir>/offset_crank_slider.asmt <dir>/offset_crank_slider.out.asmt \
                      <dir>/spatial_rsur.asmt        <dir>/spatial_rsur.out.asmt \
                      <dir>/nested_chain.asmt        <dir>/nested_chain.out.asmt
   cargo run --example tier_b -- reference <dir> "OndselSolver <commit>"
   ```

4. `cargo test --test tier_b -- --include-ignored` (drop the `#[ignore]` on
   `tier_b_matches_ondsel` once the JSON files are committed).

## What is compared

Per part (`b{id}`) and output time: the part-frame origin position, the
orientation (OndselSolver's Bryant x-y-z angles, rebuilt to a rotation), the
origin velocity, the angular velocity, the origin acceleration and the
angular acceleration — all world frame, which is also this crate's `Twist`
convention. Joint and driver conventions match one to one: both solvers use
the same direction-cosine joint equations and measure a rotation driver as
`atan2(xJ·yI, xJ·xI)` unwrapped across turns.
