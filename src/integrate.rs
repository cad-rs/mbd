//! Time integration (M2): the index-3 HHT-α method (Hilber–Hughes–Taylor,
//! the Adams `HHT` integrator) on the maximal-coordinate DAE.
//!
//! **Scheme.** HHT-α written as the generalized-α method with `α_m = 0`,
//! `α_f = −α`, in the form of Arnold & Brüls (2007): the equations of
//! motion hold exactly at `t + h` with the true accelerations `q̈`, and an
//! auxiliary acceleration `a` carries the HHT weighting into the Newmark
//! formulas (`β = (1 − α)²/4`, `γ = ½ − α`). Unknowns per step are `q̈` and
//! `λ` at `t + h`:
//!
//! ```text
//! a    = (1 + α)·q̈ − α·q̈ₙ
//! Δ    = h·vₙ + h²·((½ − β)·aₙ + β·a)          q = exp(Δ)·qₙ  (Lie-group increment)
//! v    = vₙ + h·((1 − γ)·aₙ + γ·a)
//! M(q)·q̈ + Jᵀλ + b(q, v) − Q(q, v, t) = 0
//! Φ(q, t + h) / ((1 + α)·β·h²) = 0
//! ```
//!
//! For constant mass matrices this is the classical HHT recursion; unlike
//! the classical form it stays second order with a configuration-dependent
//! mass matrix (a tumbling body's world inertia `R·I·Rᵀ`) — the classical
//! form drops to first order there, and its numerical dissipation then
//! reaches the low frequencies.
//!
//! Index 3: the position constraints are imposed directly — no Baumgarte
//! terms. The rotation increment is applied as `R ← exp(Δθ)·R` in the
//! world-frame tangent space of [`crate::Twist`] (second-order consistent
//! with `Ṙ = [ω]×·R`), so no quaternion constraint enters the step. The
//! constraint rows are scaled so their derivative is `J` (Negrut et al.
//! 2007): the iteration matrix `[∂r/∂q̈  Jᵀ; J  0]` then stays well
//! conditioned as `h → 0`. `∂r/∂q̈` (mass, plus the stiffness and damping of
//! the forces through `q(q̈)`, `v(q̈)`) is formed by forward differences and
//! reused across steps (modified Newton) until convergence slows.
//!
//! `α ∈ [−⅓, 0]`: `α = 0` is the trapezoidal rule (no numerical damping,
//! conservative), negative values damp the high-frequency content; the
//! default −0.3 is Adams'. The method is second-order accurate for every
//! α. On constrained systems keep α < 0: at α = 0 the index-3
//! constraint-force modes are undamped and the scheme is only weakly
//! stable there (a spinning two-body chain on a spherical joint blows up
//! after ~9 s at α = 0 and conserves to 1e-5 at α = −0.3).
//!
//! **Step control** (`error_tol`): the Newmark local error estimate
//! `(β − ⅙)·h²·(q̈ₙ₊₁ − q̈ₙ)` (Zienkiewicz–Xie) against an absolute
//! tolerance (m / rad); the step changes by `0.9·r^(−1/3)` (bounded to
//! `[0.2, 2]`), with hysteresis — an accepted step keeps its size unless it
//! may grow ≥ 1.5× or its error ratio exceeds 0.9 — no growth right after a
//! rejection, and the way to each output time spread evenly. Each step-size
//! change re-excites the (damped) index-3 λ oscillation that the estimate
//! sees as noise; these rules keep the controller from feeding it. Per-step
//! control of an O(h³) local error yields a global error ∝ tol^(2/3).
//!
//! **Output.** Index 3 satisfies the position constraints only; the
//! velocities drift `O(h²)` off the velocity constraints. Every output
//! frame is therefore projected: the velocity onto `J·v = ṡ` (kinetic
//! energy norm), then acceleration and multipliers re-solved on the
//! acceleration level. The integrator's own state is never modified by the
//! projection.

use crate::dynamics::{
    applied_forces, dynamic_ic, flatten, inertias, output_frame, retract, slot_twists, DynamicFrame,
};
use crate::error::{codes, MbdError};
use crate::kinematics::{KinematicState, Problem};
use crate::linalg::RowFactor;
use crate::system::System;

/// The Adams default HHT α.
pub const DEFAULT_ALPHA: f64 = -0.3;

/// Position-constraint tolerance of a converged step (m / rad).
const CONSTRAINT_TOL: f64 = 1.0e-10;

/// Relative forward-difference step of the iteration matrix.
const FD_STEP: f64 = 1.0e-6;

/// Stale-matrix Newton iterations before the iteration matrix is rebuilt.
const STALE_ITERATIONS: usize = 5;

/// Step controller: safety factor, and the largest growth / shrink per step.
const SAFETY: f64 = 0.9;
const MAX_GROWTH: f64 = 2.0;
const MAX_SHRINK: f64 = 0.2;
/// Accepted steps keep their size unless the ideal factor reaches
/// `GROW_AT`, or the error ratio exceeds `SHRINK_ABOVE` (then it shrinks).
const GROW_AT: f64 = 1.5;
const SHRINK_ABOVE: f64 = 0.9;

/// The Adams-equivalent default local error tolerance: 1e-6 m (Adams
/// `ERROR = 1e-3` in millimetre models) / rad.
pub const DEFAULT_ERROR_TOL: f64 = 1.0e-6;

/// Integrator settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HhtSettings {
    /// HHT α in `[−⅓, 0]` (0: trapezoidal, no numerical damping).
    pub alpha: f64,
    /// Initial step (s) with error control, the nominal step without.
    /// Steps are shortened to land on every output time and halved when
    /// Newton fails.
    pub step: f64,
    /// Local error tolerance per step (m / rad, max norm over the
    /// coordinates): `Some` = variable step, `None` = fixed nominal step.
    /// The estimate is the Newmark local error on the true accelerations
    /// `(β − ⅙)·h²·(q̈ₙ₊₁ − q̈ₙ)` (Zienkiewicz–Xie; the auxiliary `a` carries
    /// the damped constraint-force oscillation of index 3 and would make a
    /// noisy estimate).
    pub error_tol: Option<f64>,
    /// Step bounds (s) for the controller and for Newton-failure halving.
    pub min_step: f64,
    pub max_step: f64,
    /// Newton iteration cap per step.
    pub max_newton: usize,
    /// Newton tolerance on the velocity correction `γ·h·|δa|`, relative to
    /// `1 + |v|`.
    pub newton_tol: f64,
}

impl Default for HhtSettings {
    /// α = −0.3, variable step at [`DEFAULT_ERROR_TOL`] from 1 ms, steps
    /// in `[1e-9 s, ∞)` (output times still bound them).
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            step: 1.0e-3,
            error_tol: Some(DEFAULT_ERROR_TOL),
            min_step: 1.0e-9,
            max_step: f64::INFINITY,
            max_newton: 12,
            newton_tol: 1.0e-9,
        }
    }
}

impl HhtSettings {
    /// A fixed-step integrator (the conservation audits): step `h`, α.
    pub fn fixed(alpha: f64, h: f64) -> Self {
        Self {
            alpha,
            step: h,
            error_tol: None,
            min_step: h / 1024.0,
            ..Self::default()
        }
    }
}

/// Integrator work counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IntegrationStats {
    /// Accepted steps.
    pub steps: usize,
    /// Rejected step attempts (Newton failure or error test).
    pub rejected: usize,
    /// Newton iterations over all attempts.
    pub newton_iterations: usize,
    /// Iteration-matrix evaluations.
    pub jacobians: usize,
}

/// A simulated trajectory: one consistent frame per output time.
#[derive(Debug, Clone, PartialEq)]
pub struct Simulation {
    pub frames: Vec<DynamicFrame>,
    pub stats: IntegrationStats,
}

/// The integrator state at an accepted instant (tangent vectors flat over
/// the free bodies).
struct Point {
    time: f64,
    state: KinematicState,
    v: Vec<f64>,
    /// The true accelerations `q̈` (the equations of motion hold with them).
    acc: Vec<f64>,
    /// The auxiliary Newmark acceleration `a`.
    aux: Vec<f64>,
    lambda: Vec<f64>,
}

/// One residual evaluation of a step.
struct Eval {
    /// Dynamic rows, then the scaled constraint rows.
    r: Vec<f64>,
    phi: f64,
    params: Vec<f64>,
    v: Vec<f64>,
    j: Vec<Vec<f64>>,
}

fn inf_norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// One step attempt from `from` to `problem.time`.
struct Step<'s, 'a> {
    problem: &'s Problem<'a>,
    from: &'s Point,
    h: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

impl Step<'_, '_> {
    /// The auxiliary acceleration for trial accelerations `x`.
    fn aux(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.from.acc)
            .map(|(x, prev)| (1.0 + self.alpha) * x - self.alpha * prev)
            .collect()
    }

    /// Positions and velocities from the trial accelerations.
    fn advance(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (h, p) = (self.h, self.from);
        let a = self.aux(x);
        let delta: Vec<f64> = (0..a.len())
            .map(|i| h * p.v[i] + h * h * ((0.5 - self.beta) * p.aux[i] + self.beta * a[i]))
            .collect();
        let params = retract(self.problem, &p.state.params, &delta);
        let v = (0..a.len())
            .map(|i| p.v[i] + h * ((1.0 - self.gamma) * p.aux[i] + self.gamma * a[i]))
            .collect();
        (params, v)
    }

    /// `[M·q̈ + Jᵀλ + b − Q; Φ / ((1+α)·β·h²)]` at trial `(x, λ)`.
    fn eval(&self, x: &[f64], lambda: &[f64]) -> Result<Eval, MbdError> {
        let problem = self.problem;
        let (params, v) = self.advance(x);
        let vel = problem.twists(&v);
        let acc = problem.twists(x);
        let mut net = applied_forces(problem, &params, &vel)?;
        for ((slot, i), n) in inertias(problem, &params)
            .iter()
            .enumerate()
            .zip(net.iter_mut())
        {
            if let Some(i) = i {
                // Q − b − M·q̈
                let ma = i.apply(acc[slot]);
                let b = i.bias(vel[slot].angular);
                n.linear -= ma.linear + b.linear;
                n.angular -= ma.angular + b.angular;
            }
        }
        let j = problem.jacobian(&params);
        let mut r: Vec<f64> = flatten(problem, &net).iter().map(|v| -v).collect();
        for (row, l) in j.iter().zip(lambda) {
            for (ri, ji) in r.iter_mut().zip(row) {
                *ri += ji * l;
            }
        }
        let mut phi = vec![0.0; problem.rows.len()];
        problem.constraint_residuals(&params, &mut phi);
        let scale = 1.0 / ((1.0 + self.alpha) * self.beta * self.h * self.h);
        r.extend(phi.iter().map(|p| p * scale));
        Ok(Eval {
            r,
            phi: inf_norm(&phi),
            params,
            v,
            j,
        })
    }

    /// The iteration matrix at `(x, λ)`: `[∂r/∂q̈ Jᵀ; J 0]`.
    fn matrix(&self, x: &[f64], lambda: &[f64], base: &Eval) -> Result<RowFactor, MbdError> {
        let n = x.len();
        let m = base.j.len();
        let mut rows = vec![vec![0.0; n + m]; n + m];
        let mut trial = x.to_vec();
        for c in 0..n {
            let eps = FD_STEP * (1.0 + x[c].abs());
            trial[c] = x[c] + eps;
            let e = self.eval(&trial, lambda)?;
            trial[c] = x[c];
            for (row, (ri, bi)) in rows.iter_mut().zip(e.r.iter().zip(&base.r)).take(n) {
                row[c] = (ri - bi) / eps;
            }
        }
        for (k, jr) in base.j.iter().enumerate() {
            for (i, row) in rows.iter_mut().enumerate().take(n) {
                row[n + k] = jr[i];
            }
            rows[n + k][..n].copy_from_slice(jr);
        }
        Ok(RowFactor::new(&rows, n + m))
    }
}

/// The iteration matrix reused across steps. With the `1/(β·h²)` scaling
/// its constraint blocks do not depend on `h`, only the (small) stiffness
/// and damping terms do, so it stays a good modified-Newton matrix across
/// step-size changes; slow convergence or a failed step rebuilds it.
struct Cache {
    factor: RowFactor,
}

/// Attempt one step; `None` when Newton does not converge.
fn take_step(
    problem: &Problem,
    from: &Point,
    h: f64,
    settings: &HhtSettings,
    cache: &mut Option<Cache>,
    stats: &mut IntegrationStats,
) -> Result<Option<Point>, MbdError> {
    let alpha = settings.alpha;
    let step = Step {
        problem,
        from,
        h,
        alpha,
        beta: (1.0 - alpha) * (1.0 - alpha) / 4.0,
        gamma: 0.5 - alpha,
    };
    let n = from.acc.len();
    let mut a = from.acc.clone();
    let mut lambda = from.lambda.clone();
    let mut fresh = false;
    if cache.is_none() {
        let base = step.eval(&a, &lambda)?;
        *cache = Some(Cache {
            factor: step.matrix(&a, &lambda, &base)?,
        });
        stats.jacobians += 1;
        fresh = true;
    }
    let mut stale = 0;
    for _ in 0..settings.max_newton {
        let ev = step.eval(&a, &lambda)?;
        let rhs: Vec<f64> = ev.r.iter().map(|v| -v).collect();
        let factor = &cache.as_ref().expect("built above").factor;
        let (dx, _) = factor.solve(&rhs);
        stats.newton_iterations += 1;
        if dx.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        for (ai, d) in a.iter_mut().zip(&dx[..n]) {
            *ai += d;
        }
        for (li, d) in lambda.iter_mut().zip(&dx[n..]) {
            *li += d;
        }
        let correction = step.gamma * (1.0 + alpha) * h * inf_norm(&dx[..n]);
        if correction <= settings.newton_tol * (1.0 + inf_norm(&ev.v)) && ev.phi <= CONSTRAINT_TOL {
            let fin = step.eval(&a, &lambda)?;
            if fin.phi > 10.0 * CONSTRAINT_TOL {
                return Ok(None);
            }
            let mut state = from.state.clone();
            state.params = fin.params;
            state.angles = problem.track_angles(&state.params);
            return Ok(Some(Point {
                time: problem.time,
                state,
                v: fin.v,
                aux: step.aux(&a),
                acc: a,
                lambda,
            }));
        }
        stale += 1;
        if !fresh && stale >= STALE_ITERATIONS {
            let base = step.eval(&a, &lambda)?;
            *cache = Some(Cache {
                factor: step.matrix(&a, &lambda, &base)?,
            });
            stats.jacobians += 1;
            fresh = true;
        }
    }
    Ok(None)
}

fn check_settings(settings: &HhtSettings) -> Result<(), MbdError> {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    let ok = (-1.0 / 3.0 - 1e-12..=0.0).contains(&settings.alpha)
        && positive(settings.step)
        && positive(settings.min_step)
        && settings.max_step > 0.0
        && !settings.max_step.is_nan()
        && settings.min_step <= settings.step
        && settings.step <= settings.max_step
        && settings.error_tol.is_none_or(positive)
        && settings.max_newton >= 1
        && positive(settings.newton_tol);
    if ok {
        Ok(())
    } else {
        Err(MbdError::new(
            codes::BAD_INTEGRATOR_SETTINGS,
            format!("unusable integrator settings: {settings:?}"),
            "α must lie in [−1/3, 0]; steps, tolerances must be positive and finite with min_step ≤ step ≤ max_step",
        ))
    }
}

/// Error-test outcome of an accepted Newton solve.
enum Verdict {
    Accept,
    Reject,
}

/// The local error ratio of a converged step (estimate / tolerance) and
/// the controller's verdict with the next step proposal.
/// `None` as the proposal: keep the controller step as it is.
fn control(settings: &HhtSettings, from: &Point, to: &Point, h: f64) -> (Verdict, Option<f64>) {
    let Some(tol) = settings.error_tol else {
        return (Verdict::Accept, None);
    };
    let alpha = settings.alpha;
    let beta = (1.0 - alpha) * (1.0 - alpha) / 4.0;
    let jump = from
        .acc
        .iter()
        .zip(&to.acc)
        .fold(0.0_f64, |m, (p, q)| m.max((q - p).abs()));
    let ratio = (beta - 1.0 / 6.0) * h * h * jump / tol;
    let ideal = if ratio > 0.0 {
        (SAFETY * ratio.powf(-1.0 / 3.0)).clamp(MAX_SHRINK, MAX_GROWTH)
    } else {
        MAX_GROWTH
    };
    // Hysteresis: every step-size change re-excites the damped λ /
    // acceleration oscillation of index 3 (the estimate's noise), so the step
    // only changes when the change is worth it.
    let next = (h * ideal).clamp(settings.min_step, settings.max_step);
    if ratio > 1.0 {
        (Verdict::Reject, Some(next))
    } else if ratio > SHRINK_ABOVE || ideal >= GROW_AT {
        (Verdict::Accept, Some(next))
    } else {
        (Verdict::Accept, None)
    }
}

fn too_small(time: f64, h: f64, why: &str) -> MbdError {
    MbdError::new(
        codes::INTEGRATION_FAILED,
        format!("{why} at t = {time}: the step fell to {h:.3e} s, below the minimum"),
        "the motion may reach a lock-up/toggle position, or the system is too stiff for the tolerance (raise min_step or the error tolerance)",
    )
}

/// Simulate `sys` over an output time grid (finite, non-decreasing; the
/// first time is the start): the dynamic initial condition at `times[0]`
/// ([`dynamic_ic`]), then index-3 HHT-α steps landing on every output time.
/// Each frame carries the projected velocity, acceleration and multipliers.
/// With `error_tol` the step follows the local error estimate; a step that
/// cannot converge or pass the error test above `min_step` is
/// [`codes::INTEGRATION_FAILED`].
pub fn simulate(
    sys: &System,
    times: &[f64],
    settings: &HhtSettings,
) -> Result<Simulation, MbdError> {
    check_settings(settings)?;
    if times.is_empty()
        || times.iter().any(|t| !t.is_finite())
        || times.windows(2).any(|w| w[1] < w[0])
    {
        return Err(MbdError::new(
            codes::BAD_TIME_GRID,
            "the output grid must be non-empty, finite and non-decreasing",
            "pass output times in ascending order, starting at the initial time",
        ));
    }
    let first = dynamic_ic(sys, times[0])?;
    let mut point = {
        let problem = Problem::new(sys, times[0], &first.state.angles);
        let v = flatten(&problem, &slot_twists(&problem, &first.velocity));
        let a = flatten(&problem, &slot_twists(&problem, &first.acceleration));
        let lambda: Vec<f64> = first.multipliers.iter().map(|(_, l)| *l).collect();
        Point {
            time: times[0],
            state: first.state.clone(),
            v,
            aux: a.clone(),
            acc: a,
            lambda,
        }
    };
    let mut frames = vec![first];
    let mut stats = IntegrationStats::default();
    let mut cache: Option<Cache> = None;
    // The controller's step (fixed step: the nominal one).
    let mut h_ctrl = settings.step;
    for &target in &times[1..] {
        let time_eps = 1e-12 * target.abs().max(1.0);
        while target - point.time > time_eps {
            let remaining = target - point.time;
            let mut h = if settings.error_tol.is_some() {
                // Spread the way to the output time evenly: no sliver step
                // before each output (a sudden step change re-excites the
                // index-3 λ oscillation and spoils the next error estimate).
                remaining / (remaining / h_ctrl - 1e-9).ceil().max(1.0)
            } else {
                h_ctrl.min(remaining)
            };
            // After a rejection the accepted step may not grow (the standard
            // guard against reject / grow oscillation).
            let mut retried = false;
            loop {
                let landing = target - point.time - h <= time_eps;
                let t1 = if landing { target } else { point.time + h };
                let h_step = t1 - point.time;
                let problem = Problem::new(sys, t1, &point.state.angles);
                let Some(next) =
                    take_step(&problem, &point, h_step, settings, &mut cache, &mut stats)?
                else {
                    stats.rejected += 1;
                    cache = None;
                    retried = true;
                    h *= 0.5;
                    if settings.error_tol.is_some() {
                        h_ctrl = h;
                    }
                    if h < settings.min_step {
                        return Err(too_small(
                            point.time,
                            h,
                            "the HHT corrector did not converge",
                        ));
                    }
                    continue;
                };
                let (verdict, proposal) = control(settings, &point, &next, h_step);
                match verdict {
                    Verdict::Reject => {
                        stats.rejected += 1;
                        retried = true;
                        h = proposal.unwrap_or(h_step).min(0.9 * h_step);
                        h_ctrl = h;
                        if h < settings.min_step {
                            return Err(too_small(point.time, h, "the local error test failed"));
                        }
                    }
                    Verdict::Accept => {
                        // (Spread steps are at least half the controller step, so
                        // their proposals are representative; no growth right
                        // after a rejection.)
                        if let Some(p) = proposal {
                            h_ctrl = if retried { p.min(h_step) } else { p };
                        } else if retried {
                            h_ctrl = h_ctrl.min(h_step);
                        }
                        point = next;
                        stats.steps += 1;
                        break;
                    }
                }
            }
        }
        let problem = Problem::new(sys, target, &point.state.angles);
        let mut frame = output_frame(&problem, &point.state, &problem.twists(&point.v))?;
        frame.time = target;
        frames.push(frame);
    }
    Ok(Simulation { frames, stats })
}
