//! Time integration (M2): the index-3 HHT-α method (Hilber–Hughes–Taylor,
//! the Adams `HHT` integrator) on the maximal-coordinate DAE.
//!
//! **Scheme.** Unknowns per step are the accelerations `a` and multipliers
//! `λ` at `t + h`; positions and velocities follow from the Newmark
//! formulas with `β = (1 − α)²/4`, `γ = ½ − α`:
//!
//! ```text
//! Δ    = h·vₙ + h²·((½ − β)·aₙ + β·a)          q = exp(Δ)·qₙ  (Lie-group increment)
//! v    = vₙ + h·((1 − γ)·aₙ + γ·a)
//! M(q)·a + b(q, v) + (1 + α)·(Jᵀλ − Q)(q, v) − α·(Jᵀλ − Q)ₙ = 0
//! Φ(q, t + h) / (β·h²) = 0
//! ```
//!
//! Index 3: the position constraints are imposed directly — no Baumgarte
//! terms. The rotation increment is applied as `R ← exp(Δθ)·R` in the
//! world-frame tangent space of [`crate::Twist`] (second-order consistent
//! with `Ṙ = [ω]×·R`), so no quaternion constraint enters the step. The
//! constraint rows are scaled by `1/(β·h²)` (Negrut et al. 2007): the
//! iteration matrix `[∂r/∂a  (1+α)·Jᵀ; J  0]` then stays well conditioned
//! as `h → 0`. `∂r/∂a` (mass, plus the stiffness and damping of the forces
//! through `q(a)`, `v(a)`) is formed by forward differences and reused
//! across steps (modified Newton) until convergence slows.
//!
//! `α ∈ [−⅓, 0]`: `α = 0` is the trapezoidal rule (no numerical damping,
//! conservative), negative values damp the high-frequency content; the
//! default −0.3 is Adams'. The method is second-order accurate for every
//! α.
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
use crate::kinematics::{KinematicState, Problem, Twist};
use crate::linalg::RowFactor;
use crate::system::System;

/// The Adams default HHT α.
pub const DEFAULT_ALPHA: f64 = -0.3;

/// Position-constraint tolerance of a converged step (m / rad).
const CONSTRAINT_TOL: f64 = 1.0e-10;

/// Relative forward-difference step of the iteration matrix.
const FD_STEP: f64 = 1.0e-6;

/// A step is halved at most down to this fraction of the nominal step.
const MIN_STEP_FRACTION: f64 = 1.0 / 1024.0;

/// Stale-matrix Newton iterations before the iteration matrix is rebuilt.
const STALE_ITERATIONS: usize = 3;

/// Integrator settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HhtSettings {
    /// HHT α in `[−⅓, 0]` (0: trapezoidal, no numerical damping).
    pub alpha: f64,
    /// Nominal step (s). Steps are shortened to land on every output time
    /// and halved when Newton fails.
    pub step: f64,
    /// Newton iteration cap per step.
    pub max_newton: usize,
    /// Newton tolerance on the velocity correction `γ·h·|δa|`, relative to
    /// `1 + |v|`.
    pub newton_tol: f64,
}

impl Default for HhtSettings {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            step: 1.0e-3,
            max_newton: 12,
            newton_tol: 1.0e-10,
        }
    }
}

/// Integrator work counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IntegrationStats {
    /// Accepted steps.
    pub steps: usize,
    /// Rejected (halved) step attempts.
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
    a: Vec<f64>,
    lambda: Vec<f64>,
    /// `Jᵀλ − Q` at this instant (the HHT force history).
    g: Vec<f64>,
}

/// One residual evaluation of a step.
struct Eval {
    /// Dynamic rows, then the scaled constraint rows.
    r: Vec<f64>,
    phi: f64,
    params: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    j: Vec<Vec<f64>>,
}

fn inf_norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// `Jᵀλ − Q`.
fn constraint_minus_applied(j: &[Vec<f64>], lambda: &[f64], q: &[f64]) -> Vec<f64> {
    let mut g: Vec<f64> = q.iter().map(|v| -v).collect();
    for (row, l) in j.iter().zip(lambda) {
        for (gi, ji) in g.iter_mut().zip(row) {
            *gi += ji * l;
        }
    }
    g
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
    /// Positions and velocities from the trial accelerations.
    fn advance(&self, a: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (h, p) = (self.h, self.from);
        let delta: Vec<f64> = (0..a.len())
            .map(|i| h * p.v[i] + h * h * ((0.5 - self.beta) * p.a[i] + self.beta * a[i]))
            .collect();
        let params = retract(self.problem, &p.state.params, &delta);
        let v = (0..a.len())
            .map(|i| p.v[i] + h * ((1.0 - self.gamma) * p.a[i] + self.gamma * a[i]))
            .collect();
        (params, v)
    }

    fn eval(&self, a: &[f64], lambda: &[f64]) -> Result<Eval, MbdError> {
        let problem = self.problem;
        let (params, v) = self.advance(a);
        let vel = problem.twists(&v);
        let q = flatten(problem, &applied_forces(problem, &params, &vel)?);
        let j = problem.jacobian(&params);
        let acc = problem.twists(a);
        let inertial: Vec<Twist> = inertias(problem, &params)
            .iter()
            .enumerate()
            .map(|(slot, i)| {
                i.map_or_else(Twist::default, |i| {
                    let ma = i.apply(acc[slot]);
                    let b = i.bias(vel[slot].angular);
                    Twist {
                        linear: ma.linear + b.linear,
                        angular: ma.angular + b.angular,
                    }
                })
            })
            .collect();
        let inertial = flatten(problem, &inertial);
        let g = constraint_minus_applied(&j, lambda, &q);
        let mut r: Vec<f64> = (0..a.len())
            .map(|c| inertial[c] + (1.0 + self.alpha) * g[c] - self.alpha * self.from.g[c])
            .collect();
        let mut phi = vec![0.0; problem.rows.len()];
        problem.constraint_residuals(&params, &mut phi);
        let scale = 1.0 / (self.beta * self.h * self.h);
        r.extend(phi.iter().map(|p| p * scale));
        Ok(Eval {
            r,
            phi: inf_norm(&phi),
            params,
            v,
            g,
            j,
        })
    }

    /// The iteration matrix at `(a, λ)`: `[∂r/∂a (1+α)·Jᵀ; J 0]`.
    fn matrix(&self, a: &[f64], lambda: &[f64], base: &Eval) -> Result<RowFactor, MbdError> {
        let n = a.len();
        let m = base.j.len();
        let mut rows = vec![vec![0.0; n + m]; n + m];
        let mut trial = a.to_vec();
        for c in 0..n {
            let eps = FD_STEP * (1.0 + a[c].abs());
            trial[c] = a[c] + eps;
            let e = self.eval(&trial, lambda)?;
            trial[c] = a[c];
            for (row, (ri, bi)) in rows.iter_mut().zip(e.r.iter().zip(&base.r)).take(n) {
                row[c] = (ri - bi) / eps;
            }
        }
        for (k, jr) in base.j.iter().enumerate() {
            for (i, row) in rows.iter_mut().enumerate().take(n) {
                row[n + k] = (1.0 + self.alpha) * jr[i];
            }
            rows[n + k][..n].copy_from_slice(jr);
        }
        Ok(RowFactor::new(&rows, n + m))
    }
}

/// The iteration matrix reused across steps, valid for one step size.
struct Cache {
    h: f64,
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
    let n = from.a.len();
    let mut a = from.a.clone();
    let mut lambda = from.lambda.clone();
    let mut fresh = false;
    if cache.as_ref().is_none_or(|c| (c.h - h).abs() > 1e-6 * h) {
        let base = step.eval(&a, &lambda)?;
        *cache = Some(Cache {
            h,
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
        let correction = step.gamma * h * inf_norm(&dx[..n]);
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
                a,
                lambda,
                g: fin.g,
            }));
        }
        stale += 1;
        if !fresh && stale >= STALE_ITERATIONS {
            let base = step.eval(&a, &lambda)?;
            *cache = Some(Cache {
                h,
                factor: step.matrix(&a, &lambda, &base)?,
            });
            stats.jacobians += 1;
            fresh = true;
        }
    }
    Ok(None)
}

fn check_settings(settings: &HhtSettings) -> Result<(), MbdError> {
    let ok = (-1.0 / 3.0 - 1e-12..=0.0).contains(&settings.alpha)
        && settings.step.is_finite()
        && settings.step > 0.0
        && settings.max_newton >= 1
        && settings.newton_tol.is_finite()
        && settings.newton_tol > 0.0;
    if ok {
        Ok(())
    } else {
        Err(MbdError::new(
            codes::BAD_INTEGRATOR_SETTINGS,
            format!("unusable integrator settings: {settings:?}"),
            "α must lie in [−1/3, 0], the step and Newton tolerance must be positive and finite",
        ))
    }
}

/// Simulate `sys` over an output time grid (finite, non-decreasing; the
/// first time is the start): the dynamic initial condition at `times[0]`
/// ([`dynamic_ic`]), then index-3 HHT-α steps landing on every output time.
/// Each frame carries the projected velocity, acceleration and multipliers.
/// A step that cannot converge even at `step/1024` is
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
        let q = flatten(
            &problem,
            &applied_forces(&problem, &first.state.params, &problem.twists(&v))?,
        );
        let g = constraint_minus_applied(&problem.jacobian(&first.state.params), &lambda, &q);
        Point {
            time: times[0],
            state: first.state.clone(),
            v,
            a,
            lambda,
            g,
        }
    };
    let mut frames = vec![first];
    let mut stats = IntegrationStats::default();
    let mut cache: Option<Cache> = None;
    let min_step = settings.step * MIN_STEP_FRACTION;
    for &target in &times[1..] {
        let time_eps = 1e-12 * target.abs().max(1.0);
        while target - point.time > time_eps {
            let mut h = settings.step.min(target - point.time);
            loop {
                let landing = target - point.time - h <= time_eps;
                let t1 = if landing { target } else { point.time + h };
                let problem = Problem::new(sys, t1, &point.state.angles);
                match take_step(
                    &problem,
                    &point,
                    t1 - point.time,
                    settings,
                    &mut cache,
                    &mut stats,
                )? {
                    Some(next) => {
                        point = next;
                        stats.steps += 1;
                        break;
                    }
                    None => {
                        stats.rejected += 1;
                        h *= 0.5;
                        if h < min_step {
                            return Err(MbdError::new(
                                codes::INTEGRATION_FAILED,
                                format!(
                                    "the HHT corrector did not converge at t = {} even with step {h:.3e}",
                                    point.time
                                ),
                                "the motion may reach a lock-up/toggle position, or the system is too stiff for the nominal step",
                            ));
                        }
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
