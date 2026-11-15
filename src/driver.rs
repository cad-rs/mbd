//! Drivers: prescribed motion on one slot of one joint (the MbD "motor"),
//! and the time profiles they follow.
//!
//! Every profile evaluates its value and its first two time derivatives
//! ([`DriverSample`]): the velocity and acceleration levels of the kinematic
//! solve read them directly, so a profile whose derivatives are discontinuous
//! shows up as jumps in the solved velocities/accelerations. The commercial
//! vocabulary this mirrors: constant rate, data points (piecewise linear or
//! cubic spline), oscillating (harmonic) and the smooth step (Adams `STEP5`).

use serde::{Deserialize, Serialize};

use crate::joint::JointId;

/// Which slot of a joint a driver moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverSlot {
    /// Joint rotation about the common Z axis (rad): revolute, cylindrical,
    /// planar.
    Rotation,
    /// Joint translation along the common Z axis (m): prismatic,
    /// cylindrical.
    Translation,
}

/// How the driven value evolves over time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "profile", rename_all = "snake_case")]
pub enum DriverProfile {
    /// `value(t) = start + rate * t`.
    Constant { start: f64, rate: f64 },
    /// Linear interpolation through `(time_s, value)` points sorted by time,
    /// clamped at the ends. The velocity is piecewise constant (it jumps at
    /// every interior point) and the acceleration is zero between points —
    /// use [`DriverProfile::Spline`] when the acceleration level matters.
    Keyframes { points: Vec<(f64, f64)> },
    /// `value(t) = offset + amplitude * sin(2π·frequency_hz·t + phase)`.
    Harmonic {
        offset: f64,
        amplitude: f64,
        frequency_hz: f64,
        phase: f64,
    },
    /// A quintic smooth step (Adams `STEP5`): `v0` until `t0`, `v1` from
    /// `t1`, with zero velocity and acceleration at both ends (C2).
    Step5 { t0: f64, v0: f64, t1: f64, v1: f64 },
    /// A cubic spline through `(time_s, value)` points (strictly increasing
    /// times, at least two), clamped to zero slope at both ends and held
    /// constant outside them — "start and end at rest", C2 inside the range.
    Spline { points: Vec<(f64, f64)> },
}

/// A driver profile evaluated at one time: the value and its first two time
/// derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DriverSample {
    pub value: f64,
    pub velocity: f64,
    pub acceleration: f64,
}

/// A prescribed motion on one slot of one joint (the MbD "motor"). A joint
/// slot with a driver loses that degree of freedom to prescription; joints
/// without any driver keep their free degrees of freedom for the dynamics
/// to determine.
///
/// A rotation driver prescribes the angle of marker 2's X axis about marker
/// 1's Z axis, measured from marker 1's X axis; the angle is unbounded (a
/// motor may turn any number of revolutions).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Driver {
    pub joint: JointId,
    pub slot: DriverSlot,
    pub profile: DriverProfile,
}

impl Driver {
    /// The driven value at `t` (s).
    pub fn value_at(&self, t: f64) -> f64 {
        self.profile.sample(t).value
    }

    /// The driven value and its first two time derivatives at `t` (s).
    pub fn sample(&self, t: f64) -> DriverSample {
        self.profile.sample(t)
    }
}

impl DriverProfile {
    /// The profile and its first two time derivatives at `t` (s).
    pub fn sample(&self, t: f64) -> DriverSample {
        match self {
            DriverProfile::Constant { start, rate } => DriverSample {
                value: start + rate * t,
                velocity: *rate,
                acceleration: 0.0,
            },
            DriverProfile::Keyframes { points } => keyframes(points, t),
            DriverProfile::Harmonic {
                offset,
                amplitude,
                frequency_hz,
                phase,
            } => {
                let w = 2.0 * std::f64::consts::PI * frequency_hz;
                let (s, c) = (w * t + phase).sin_cos();
                DriverSample {
                    value: offset + amplitude * s,
                    velocity: amplitude * w * c,
                    acceleration: -amplitude * w * w * s,
                }
            }
            DriverProfile::Step5 { t0, v0, t1, v1 } => step5(*t0, *v0, *t1, *v1, t),
            DriverProfile::Spline { points } => spline(points, t),
        }
    }

    /// Whether the profile's data is usable: finite numbers, sorted times,
    /// enough points. `Err` carries the reason (the validation gate turns it
    /// into [`crate::codes::BAD_DRIVER_PROFILE`]).
    pub fn check(&self) -> Result<(), &'static str> {
        let finite = |v: f64| v.is_finite();
        match self {
            DriverProfile::Constant { start, rate } => {
                if !(finite(*start) && finite(*rate)) {
                    return Err("non-finite constant-rate parameters");
                }
            }
            DriverProfile::Keyframes { points } => {
                if points.is_empty() {
                    return Err("keyframes need at least one point");
                }
                if points.iter().any(|(t, v)| !(finite(*t) && finite(*v))) {
                    return Err("non-finite keyframe");
                }
                if points.windows(2).any(|w| w[1].0 < w[0].0) {
                    return Err("keyframe times must be non-decreasing");
                }
            }
            DriverProfile::Harmonic {
                offset,
                amplitude,
                frequency_hz,
                phase,
            } => {
                if ![*offset, *amplitude, *frequency_hz, *phase]
                    .into_iter()
                    .all(finite)
                {
                    return Err("non-finite harmonic parameters");
                }
            }
            DriverProfile::Step5 { t0, v0, t1, v1 } => {
                if ![*t0, *v0, *t1, *v1].into_iter().all(finite) {
                    return Err("non-finite step parameters");
                }
                if t1 <= t0 {
                    return Err("step needs t1 > t0");
                }
            }
            DriverProfile::Spline { points } => {
                if points.len() < 2 {
                    return Err("a spline needs at least two points");
                }
                if points.iter().any(|(t, v)| !(finite(*t) && finite(*v))) {
                    return Err("non-finite spline point");
                }
                if points.windows(2).any(|w| w[1].0 <= w[0].0) {
                    return Err("spline times must be strictly increasing");
                }
            }
        }
        Ok(())
    }
}

fn held(value: f64) -> DriverSample {
    DriverSample {
        value,
        velocity: 0.0,
        acceleration: 0.0,
    }
}

fn keyframes(points: &[(f64, f64)], t: f64) -> DriverSample {
    let Some(first) = points.first() else {
        return DriverSample::default();
    };
    if t <= first.0 {
        return held(first.1);
    }
    for pair in points.windows(2) {
        let (t0, v0) = pair[0];
        let (t1, v1) = pair[1];
        if t <= t1 {
            let span = t1 - t0;
            if span <= 0.0 {
                return held(v1);
            }
            let slope = (v1 - v0) / span;
            return DriverSample {
                value: v0 + slope * (t - t0),
                velocity: slope,
                acceleration: 0.0,
            };
        }
    }
    held(points[points.len() - 1].1)
}

fn step5(t0: f64, v0: f64, t1: f64, v1: f64, t: f64) -> DriverSample {
    if t <= t0 {
        return held(v0);
    }
    if t >= t1 {
        return held(v1);
    }
    let span = t1 - t0;
    let h = v1 - v0;
    let d = (t - t0) / span;
    // s(d) = d³(10 − 15d + 6d²); s' = 30d²(1−d)²; s'' = 60d(1−d)(1−2d).
    let s = d * d * d * (10.0 - 15.0 * d + 6.0 * d * d);
    let ds = 30.0 * d * d * (1.0 - d) * (1.0 - d);
    let dds = 60.0 * d * (1.0 - d) * (1.0 - 2.0 * d);
    DriverSample {
        value: v0 + h * s,
        velocity: h * ds / span,
        acceleration: h * dds / (span * span),
    }
}

/// Clamped cubic spline (zero end slopes): solve the tridiagonal system for
/// the knot second derivatives, then evaluate the segment holding `t`.
fn spline(points: &[(f64, f64)], t: f64) -> DriverSample {
    let n = points.len();
    if n == 0 {
        return DriverSample::default();
    }
    if n == 1 || t <= points[0].0 {
        return held(points[0].1);
    }
    if t >= points[n - 1].0 {
        return held(points[n - 1].1);
    }
    let h: Vec<f64> = points.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let slope: Vec<f64> = points
        .windows(2)
        .zip(&h)
        .map(|(w, hi)| (w[1].1 - w[0].1) / hi)
        .collect();
    // Rows i = 0..n: sub[i]·M[i-1] + diag[i]·M[i] + sup[i]·M[i+1] = rhs[i].
    let mut sub = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    diag[0] = 2.0 * h[0];
    sup[0] = h[0];
    rhs[0] = 6.0 * slope[0];
    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        rhs[i] = 6.0 * (slope[i] - slope[i - 1]);
    }
    sub[n - 1] = h[n - 2];
    diag[n - 1] = 2.0 * h[n - 2];
    rhs[n - 1] = -6.0 * slope[n - 2];
    // Thomas algorithm (the matrix is strictly diagonally dominant).
    for i in 1..n {
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    let mut m = vec![0.0; n];
    m[n - 1] = rhs[n - 1] / diag[n - 1];
    for i in (0..n - 1).rev() {
        m[i] = (rhs[i] - sup[i] * m[i + 1]) / diag[i];
    }
    let i = points.windows(2).position(|w| t <= w[1].0).unwrap_or(n - 2);
    let (x0, y0) = points[i];
    let (x1, y1) = points[i + 1];
    let hi = x1 - x0;
    let a = (x1 - t) / hi;
    let b = (t - x0) / hi;
    DriverSample {
        value: a * y0
            + b * y1
            + ((a * a * a - a) * m[i] + (b * b * b - b) * m[i + 1]) * hi * hi / 6.0,
        velocity: (y1 - y0) / hi - (3.0 * a * a - 1.0) / 6.0 * hi * m[i]
            + (3.0 * b * b - 1.0) / 6.0 * hi * m[i + 1],
        acceleration: a * m[i] + b * m[i + 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each profile's velocity and acceleration agree with central
    /// differences of its value (away from the kinks of the clamped ends).
    #[test]
    fn profile_derivatives_match_finite_differences() {
        let profiles = [
            DriverProfile::Constant {
                start: 0.5,
                rate: 2.0,
            },
            DriverProfile::Harmonic {
                offset: 0.1,
                amplitude: 0.3,
                frequency_hz: 1.5,
                phase: 0.2,
            },
            DriverProfile::Step5 {
                t0: 0.2,
                v0: -1.0,
                t1: 1.7,
                v1: 2.0,
            },
            DriverProfile::Spline {
                points: vec![(0.0, 0.0), (0.4, 1.0), (1.0, 0.5), (1.5, 2.0), (2.0, 2.0)],
            },
        ];
        let h = 1e-5;
        for profile in &profiles {
            assert!(profile.check().is_ok());
            for k in 1..40 {
                let t = 0.05 * k as f64 + 0.013;
                if t >= 1.99 {
                    continue;
                }
                let s = profile.sample(t);
                let ahead = profile.sample(t + h);
                let behind = profile.sample(t - h);
                let vel = (ahead.value - behind.value) / (2.0 * h);
                let acc = (ahead.velocity - behind.velocity) / (2.0 * h);
                assert!(
                    (vel - s.velocity).abs() < 1e-6,
                    "{profile:?} vel at {t}: {vel} vs {}",
                    s.velocity
                );
                assert!(
                    (acc - s.acceleration).abs() < 1e-5,
                    "{profile:?} acc at {t}: {acc} vs {}",
                    s.acceleration
                );
            }
        }
    }

    /// The spline interpolates its points and rests at both ends; the step
    /// hits its plateaus with zero rates.
    #[test]
    fn spline_and_step_hit_their_endpoints() {
        let points = vec![(0.0, 1.0), (1.0, 3.0), (2.5, -1.0)];
        let spline = DriverProfile::Spline {
            points: points.clone(),
        };
        for (t, v) in &points {
            assert!((spline.sample(*t).value - v).abs() < 1e-12);
        }
        assert!(spline.sample(1e-9).velocity.abs() < 1e-6);
        assert!(spline.sample(2.5 - 1e-9).velocity.abs() < 1e-6);
        assert_eq!(spline.sample(9.0), held(-1.0));
        let step = DriverProfile::Step5 {
            t0: 1.0,
            v0: 0.0,
            t1: 2.0,
            v1: 4.0,
        };
        assert_eq!(step.sample(0.0), held(0.0));
        assert_eq!(step.sample(2.0), held(4.0));
        assert!(
            (step.sample(1.5).value - 2.0).abs() < 1e-12,
            "symmetric midpoint"
        );
    }

    /// Bad profile data is caught by the check, with a reason.
    #[test]
    fn bad_profiles_are_rejected() {
        assert!(DriverProfile::Keyframes { points: vec![] }.check().is_err());
        assert!(DriverProfile::Keyframes {
            points: vec![(1.0, 0.0), (0.0, 1.0)]
        }
        .check()
        .is_err());
        assert!(DriverProfile::Spline {
            points: vec![(0.0, 0.0)]
        }
        .check()
        .is_err());
        assert!(DriverProfile::Spline {
            points: vec![(0.0, 0.0), (0.0, 1.0)]
        }
        .check()
        .is_err());
        assert!(DriverProfile::Step5 {
            t0: 1.0,
            v0: 0.0,
            t1: 1.0,
            v1: 1.0
        }
        .check()
        .is_err());
        assert!(DriverProfile::Constant {
            start: f64::NAN,
            rate: 0.0
        }
        .check()
        .is_err());
    }
}
