//! Second-order jets: a value with its first and second derivative along one
//! motion (forward-mode differentiation).
//!
//! The constraint residuals are written once, over jets. Evaluated with a
//! body motion `(v, ω)` and zero accelerations, a residual jet carries
//! - `v`: the residual itself (the position level),
//! - `d`: the residual rate `Φ_q · q̇` (the velocity level; with a unit motion
//!   in one tangent coordinate this is one Jacobian column, exact),
//! - `dd`: the velocity-quadratic term `(Φ_q q̇)_q q̇` (the acceleration
//!   level's right side `γ`).
//!
//! so the three kinematic levels cannot disagree about the equations.

use glam::{DQuat, DVec3};

/// A scalar jet.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Jet {
    pub v: f64,
    pub d: f64,
    pub dd: f64,
}

/// A 3-vector jet.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Jet3 {
    pub v: DVec3,
    pub d: DVec3,
    pub dd: DVec3,
}

impl Jet {
    /// `atan2(s, c)` with its derivatives (the angle of the vector `(c, s)`).
    pub fn atan2(s: Jet, c: Jet) -> Jet {
        let n = s.v * s.v + c.v * c.v;
        if n == 0.0 {
            return Jet::default();
        }
        let num = c.v * s.d - s.v * c.d;
        let dnum = c.v * s.dd - s.v * c.dd;
        let dn = 2.0 * (s.v * s.d + c.v * c.d);
        Jet {
            v: s.v.atan2(c.v),
            d: num / n,
            dd: (dnum * n - num * dn) / (n * n),
        }
    }
}

impl Jet3 {
    pub fn dot(self, o: Jet3) -> Jet {
        Jet {
            v: self.v.dot(o.v),
            d: self.d.dot(o.v) + self.v.dot(o.d),
            dd: self.dd.dot(o.v) + 2.0 * self.d.dot(o.d) + self.v.dot(o.dd),
        }
    }

    pub fn x(self) -> Jet {
        Jet {
            v: self.v.x,
            d: self.d.x,
            dd: self.dd.x,
        }
    }

    pub fn y(self) -> Jet {
        Jet {
            v: self.v.y,
            d: self.d.y,
            dd: self.dd.y,
        }
    }

    pub fn z(self) -> Jet {
        Jet {
            v: self.v.z,
            d: self.d.z,
            dd: self.dd.z,
        }
    }
}

impl std::ops::Sub for Jet3 {
    type Output = Jet3;
    fn sub(self, o: Jet3) -> Jet3 {
        Jet3 {
            v: self.v - o.v,
            d: self.d - o.d,
            dd: self.dd - o.dd,
        }
    }
}

/// A rigid body moving with constant world-frame linear velocity `v` (of
/// its frame origin) and angular velocity `w`, at pose `(t, r)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BodyMotion {
    pub t: DVec3,
    pub r: DQuat,
    pub v: DVec3,
    pub w: DVec3,
}

/// A marker frame in world space, as jets: origin and the three axes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameJet {
    pub p: Jet3,
    pub x: Jet3,
    pub y: Jet3,
    pub z: Jet3,
}

impl BodyMotion {
    /// A body-fixed direction: `a = R·local`, `ȧ = ω×a`, `ä = ω×(ω×a)`.
    pub fn direction(&self, local: DVec3) -> Jet3 {
        let a = self.r * local;
        let wa = self.w.cross(a);
        Jet3 {
            v: a,
            d: wa,
            dd: self.w.cross(wa),
        }
    }

    /// A body-fixed point: `p = t + R·local`, `ṗ = v + ω×r`, `p̈ = ω×(ω×r)`.
    pub fn point(&self, local: DVec3) -> Jet3 {
        let arm = self.r * local;
        let wa = self.w.cross(arm);
        Jet3 {
            v: self.t + arm,
            d: self.v + wa,
            dd: self.w.cross(wa),
        }
    }

    /// The world frame of a marker at `(translation, rotation)` relative to
    /// the body frame.
    pub fn frame(&self, translation: DVec3, rotation: DQuat) -> FrameJet {
        FrameJet {
            p: self.point(translation),
            x: self.direction(rotation * DVec3::X),
            y: self.direction(rotation * DVec3::Y),
            z: self.direction(rotation * DVec3::Z),
        }
    }
}
