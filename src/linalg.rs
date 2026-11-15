//! Dense rank-revealing linear algebra: the diagnosing fallback behind the
//! sparse saddle solver ([`crate::sparse`]), and the redundancy, conflict
//! and null-space analyses.
//!
//! The constraint Jacobian `J` (equations × tangent coordinates) of a real
//! CAD assembly is routinely rank-deficient in its rows: redundant
//! equations. [`RowFactor`] runs a column-pivoted Householder QR of `Jᵀ`
//! (rows equilibrated to unit norm first, so pivoting compares directions,
//! not units): the pivot columns are a maximal independent set of
//! equations, the rest are dependent. A right side is then solved on the
//! independent equations (minimum-norm, so free directions stay at rest)
//! and every dependent equation is checked against the solution —
//! consistent ones are redundant, inconsistent ones are conflicts.
//!
//! Deterministic: pivot ties break by equation order.

/// Rank threshold on the remaining column norm (equilibrated columns start
/// at norm 1).
const RANK_TOL: f64 = 1.0e-9;

/// A dependent equation is consistent when its scaled residual is within
/// this fraction of the scaled right side's magnitude (floor 1).
const CONSIST_TOL: f64 = 1.0e-8;

/// A rank-revealing factorization of the rows of `J`.
pub(crate) struct RowFactor {
    n: usize,
    rows: Vec<Vec<f64>>,
    scale: Vec<f64>,
    perm: Vec<usize>,
    rank: usize,
    /// Factored columns of `JᵀP` in pivot order: entries `[..=i]` of column
    /// `j ≥ i` are row `i` of `R`.
    cols: Vec<Vec<f64>>,
    /// Householder vectors `v_i` (acting on entries `i..n`) and `|v_i|²`.
    house: Vec<(Vec<f64>, f64)>,
}

impl RowFactor {
    /// Factor `rows` (each of length `n`, the number of unknowns).
    pub fn new(rows: &[Vec<f64>], n: usize) -> Self {
        let m = rows.len();
        let mut scale = Vec::with_capacity(m);
        let mut cols = Vec::with_capacity(m);
        for row in rows {
            debug_assert_eq!(row.len(), n);
            let s = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if s > f64::MIN_POSITIVE {
                cols.push(row.iter().map(|v| v / s).collect::<Vec<_>>());
                scale.push(s);
            } else {
                cols.push(vec![0.0; n]);
                scale.push(0.0);
            }
        }
        let mut perm: Vec<usize> = (0..m).collect();
        let mut house = Vec::new();
        let mut rank = 0;
        for i in 0..n.min(m) {
            let mut best = i;
            let mut best_norm = -1.0;
            for (k, col) in cols.iter().enumerate().skip(i) {
                let s: f64 = col[i..].iter().map(|v| v * v).sum();
                if s > best_norm {
                    best_norm = s;
                    best = k;
                }
            }
            let norm = best_norm.max(0.0).sqrt();
            if norm <= RANK_TOL {
                break;
            }
            cols.swap(i, best);
            perm.swap(i, best);
            let x0 = cols[i][i];
            let alpha = if x0 >= 0.0 { -norm } else { norm };
            let mut v = cols[i][i..].to_vec();
            v[0] -= alpha;
            let vv: f64 = v.iter().map(|a| a * a).sum();
            for col in cols.iter_mut().skip(i) {
                let proj = 2.0 * v.iter().zip(&col[i..]).map(|(a, b)| a * b).sum::<f64>() / vv;
                for (c, a) in col[i..].iter_mut().zip(&v) {
                    *c -= proj * a;
                }
            }
            house.push((v, vv));
            rank += 1;
        }
        Self {
            n,
            rows: rows.to_vec(),
            scale,
            perm,
            rank,
            cols,
            house,
        }
    }

    /// The rank of `J`.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The dependent equations (indices into the factored rows), in pivot
    /// order.
    pub fn dependent(&self) -> &[usize] {
        &self.perm[self.rank..]
    }

    /// The dependent equations, ascending.
    pub fn dependent_sorted(&self) -> Vec<usize> {
        let mut d = self.dependent().to_vec();
        d.sort_unstable();
        d
    }

    /// Solve `J x = b` on the independent equations (minimum norm) and
    /// return `x` with the dependent equations the solution violates
    /// (conflicts), ascending.
    pub fn solve(&self, b: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let scaled: Vec<f64> = b
            .iter()
            .zip(&self.scale)
            .map(|(bi, s)| if *s > 0.0 { bi / s } else { *bi })
            .collect();
        // Rᵀ y = b_I (forward substitution), x = Q [y; 0].
        let mut y = vec![0.0; self.rank];
        for j in 0..self.rank {
            let col = &self.cols[j];
            let acc: f64 = (0..j).map(|i| col[i] * y[i]).sum();
            y[j] = (scaled[self.perm[j]] - acc) / col[j];
        }
        let mut x = vec![0.0; self.n];
        x[..self.rank].copy_from_slice(&y);
        for (i, (v, vv)) in self.house.iter().enumerate().rev() {
            let proj = 2.0 * v.iter().zip(&x[i..]).map(|(a, b)| a * b).sum::<f64>() / vv;
            for (xi, a) in x[i..].iter_mut().zip(v) {
                *xi -= proj * a;
            }
        }
        let magnitude = scaled.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        let mut conflicts: Vec<usize> = self
            .dependent()
            .iter()
            .copied()
            .filter(|&k| {
                let lhs: f64 = self.rows[k].iter().zip(&x).map(|(a, b)| a * b).sum();
                let res = lhs - b[k];
                let res = if self.scale[k] > 0.0 {
                    res / self.scale[k]
                } else {
                    res
                };
                res.abs() > CONSIST_TOL * magnitude
            })
            .collect();
        conflicts.sort_unstable();
        (x, conflicts)
    }
}


/// The max norm.
pub(crate) fn inf_norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// The transpose of `rows` (each of length `n`).
pub(crate) fn transpose(rows: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|c| rows.iter().map(|r| r[c]).collect()).collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A full-rank square system solves exactly.
    #[test]
    fn square_system_solves() {
        let rows = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 4.0],
        ];
        let f = RowFactor::new(&rows, 3);
        assert_eq!(f.rank(), 3);
        let (x, conflicts) = f.solve(&[3.0, 5.0, 5.0]);
        assert!(conflicts.is_empty());
        for (xi, want) in x.iter().zip([1.0, 1.0, 1.0]) {
            assert!((xi - want).abs() < 1e-12, "{x:?}");
        }
    }

    /// A duplicated equation is dependent; consistent → no conflict,
    /// inconsistent → reported. An underdetermined system takes the
    /// minimum-norm solution.
    #[test]
    fn redundant_and_conflicting_rows() {
        let rows = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let f = RowFactor::new(&rows, 2);
        assert_eq!(f.rank(), 1);
        assert_eq!(f.dependent(), &[1]);
        let (x, conflicts) = f.solve(&[2.0, 4.0]);
        assert!(conflicts.is_empty());
        assert!(
            (x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12,
            "minimum norm: {x:?}"
        );
        let (_, conflicts) = f.solve(&[2.0, 5.0]);
        assert_eq!(conflicts, vec![1]);
    }

    /// An all-zero equation is dependent, and conflicting unless its right
    /// side is zero.
    #[test]
    fn empty_rows_are_dependent() {
        let rows = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
        let f = RowFactor::new(&rows, 2);
        assert_eq!(f.rank(), 1);
        assert_eq!(f.solve(&[0.0, 1.0]).1, Vec::<usize>::new());
        assert_eq!(f.solve(&[1.0, 1.0]).1, vec![0]);
    }
}
