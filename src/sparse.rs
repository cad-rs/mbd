//! Sparse symmetric factorization for the dynamics saddle-point systems
//! (performance channel, roadmap §3 "Costs").
//!
//! The systems are `K = [A Jᵀ; J 0]` with `A` the (block-sparse,
//! positive definite) mass or iteration matrix and `J` the constraint
//! Jacobian, whose rows each touch one or two bodies. Their dense
//! rank-revealing QR ([`crate::linalg::RowFactor`]) costs O((n+m)³); here
//! `K_δ = [A + δ_A·I  Jᵀ; J  0]` (a tiny body-block shift keeps zero-inertia
//! directions factorable) is factored `P·K_δ·Pᵀ = L·D·Lᵀ` with sparse
//! elimination (Davis' LDL algorithm: elimination tree, up-looking numeric
//! factorization).
//!
//! **Ordering.** Constrained minimum degree: a constraint row becomes
//! eligible once every body coordinate it touches is eliminated, so each
//! constraint pivot is the (negative semi-definite) Schur complement over
//! eliminated bodies and each body pivot stays positive — and a chain or a
//! tree fills in only along its topology (O(bodies) work).
//!
//! **Redundancy.** A dependent constraint row meets a pivot that is pure
//! rounding: that row is redundant, and it is removed on the spot (its
//! pivot replaced by a huge one, multiplier 0 — the Adams / OndselSolver
//! convention) rather than divided by, which would amplify rounding into
//! every later pivot. Iterative refinement against the unshifted matrix
//! removes the body-block shift. A system the
//! refinement cannot satisfy (inconsistent constraints, an unresisted load)
//! is reported as unsolved so the caller can fall back to the dense,
//! diagnosing path.

use std::collections::BTreeSet;

/// Relative regularization of the body block.
const DELTA_A: f64 = 1.0e-13;

/// A constraint pivot below this fraction of its row's scale marks the row
/// redundant (relative singular value ≈ 1e-5: a dependent row's pivot is
/// rounding, ~1e-12 of the scale).
const REDUNDANT_PIVOT: f64 = 1.0e-10;

/// The pivot that replaces a dropped (redundant) one: removes the equation.
const DROPPED: f64 = 1.0e100;

/// Refinement: residual tolerance (relative to the right side) and
/// iteration cap.
const REFINE_TOL: f64 = 1.0e-10;
const REFINE_MAX: usize = 8;

/// Sparse matrix rows: per row, `(column, value)` pairs (columns
/// ascending). The constraint Jacobian in this form costs O(rows) to build
/// and to apply.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SparseRows {
    pub(crate) n_cols: usize,
    pub(crate) rows: Vec<Vec<(usize, f64)>>,
}

impl SparseRows {
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// Dense rows (the fallback and diagnostic paths).
    pub(crate) fn to_dense(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .map(|r| {
                let mut d = vec![0.0; self.n_cols];
                for &(c, v) in r {
                    d[c] = v;
                }
                d
            })
            .collect()
    }

    /// `Jᵀ·λ`.
    pub(crate) fn t_mul(&self, lambda: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_cols];
        for (row, l) in self.rows.iter().zip(lambda) {
            for &(c, v) in row {
                out[c] += v * l;
            }
        }
        out
    }

    /// From dense rows (exact zeros dropped).
    pub(crate) fn from_dense(rows: &[Vec<f64>], n_cols: usize) -> Self {
        Self {
            n_cols,
            rows: rows
                .iter()
                .map(|r| {
                    r.iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, v)| *v != 0.0)
                        .collect()
                })
                .collect(),
        }
    }
}

/// `(row, column, value)` entries of a square matrix; duplicates add.
pub(crate) type Triplets = Vec<(usize, usize, f64)>;

/// Dense rows as triplets (exact zeros dropped).
pub(crate) fn triplets_from_dense(rows: &[Vec<f64>]) -> Triplets {
    let mut t = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if *v != 0.0 {
                t.push((i, j, *v));
            }
        }
    }
    t
}

/// The dense rows of `[A Jᵀ; J 0]` (`A` as `n × n` triplets): the dense,
/// diagnosing fallback of the saddle solves.
pub(crate) fn dense_saddle(n: usize, a: &[(usize, usize, f64)], j: &SparseRows) -> Vec<Vec<f64>> {
    let m = j.len();
    let mut rows = vec![vec![0.0; n + m]; n + m];
    for &(r, c, v) in a {
        rows[r][c] += v;
    }
    for (k, jr) in j.rows.iter().enumerate() {
        for &(c, v) in jr {
            rows[c][n + k] = v;
            rows[n + k][c] = v;
        }
    }
    rows
}

/// A symmetric matrix in compressed sparse columns, both triangles stored.
#[derive(Clone)]
struct Csc {
    n: usize,
    colp: Vec<usize>,
    rowi: Vec<usize>,
    vals: Vec<f64>,
}

impl Csc {
    /// From entries (duplicates summed; the diagonal always present).
    fn from_entries(n: usize, mut entries: Vec<(usize, usize, f64)>) -> Self {
        entries.extend((0..n).map(|i| (i, i, 0.0)));
        entries.sort_unstable_by_key(|&(i, j, _)| (j, i));
        let mut colp = vec![0; n + 1];
        let mut rowi = Vec::with_capacity(entries.len());
        let mut vals: Vec<f64> = Vec::with_capacity(entries.len());
        let mut last: Option<(usize, usize)> = None;
        for (i, j, v) in entries {
            if last == Some((i, j)) {
                *vals.last_mut().expect("entry") += v;
            } else {
                rowi.push(i);
                vals.push(v);
                colp[j + 1] += 1;
                last = Some((i, j));
            }
        }
        for j in 0..n {
            colp[j + 1] += colp[j];
        }
        Self {
            n,
            colp,
            rowi,
            vals,
        }
    }

    /// Add `shift` to the first `upto` diagonal entries.
    fn shift_diagonal(&mut self, upto: usize, shift: f64) {
        for j in 0..upto {
            let p = (self.colp[j]..self.colp[j + 1])
                .find(|&p| self.rowi[p] == j)
                .expect("diagonal stored");
            self.vals[p] += shift;
        }
    }

    fn mul(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n];
        for (j, xj) in x.iter().enumerate() {
            for p in self.colp[j]..self.colp[j + 1] {
                y[self.rowi[p]] += self.vals[p] * xj;
            }
        }
        y
    }
}

/// Constrained minimum degree: body coordinates (`0..first_body_end`) are
/// always eligible, a constraint row only once every body coordinate it
/// touches is eliminated; among the eligible nodes the one of least current
/// degree goes first (ties: lowest index). Every constraint pivot is then
/// the (negative) Schur complement over bodies already eliminated — well
/// defined, pure rounding for a dependent row — while a chain or a tree
/// fills in only along its topology. (Eliminating *all* bodies first would
/// form `J·A⁻¹·Jᵀ`, dense as soon as `A` couples bodies.)
fn ordering(a: &Csc, first_body_end: usize) -> Vec<usize> {
    let n = a.n;
    let mut adj: Vec<BTreeSet<usize>> = (0..n)
        .map(|j| {
            (a.colp[j]..a.colp[j + 1])
                .map(|p| a.rowi[p])
                .filter(|&i| i != j)
                .collect()
        })
        .collect();
    // Body coordinates each constraint row still waits for.
    let mut pending: Vec<usize> = (0..n)
        .map(|v| {
            if v < first_body_end {
                0
            } else {
                adj[v].iter().filter(|&&u| u < first_body_end).count()
            }
        })
        .collect();
    let rows_of: Vec<Vec<usize>> = (0..n)
        .map(|v| {
            if v < first_body_end {
                adj[v]
                    .iter()
                    .copied()
                    .filter(|&u| u >= first_body_end)
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    let mut done = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let key = |v: usize, adj: &[BTreeSet<usize>]| (adj[v].len(), v);
    let mut queue: BTreeSet<(usize, usize)> = (0..n)
        .filter(|&v| pending[v] == 0)
        .map(|v| key(v, &adj))
        .collect();
    let mut queued: Vec<bool> = pending.iter().map(|&p| p == 0).collect();
    while let Some(&(degree, v)) = queue.iter().next() {
        queue.remove(&(degree, v));
        done[v] = true;
        order.push(v);
        let neighbors: Vec<usize> = adj[v].iter().copied().filter(|&u| !done[u]).collect();
        for &u in &neighbors {
            if queued[u] {
                queue.remove(&key(u, &adj));
            }
        }
        for &u in &neighbors {
            adj[u].remove(&v);
            for &w in &neighbors {
                if w != u {
                    adj[u].insert(w);
                }
            }
        }
        for &r in &rows_of[v] {
            pending[r] -= 1;
            if pending[r] == 0 {
                queued[r] = true;
            }
        }
        // (A row that just became eligible is a neighbour of `v`: it waited
        // for `v`, so it is queued here.)
        for &u in &neighbors {
            if queued[u] {
                queue.insert(key(u, &adj));
            }
        }
        adj[v].clear();
    }
    debug_assert_eq!(order.len(), n);
    order
}

/// `P·K·Pᵀ = L·D·Lᵀ` (unit lower `L` in compressed columns).
struct Ldl {
    perm: Vec<usize>,
    lp: Vec<usize>,
    li: Vec<usize>,
    lx: Vec<f64>,
    d: Vec<f64>,
}

impl Ldl {
    /// Davis' LDL: symbolic (elimination tree, column counts), then the
    /// up-looking numeric factorization. `None` on a zero pivot.
    ///
    /// `drop_below[i]` > 0 marks original index `i` droppable: a pivot of
    /// magnitude below it is replaced by a huge negative one, which removes
    /// that equation (multiplier 0) instead of dividing by it — a
    /// redundant row met at a rounding-size pivot would otherwise amplify
    /// the rounding into every later pivot. The dropped indices are returned.
    fn factor(a: &Csc, perm: Vec<usize>, drop_below: &[f64]) -> Option<(Self, Vec<usize>)> {
        let n = a.n;
        let mut pinv = vec![0; n];
        for (k, &p) in perm.iter().enumerate() {
            pinv[p] = k;
        }
        const NONE: usize = usize::MAX;
        let mut parent = vec![NONE; n];
        let mut flag = vec![0; n];
        let mut lnz = vec![0; n];
        for k in 0..n {
            flag[k] = k;
            let kk = perm[k];
            for p in a.colp[kk]..a.colp[kk + 1] {
                let mut i = pinv[a.rowi[p]];
                if i < k {
                    while flag[i] != k {
                        if parent[i] == NONE {
                            parent[i] = k;
                        }
                        lnz[i] += 1;
                        flag[i] = k;
                        i = parent[i];
                    }
                }
            }
        }
        let mut lp = vec![0; n + 1];
        for k in 0..n {
            lp[k + 1] = lp[k] + lnz[k];
        }
        let nnz = lp[n];
        let mut li = vec![0; nnz];
        let mut lx = vec![0.0; nnz];
        let mut d = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut pattern = vec![0; n];
        let mut dropped = Vec::new();
        lnz.iter_mut().for_each(|c| *c = 0);
        for k in 0..n {
            let mut top = n;
            flag[k] = k;
            let kk = perm[k];
            for p in a.colp[kk]..a.colp[kk + 1] {
                let mut i = pinv[a.rowi[p]];
                if i <= k {
                    y[i] += a.vals[p];
                    let mut len = 0;
                    while flag[i] != k {
                        pattern[len] = i;
                        len += 1;
                        flag[i] = k;
                        i = parent[i];
                    }
                    while len > 0 {
                        top -= 1;
                        len -= 1;
                        pattern[top] = pattern[len];
                    }
                }
            }
            d[k] = y[k];
            y[k] = 0.0;
            for &i in &pattern[top..n] {
                let yi = y[i];
                y[i] = 0.0;
                let p2 = lp[i] + lnz[i];
                for p in lp[i]..p2 {
                    y[li[p]] -= lx[p] * yi;
                }
                let l_ki = yi / d[i];
                d[k] -= l_ki * yi;
                li[p2] = k;
                lx[p2] = l_ki;
                lnz[i] += 1;
            }
            let limit = drop_below[kk];
            if limit > 0.0 && d[k].abs() < limit {
                d[k] = -DROPPED;
                dropped.push(kk);
            }
            if d[k] == 0.0 || !d[k].is_finite() {
                return None;
            }
        }
        Some((
            Self {
                perm,
                lp,
                li,
                lx,
                d,
            },
            dropped,
        ))
    }

    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.d.len();
        let mut x: Vec<f64> = self.perm.iter().map(|&p| b[p]).collect();
        for j in 0..n {
            for p in self.lp[j]..self.lp[j + 1] {
                x[self.li[p]] -= self.lx[p] * x[j];
            }
        }
        for (xj, dj) in x.iter_mut().zip(&self.d) {
            *xj /= dj;
        }
        for j in (0..n).rev() {
            for p in self.lp[j]..self.lp[j + 1] {
                x[j] -= self.lx[p] * x[self.li[p]];
            }
        }
        let mut out = vec![0.0; n];
        for (k, &p) in self.perm.iter().enumerate() {
            out[p] = x[k];
        }
        out
    }
}

/// A factored saddle-point system `[A Jᵀ; J 0]`.
pub(crate) struct SaddleFactor {
    n: usize,
    ldl: Ldl,
    /// The unregularized (symmetrized) matrix, for refinement.
    k: Csc,
    /// Constraint rows (`0..m`) met by a rounding-size pivot: redundant.
    redundant: Vec<usize>,
}

impl SaddleFactor {
    /// Factor from the entries of `A` (`n × n`, symmetrized) and the rows
    /// of `J` (`m × n`). `None` when the matrix is not quasi-definite in
    /// this ordering (a non-positive body pivot, a zero pivot): use the
    /// dense path.
    pub(crate) fn new(n: usize, a: &[(usize, usize, f64)], j: &SparseRows) -> Option<Self> {
        let m = j.len();
        let mut diag = vec![0.0; n];
        let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(2 * a.len() + 4 * m * 12);
        for &(r, c, v) in a {
            if r == c {
                diag[r] += v;
                entries.push((r, c, v));
            } else {
                entries.push((r, c, 0.5 * v));
                entries.push((c, r, 0.5 * v));
            }
        }
        for (k, row) in j.rows.iter().enumerate() {
            for &(c, v) in row {
                entries.push((n + k, c, v));
                entries.push((c, n + k, v));
            }
        }
        let k = Csc::from_entries(n + m, entries);
        // Regularization scales: the body diagonal, and per constraint row
        // the diagonal estimate of J·A⁻¹·Jᵀ.
        let a_max = diag.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        if a_max <= 0.0 {
            return None;
        }
        let row_scale: Vec<f64> = j
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(c, v)| v * v / diag[c].abs().max(DELTA_A * a_max))
                    .sum::<f64>()
            })
            .collect();
        let mut k_reg = k.clone();
        k_reg.shift_diagonal(n, DELTA_A * a_max);
        let drop_below: Vec<f64> = (0..n + m)
            .map(|i| {
                if i < n {
                    0.0
                } else {
                    REDUNDANT_PIVOT * row_scale[i - n]
                }
            })
            .collect();
        let (ldl, dropped) = Ldl::factor(&k_reg, ordering(&k_reg, n), &drop_below)?;
        for (pos, &p) in ldl.perm.iter().enumerate() {
            let dk = ldl.d[pos];
            if (p < n && dk <= 0.0) || (p >= n && dk > 0.0) {
                return None;
            }
        }
        let mut redundant: Vec<usize> = dropped.into_iter().map(|p| p - n).collect();
        redundant.sort_unstable();
        Some(Self {
            n,
            ldl,
            k,
            redundant,
        })
    }

    /// The redundant constraint rows (ascending).
    pub(crate) fn redundant(&self) -> &[usize] {
        &self.redundant
    }

    /// One solve with the regularized factor (no refinement): the
    /// modified-Newton use.
    pub(crate) fn solve_approx(&self, rhs: &[f64]) -> Vec<f64> {
        self.ldl.solve(rhs)
    }

    /// Solve `[A Jᵀ; J 0]·[x; μ] = [top; bottom]` with iterative refinement
    /// against the unregularized matrix. `None` when the refinement does not
    /// reach the tolerance (an inconsistent or singular system).
    pub(crate) fn solve(&self, top: &[f64], bottom: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let b: Vec<f64> = top.iter().chain(bottom).copied().collect();
        let scale = b
            .iter()
            .fold(0.0_f64, |s, v| s.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let mut x = self.ldl.solve(&b);
        let residual = |x: &[f64]| -> (Vec<f64>, f64) {
            let kx = self.k.mul(x);
            let r: Vec<f64> = b.iter().zip(&kx).map(|(bi, ki)| bi - ki).collect();
            let res = r.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
            (r, res)
        };
        // Refine while the residual keeps falling (to the rounding floor),
        // then accept if it is small.
        let (mut r, mut res) = residual(&x);
        for _ in 0..REFINE_MAX {
            if res <= f64::EPSILON * scale {
                break;
            }
            let dx = self.ldl.solve(&r);
            let trial: Vec<f64> = x.iter().zip(&dx).map(|(a, d)| a + d).collect();
            let (tr, tres) = residual(&trial);
            if tres >= 0.5 * res {
                if tres < res {
                    (x, res) = (trial, tres);
                }
                break;
            }
            (x, r, res) = (trial, tr, tres);
        }
        if res <= REFINE_TOL * scale {
            let mu = x.split_off(self.n);
            Some((x, mu))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::RowFactor;

    /// A small saddle system with a redundant (duplicated) constraint row:
    /// the sparse solve matches the dense one on everything determinate and
    /// removes the duplicate.
    #[test]
    fn saddle_matches_dense_minimum_norm() {
        let a = vec![
            vec![2.0, 0.3, 0.0, 0.0],
            vec![0.3, 1.5, 0.0, 0.0],
            vec![0.0, 0.0, 3.0, -0.2],
            vec![0.0, 0.0, -0.2, 1.0],
        ];
        let j = vec![
            vec![1.0, 0.0, -1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.5],
            vec![2.0, 0.0, -2.0, 0.0],
        ];
        let top = [1.0, -2.0, 0.5, 3.0];
        let bottom = [0.1, 0.2, 0.2];
        let f = SaddleFactor::new(4, &triplets_from_dense(&a), &SparseRows::from_dense(&j, 4)).expect("quasi-definite");
        assert_eq!(f.redundant(), &[2]);
        let (x, mu) = f.solve(&top, &bottom).expect("consistent");
        let mut rows = Vec::new();
        for (i, ar) in a.iter().enumerate() {
            let mut r = ar.clone();
            r.extend(j.iter().map(|jr| jr[i]));
            rows.push(r);
        }
        for jr in &j {
            let mut r = jr.clone();
            r.resize(7, 0.0);
            rows.push(r);
        }
        let rhs: Vec<f64> = top.iter().chain(&bottom).copied().collect();
        let (dense, conflicts) = RowFactor::new(&rows, 7).solve(&rhs);
        assert!(conflicts.is_empty());
        for (s, d) in x.iter().zip(&dense) {
            assert!((s - d).abs() < 1e-12, "x: {s} vs {d}");
        }
        // The multipliers: the determinate part (Jᵀμ) exactly; the duplicate
        // row is removed (multiplier 0), its share carried by the original.
        for c in 0..4 {
            let ours: f64 = j.iter().zip(&mu).map(|(r, m)| r[c] * m).sum();
            let theirs: f64 = j.iter().zip(&dense[4..]).map(|(r, m)| r[c] * m).sum();
            assert!((ours - theirs).abs() < 1e-12, "Jᵀμ: {ours} vs {theirs}");
        }
        assert!(mu[2].abs() < 1e-90, "removed row carries {}", mu[2]);
        // An inconsistent right side is refused.
        assert!(f.solve(&top, &[0.1, 0.2, 0.3]).is_none());
    }

    /// A 30-body chain with body-coupling `A` (block tridiagonal, as the
    /// integrator's stiffness terms make it) and five rows per joint: the
    /// constrained ordering keeps the fill linear (`nnz(L)` within a small
    /// multiple of `nnz(K)` — eliminating all bodies first would fill in
    /// `J·A⁻¹·Jᵀ` densely), and the solve matches the dense one.
    #[test]
    fn chain_fill_stays_linear() {
        let bodies = 30;
        let n = 6 * bodies;
        let mut a: Triplets = Vec::new();
        for b in 0..bodies {
            for i in 0..6 {
                a.push((6 * b + i, 6 * b + i, 4.0 + 0.1 * i as f64));
                if b + 1 < bodies {
                    a.push((6 * b + i, 6 * (b + 1) + i, -0.5));
                    a.push((6 * (b + 1) + i, 6 * b + i, -0.5));
                }
            }
        }
        let mut rows = Vec::new();
        for b in 0..bodies {
            for r in 0..5 {
                let mut row = Vec::new();
                if b > 0 {
                    for c in 0..6 {
                        row.push((6 * (b - 1) + c, ((r + c) % 4) as f64 - 1.5));
                    }
                }
                for c in 0..6 {
                    row.push((6 * b + c, ((r * c + 1) % 5) as f64 - 2.0 + 0.1 * r as f64));
                }
                rows.push(row);
            }
        }
        let j = SparseRows { n_cols: n, rows };
        let m = j.len();
        let f = SaddleFactor::new(n, &a, &j).expect("sparse path");
        assert!(
            f.ldl.lx.len() < 4 * f.k.rowi.len(),
            "fill {} for {} entries",
            f.ldl.lx.len(),
            f.k.rowi.len()
        );
        let top: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let bottom: Vec<f64> = (0..m).map(|i| (i as f64 * 0.11).cos()).collect();
        let (x, mu) = f.solve(&top, &bottom).expect("solvable");
        let mut dense = vec![vec![0.0; n + m]; n + m];
        for &(r, c, v) in &a {
            dense[r][c] += v;
        }
        for (k, row) in j.rows.iter().enumerate() {
            for &(c, v) in row {
                dense[n + k][c] = v;
                dense[c][n + k] = v;
            }
        }
        let rhs: Vec<f64> = top.iter().chain(&bottom).copied().collect();
        let (reference, conflicts) = RowFactor::new(&dense, n + m).solve(&rhs);
        assert!(conflicts.is_empty());
        let worst = x
            .iter()
            .chain(&mu)
            .zip(&reference)
            .fold(0.0_f64, |w, (s, d)| w.max((s - d).abs()));
        assert!(worst < 1e-9, "sparse vs dense {worst:e}");
    }
}
