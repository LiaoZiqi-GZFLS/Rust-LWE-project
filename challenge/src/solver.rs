use std::ops::Div;

use crate::lll::{self, VecLinearAlgebra};
use ndarray::{Array1, Array2, Axis, concatenate};

/// Solves the LWE problem.
///
/// # Arguments
///
/// * `n` - Secret dimension
/// * `m` - Number of samples
/// * `q` - Modulus
/// * `alpha` - Relative error size
/// * `A` - Matrix of dimensions m x n (mod q)
/// * `b` - Vector of length m (mod q)
///
/// # Returns
///
/// * `Array1<u64>` - Recovered secret vector s of length n
pub(crate) fn solve_lwe(
    n: usize,
    m: usize,
    q: u64,
    alpha: f64,
    A: &Array2<u64>,
    b: &Array1<u64>,
) -> Array1<u64> {
    let pIm = Array2::from_shape_fn((m, m), |(i, j)| q as f64 * (i == j) as u64 as f64);
    let A = A.mapv(|x| x as f64);
    let M = concatenate(Axis(1), &[pIm.view(), A.view()]).unwrap();
    let b = b.mapv(|x| x as f64);
    let br = babai_nearest_vector(&M, &b);

    // solve A * s = br mod q

    br.into_iter()
        .map(|x| (x.round() as u64).rem_euclid(q))
        .collect::<Array1<u64>>()
}

fn babai_nearest_vector(B: &Array2<f64>, t: &Array1<f64>) -> Array1<f64> {
    let B = lll::Lattice::from_array2(B);
    let G = lll::gram_schmidt(&B.basis);
    let B = lll::int_lll(&B).unwrap();
    let mut b = t.to_vec();

    // Iterating in reverse from len-1 down to 0
    for i in (0..B.basis.len()).rev() {
        let b_i = &B.basis[i];
        let g_i = &G[i];
        // Calculate coefficient: round to nearest integer
        let coeff = (b.dot(g_i) / g_i.dot(g_i)).round();
        // Subtract the projection and store result back in b
        b = b.sub(&b_i.scalar_mult(coeff).as_slice());
    }

    Array1::from_vec(b)
}
