use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::lll::{self, VecLinearAlgebra};
use ndarray::{Array1, Array2, Axis, concatenate};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::thread;

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
    /*
    let num_cores = thread::available_parallelism().expect("Failed to get number of cores");
    let pool = ThreadPoolBuilder::new()
       .num_threads(num_cores.get())
       .build()
       .expect("Failed to build thread pool");
    */
    let q_f64 = q as f64;
    // 并行初始化 pIm 矩阵
    let pIm_vec: Vec<Vec<f64>> = (0..m).into_par_iter()
       .map(|i| {
            (0..m).map(|j| q_f64 * (i == j) as u64 as f64).collect::<Vec<f64>>()
        })
       .collect();
    let pIm = Array2::from_shape_vec((m, m), pIm_vec.into_iter().flatten().collect()).unwrap();

    // 手动实现并行映射
    let A_f64_vec: Vec<f64> = A.iter().cloned().collect::<Vec<u64>>()
       .into_par_iter()
       .map(|x| x as f64)
       .collect();
    let A_f64 = Array2::from_shape_vec(A.raw_dim(), A_f64_vec).unwrap();

    let M = concatenate(Axis(1), &[pIm.view(), A_f64.view()]).unwrap();
    let b_f64 = b.mapv(|x| x as f64);
    let br = babai_nearest_vector(&M, &b_f64);

    // 先转换为 Vec 再进行并行迭代
    let result: Vec<u64> = br.to_vec()
       .into_par_iter()
       .map(|x| (x.round() as u64).rem_euclid(q))
       .collect();
    Array1::from_vec(result)
}

fn babai_nearest_vector(B: &Array2<f64>, t: &Array1<f64>) -> Array1<f64> {
    let B = lll::Lattice::from_array2(B);
    let G = lll::gram_schmidt(&B.basis);
    let B = lll::lll(&B).unwrap();
    let mut b = t.to_vec();

    // 这里由于迭代有依赖关系，不能完全并行化，但可以并行计算点积等操作
    for i in (0..B.basis.len()).rev() {

        let b_i = &B.basis[i];
        let g_i = &G[i];
        // 并行计算点积
        let dot_b_g: f64 = b.par_iter().zip(g_i.par_iter()).map(|(x, y)| x * y).sum();
        let dot_g_g: f64 = g_i.par_iter().map(|x| x * x).sum();
        // Calculate coefficient: round to nearest integer
        let coeff = (dot_b_g / dot_g_g).round();
        // Subtract the projection and store result back in b
        let projection = b_i.scalar_mult(coeff);
        b = b.par_iter().zip(projection.par_iter()).map(|(x, y)| x - y).collect();
    }

    Array1::from_vec(b)
}