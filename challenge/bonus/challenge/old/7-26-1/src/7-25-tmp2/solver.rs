use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::lll::{self, VecLinearAlgebra, Lattice};
use ndarray::{Array1, Array2, Axis, concatenate};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::thread;
use std::cmp::min;

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
    //println!("strat Babai");
    let br = babai_nearest_vector_with_bkz(&M, &b_f64);
    //println!("end Babai");

    // 先转换为 Vec 再进行并行迭代
    let result: Vec<u64> = br.to_vec()
       .into_par_iter()
       .map(|x| (x.round() as u64).rem_euclid(q))
       .collect();
    Array1::from_vec(result)
}

/// 穷举系数 [-1, 0, 1] 的线性组合，寻找最短向量近似
pub fn coeff_vec(basis: &Vec<Vec<f64>>, _gs_norms: &[i128]) -> Option<Vec<f64>> {
    let n = basis.len();
    let mut best_vec = None;
    let mut best_norm = f64::MAX;

    // 枚举所有 3^n 种组合（±1, 0）
    let total = 3_usize.pow(n as u32);
    for i in 0..total {
        let mut coeffs = vec![0; n];
        let mut x = i;
        for j in 0..n {
            coeffs[j] = match x % 3 {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => unreachable!(),
            };
            x /= 3;
        }

        // 跳过全零系数
        if coeffs.iter().all(|&c| c == 0) {
            continue;
        }

        // 构造向量并求范数
        let mut v = vec![0.0; basis[0].len()];
        for (ci, bi) in coeffs.iter().zip(basis.iter()) {
            let scaled = bi.scalar_mult(*ci as f64);
            for j in 0..v.len() {
                v[j] += scaled[j];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm < best_norm {
            best_norm = norm;
            best_vec = Some(v);
        }
    }

    best_vec
}

/// 简化 BKZ 主函数（非递归、无枚举，仅用 LLL + Local SVP 替代）
pub fn bkz_reduce(lat: &Lattice, block_size: usize) -> Lattice {
    let mut basis = lat.basis.clone();
    let n = basis.len();

    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            let b_end = min(i + block_size, n);
            if b_end - i < 2 {
                continue;
            }
            let sub_basis = basis[i..b_end].to_vec();
            let mut sub_lat = Lattice { basis: sub_basis };

            // 子块LLL化
            if let Ok(reduced) = lll::lll(&sub_lat) {
                for j in 0..(b_end - i) {
                    basis[i + j] = reduced.basis[j].clone();
                }

                // Local coefficient sieving 替代 SVP 枚举
                let gs_norms = {
                    let gs = lll::gram_schmidt(&basis);
                    gs.iter().map(|v| v.iter().map(|x| x * x).sum::<f64>() as i128).collect::<Vec<_>>()
                };
                if let Some(v) = coeff_vec(&basis[i..b_end].to_vec(), &gs_norms[i..b_end]) {
                    basis[i] = v;
                    changed = true;
                }
            }
        }
    }

    Lattice { basis }
}

fn babai_nearest_vector_with_bkz(B: &Array2<f64>, t: &Array1<f64>) -> Array1<f64> {
    let lat = lll::Lattice::from_array2(B);
    let lat = bkz_reduce(&lat, 20);// 使用块大小为 20 的 BKZ
    let G = lll::gram_schmidt(&lat.basis);
    let mut b = t.to_vec();

    for i in (0..lat.basis.len()).rev() {
        let b_i = &lat.basis[i];
        let g_i = &G[i];
        let dot_b_g: f64 = b.par_iter().zip(g_i.par_iter()).map(|(x, y)| x * y).sum();
        let dot_g_g: f64 = g_i.par_iter().map(|x| x * x).sum();
        let coeff = (dot_b_g / dot_g_g).round();
        let projection = b_i.scalar_mult(coeff);
        b = b.par_iter().zip(projection.par_iter()).map(|(x, y)| x - y).collect();
    }

    Array1::from_vec(b)
}
