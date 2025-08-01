use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::lll::{self, VecLinearAlgebra};
use ndarray::{Array1, Array2, Axis, stack, concatenate, s};
use rayon::prelude::*;

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
    let A0 = A.clone();
    let b0 = b.clone();
    let pIm = Array2::from_shape_fn((m, m), |(i, j)| q as f64 * (i == j) as u64 as f64);
    let A = A.mapv(|x| x as f64);
    let M = concatenate(Axis(1), &[pIm.view(), A.view()]).unwrap();
    let b = b.mapv(|x| x as f64);
    let br = babai_nearest_vector(&M, &b);

    // this solution requires solving a tall matrix under modulo q
    // the solutions are not unique, so we take the one that is closest to the original
    // vector b, which is the result of the LWE problem
    let mut b_pred = b0.clone();
    let mut s_recovered = solve_mod_linear_system(&A0, &b_pred, q).unwrap();
    if is_key_correct(&A0, &b0, &s_recovered, q, alpha, 0.01, true) {
        return s_recovered;
    }

    if n < 20 {
        // For small n, we can use statistical attack
        //return statistical_attack(&A0, &b0, q);
        //let b_pred = brute_force_lwe_solver(&A0, &b0, q, n / 2, n / 2).unwrap();
        let b_pred = enumerate_approximate(&b_pred, n, q, &A0.t().to_owned(), &b0);
        let s_pred = solve_mod_linear_system(&A0, &b_pred, q).unwrap();
        if is_key_correct(&A0, &b0, &s_pred, q, alpha, 0.01, true) {
            return b_pred;
        }else {
            println!("Small error attack failed, falling back to Babai's nearest vector");
        }
    }

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

pub fn is_key_correct(
    a: &Array2<u64>,
    b: &Array1<u64>,
    s_pred: &Array1<u64>,
    q: u64,
    alpha: f64,
    rtol: f64, 
    echo: bool,
) -> bool {
    // 1. 计算 A·s_pred mod q
    let as_mod_q = a.dot(s_pred).mapv(|x| x % q);

    // 2. 把差值拉到 (-q/2, q/2]
    fn center(x: i64, q: i64) -> i64 {
        let r = ((x % q) + q) % q;
        if r > q / 2 { r - q } else { r }
    }

    // 3. 计算误差向量及其 ℓ₂ 范数
    let e_vec: Array1<i64> = as_mod_q
        .iter()
        .zip(b.iter())
        .map(|(&as_i, &b_i)| center(b_i as i64 - as_i as i64, q as i64))
        .collect();

    let l2_sq: f64 = e_vec.iter().map(|&e| (e as f64).powi(2)).sum();
    let l2_norm = l2_sq.sqrt();

    // 4. 期望阈值：alpha*q*sqrt(m)
    let threshold = alpha * q as f64 * (b.len() as f64).sqrt();
    let added = if a.shape()[1] < 20 {
        1.0
    } else if a.shape()[1] < 45 {
        2.0
    }else {
        3.0
    };
    let tolerance = rtol * threshold + added as f64;

    if echo {
        println!("l2_norm = {:.3}, threshold = {:.3}, tolerance = {:.3}", l2_norm, threshold, tolerance);
        //println!("{},{}",a.shape()[0], a.shape()[1]);
    }

    // 5. 可放宽一点防止浮点或舍入误差
    //l2_norm <= threshold
    (l2_norm - threshold).abs() <= tolerance
}


fn center_mod_q(x: i64, q: i64) -> i64 {
    let r = ((x % q) + q) % q;
    if r > q / 2 { r - q } else { r }
}


/// 计算 LWE 残差 ‖b - A·s‖²
/// 先把残差 mod q 再映射到 [-q/2, q/2) 区间，避免环绕误差

pub fn compute_error(
    A: &Array2<u64>,
    s: &Array1<u64>,
    b: &Array1<u64>,
    q: u64,
    alpha: f64,
) -> f64 {
       // 1. 计算 A·s_pred mod q
    let as_mod_q = A.dot(s).mapv(|x| x % q);

    // 2. 把差值拉到 (-q/2, q/2]
    fn center(x: i64, q: i64) -> i64 {
        let r = ((x % q) + q) % q;
        if r > q / 2 { r - q } else { r }
    }

    // 3. 计算误差向量及其 ℓ₂ 范数
    let e_vec: Array1<i64> = as_mod_q
        .iter()
        .zip(b.iter())
        .map(|(&as_i, &b_i)| center(b_i as i64 - as_i as i64, q as i64))
        .collect();

    let l2_sq: f64 = e_vec.iter().map(|&e| (e as f64).powi(2)).sum();
    let l2_norm = l2_sq.sqrt();

    // 4. 期望阈值：alpha*q*sqrt(m)
    let threshold = alpha * q as f64 * (b.len() as f64).sqrt();

    // 5. 可放宽一点防止浮点或舍入误差
    //l2_norm <= threshold
    (l2_norm - threshold).abs()
}


fn mod_inv(a: i64, m: i64) -> Option<i64> {
    // 扩展欧几里得法求模逆
    let (mut t, mut newt) = (0, 1);
    let (mut r, mut newr) = (m, a.rem_euclid(m));
    while newr != 0 {
        let quotient = r / newr;
        t = t - quotient * newt;
        std::mem::swap(&mut t, &mut newt);
        r = r - quotient * newr;
        std::mem::swap(&mut r, &mut newr);
    }
    if r > 1 {
        return None; // 不可逆
    }
    if t < 0 {
        t += m;
    }
    Some(t)
}

// A: m x n 矩阵，y: m 维向量，q: 模数
// 返回 n 维解向量
fn solve_mod_linear_system(A: &Array2<u64>, y: &Array1<u64>, q: u64) -> Option<Array1<u64>> {
    let m = A.shape()[0];
    let n = A.shape()[1];
    //println!("Solving {}x{} system div {}", m, n, y.len());
    assert_eq!(y.len(), m);

    // 转成i64矩阵便于计算
    let mut mat = Array2::<i64>::zeros((m, n + 1));
    for i in 0..m {
        for j in 0..n {
            mat[[i, j]] = A[[i, j]] as i64 % q as i64;
        }
        mat[[i, n]] = y[i] as i64 % q as i64;
    }

    // 高斯消元（模 q）
    let modulus = q as i64;

    let mut rank = 0;
    for col in 0..n {
        // 找主元
        let mut pivot = None;
        for row in rank..m {
            if mat[[row, col]] % modulus != 0 {
                pivot = Some(row);
                break;
            }
        }
        if pivot.is_none() {
            continue; // 此列全零，跳过
        }
        let pivot = pivot.unwrap();

        // 交换当前行与rank行
        if pivot != rank {
            for c in col..=n {
                let tmp = mat[[rank, c]];
                mat[[rank, c]] = mat[[pivot, c]];
                mat[[pivot, c]] = tmp;
            }
        }

        // 主元归一
        let inv = mod_inv(mat[[rank, col]], modulus)?;
        for c in col..=n {
            mat[[rank, c]] = (mat[[rank, c]] * inv).rem_euclid(modulus);
        }

        // 消去其他行
        for r in 0..m {
            if r != rank && mat[[r, col]] != 0 {
                let factor = mat[[r, col]];
                for c in col..=n {
                    mat[[r, c]] = (mat[[r, c]] - factor * mat[[rank, c]]).rem_euclid(modulus);
                }
            }
        }
        rank += 1;
        if rank == m {
            break;
        }
    }

    

    // 解向量
    let mut x = vec![0u64; n];
    for i in 0..n {
        x[i] = mat[[i, n]] as u64;
    }
    
    let a = Some(Array1::from_vec(x));

    a
}


/// 执行统计攻击，逐列猜测 secret 中的每个 s_j 值
pub fn statistical_attack(
    A: &Array2<u64>,
    b: &Array1<u64>,
    q: u64,
) -> Array1<u64> {
    let m = A.nrows();
    let n = A.ncols();

    // 并行猜测每个 s_j ∈ [0, q)
    let s = (0..n).into_par_iter().map(|j| {
        let a_col = A.column(j);
        let mut best_score = std::f64::INFINITY;
        let mut best_guess = 0;

        for guess in 0..q {
            // 统计 b_i - a_ij * guess mod q
            let mut counts = vec![0u32; q as usize];

            for i in 0..m {
                let prod = a_col[i] * guess;
                let residue = (b[i] + q - prod % q) % q;
                counts[residue as usize] += 1;
            }

            // 计算分布的熵/方差作为评价标准（越集中越好）
            let variance = compute_variance(&counts);
            if variance < best_score {
                best_score = variance;
                best_guess = guess;
            }
        }

        best_guess
    }).collect();

    Array1::from_vec(s)
}

/// 计算直方图的离散方差
fn compute_variance(counts: &[u32]) -> f64 {
    let total: u32 = counts.iter().sum();
    if total == 0 { return f64::INFINITY; }

    let mean = total as f64 / counts.len() as f64;
    counts.iter()
        .map(|&c| {
            let diff = c as f64 - mean;
            diff * diff
        })
        .sum::<f64>() / counts.len() as f64
}


/// 最简洁暴力实现：枚举 secret 的前缀和后缀，检查是否满足误差 ≤ 1
pub fn brute_force_lwe_solver(
    A: &Array2<u64>,
    b: &Array1<u64>,
    q: u64,
    prefix_len: usize,
    suffix_len: usize,
) -> Option<Array1<u64>> {
    let n = A.ncols();
    let q_values: Vec<u64> = (0..q).collect();

    let prefix_space = cartesian_product(&q_values, prefix_len);
    let suffix_space = cartesian_product(&q_values, suffix_len);

    prefix_space.into_par_iter().find_map_first(|prefix| {
        let suffix_start = prefix_len;
        let suffix_end = suffix_start + suffix_len;

        for suffix in &suffix_space {
            let mut s = vec![0u64; n];
            s[0..prefix_len].copy_from_slice(&prefix[..]);
            s[suffix_start..suffix_end].copy_from_slice(&suffix[..]);

            if check_small_error(A, &Array1::from(s.clone()), b, q) {
                return Some(Array1::from(s));
            }
        }
        None
    })
}

fn cartesian_product(values: &[u64], k: usize) -> Vec<Vec<u64>> {
    let mut result = vec![vec![]];
    for _ in 0..k {
        result = result.into_iter().flat_map(|prefix| {
            values.iter().map(move |&x| {
                let mut new = prefix.clone();
                new.push(x);
                new
            })
        }).collect();
    }
    result
}

fn check_small_error(A: &Array2<u64>, s: &Array1<u64>, b: &Array1<u64>, q: u64) -> bool {
    let m = A.nrows();
    for i in 0..m {
        let mut sum = 0u64;
        for j in 0..A.ncols() {
            sum = (sum + A[[i, j]] * s[j]) % q;
        }
        let diff = (b[i] + q - sum) % q;
        let centered = if diff > q / 2 { diff as i64 - q as i64 } else { diff as i64 };
        if centered.abs() > 1 {
            return false;
        }
    }
    true
}

fn enumerate_approximate(
    s_pred: &Array1<u64>,
    n: usize,
    q: u64,
    A: &Array2<u64>,
    b: &Array1<u64>,
) -> Array1<u64> {
    let alpha = 0.005; // 可调参数，也可以作为函数参数传入
    let m = b.len();
    let mut best = s_pred.clone();
    let mut best_err = compute_error(A, &best, b, q);
    let mut current = s_pred.clone();
    let mut found_correct = false;

    fn dfs(
        i: usize,
        n: usize,
        current: &mut Array1<u64>,
        best: &mut Array1<u64>,
        best_err: &mut f64,
        found_correct: &mut bool,
        A: &Array2<u64>,
        b: &Array1<u64>,
        q: u64,
        m: usize,
        alpha: f64,
    ) {
        if *found_correct {
            println!("Found correct key, exiting early.");
            return; // 如果已经找到正确密钥，提前退出
        }

        if i == n {
            let current_recovered = solve_mod_linear_system(&A.t().to_owned(), current, q).unwrap();
            let err = compute_error(&A.t().to_owned(), &current_recovered, b, q);
            if err < *best_err {
                *best_err = err;
                *best = current.clone();
            }

            if is_key_correct(&A.t().to_owned(), b, &current_recovered, q, alpha) {
                *best = current.clone();
                *found_correct = true;
            }
            return;
        }

        let orig = current[i];
        for delta in [-1i64, 0, 1] {
            let new_val = ((orig as i64 + delta).rem_euclid(q as i64)) as u64;
            current[i] = new_val;
            dfs(i + 1, n, current, best, best_err, found_correct, A, b, q, m, alpha);
        }
        current[i] = orig;
    }

    dfs(
        0,
        n,
        &mut current,
        &mut best,
        &mut best_err,
        &mut found_correct,
        A,
        b,
        q,
        m,
        alpha,
    );

    best
}