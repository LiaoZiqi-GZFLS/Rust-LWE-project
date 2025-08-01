use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::lll::{self, VecLinearAlgebra};
use ndarray::{Array1, Array2, Axis, concatenate, s};
use ndarray_linalg::{QR, Scalar};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::rand_distr::{Normal, Uniform};
use itertools::Itertools;
use rayon::prelude::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

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
    let A_f64 = A.mapv(|x| x as f64);
    let M = concatenate(Axis(1), &[pIm.view(), A_f64.view()]).unwrap();
    let b_f64 = b.mapv(|x| x as f64);
    //let br = babai_nearest_vector(&M, &b_f64);

    // this solution requires solving a tall matrix under modulo q
    // the solutions are not unique, so we take the one that is closest to the original
    // vector b, which is the result of the LWE problem

    // Enumeration to approximate the original secret vector

    println!("{:?}", A.shape());

    let mut s_pred = b.mapv(|x| x % q);
    let s_recovered = solve_mod_linear_system(A, &s_pred, q).unwrap();
    if is_key_correct(&A, b, &s_recovered, q, alpha, 0.01, false) {
        println!("Recovered secret vector: {:?}", s_pred);
        return s_recovered;
    }
    if n<30 {
        let b_s = b.clone().slice_move(s![..]);
        let A_s = A.clone().slice_move(s![.., ..]);
        //let b_pred = enumerate_approximate_parallel(&b_s, n, q, &A_s, &b_s);
        let b_pred = enumerate_prefix_parallel(&b_s, n, q, &A_s, &b_s);
        return b_pred;
        
    }else{
        return b.clone();
    }

    //let s_pred = enumerate_approximate(b, n, q, A, b);
    //let s_pred = greedy_approximate(&b, n, q, A, b, 10000*n as u64);
    //let s_pred = b.mapv(|x| x % q);

    //println!("Recovered secret vector: {:?}", s_pred);
    let s_recovered = solve_mod_linear_system(A, &s_pred, q).unwrap();
    println!("Correct? {}", is_key_correct(&A, b, &s_recovered, q, alpha, 0.01, true));

    let error_norm = compute_error(&A, &s_recovered, b, q, alpha);
    println!("Error norm: {:.4}", error_norm);


    //b.clone()
    //s_recovered
    s_pred
}

pub fn enumerate_approximate_parallel(
    s_pred: &Array1<u64>,
    n: usize,
    q: u64,
    A: &Array2<u64>,
    b: &Array1<u64>,
) -> Array1<u64> {
    let alpha = 0.005;
    let m = b.len();

    let best = Arc::new(Mutex::new(s_pred.clone()));
    let best_err = Arc::new(Mutex::new({
        let s_recovered = solve_mod_linear_system(A, s_pred, q).unwrap();
        compute_error(A, &s_recovered, b, q, alpha)
    }));
    let found_correct = Arc::new(AtomicBool::new(false));

    fn dfs_parallel(
        i: usize,
        n: usize,
        current: &mut Array1<u64>,
        best: &Arc<Mutex<Array1<u64>>>,
        best_err: &Arc<Mutex<f64>>,
        found_correct: &Arc<AtomicBool>,
        A: &Array2<u64>,
        b: &Array1<u64>,
        q: u64,
        m: usize,
        alpha: f64,
    ) {
        if found_correct.load(Ordering::Relaxed) {
            return;
        }

        if i == current.len() {
            let s_recovered = solve_mod_linear_system(A, current, q).unwrap();
            let err = compute_error(A, &s_recovered, b, q, alpha);

            let mut best_guard = best.lock().unwrap();
            let mut err_guard = best_err.lock().unwrap();

            if err < *err_guard {
                *err_guard = err;
                *best_guard = current.clone();
            }

            if is_key_correct(A, b, &s_recovered, q, alpha, 0.01, false) {
                *best_guard = current.clone();
                found_correct.store(true, Ordering::Relaxed);
            }

            return;
        }

        let orig = current[i];
        [-1i64, 0, 1].into_par_iter().for_each(|delta| {
            if found_correct.load(Ordering::Relaxed) {
                return;
            }

            let new_val = ((orig as i64 + delta).rem_euclid(q as i64)) as u64;
            let mut next = current.clone(); // clone to avoid mutable conflict
            next[i] = new_val;

            dfs_parallel(
                i + 1,
                n,
                &mut next,
                best,
                best_err,
                found_correct,
                A,
                b,
                q,
                m,
                alpha,
            );
        });
    }

    let mut current = s_pred.clone();
    dfs_parallel(
        0,
        n,
        &mut current,
        &best,
        &best_err,
        &found_correct,
        A,
        b,
        q,
        m,
        alpha,
    );

    Arc::try_unwrap(best).unwrap().into_inner().unwrap()
}

pub fn enumerate_prefix_parallel(
    s_pred: &Array1<u64>,
    n: usize,
    q: u64,
    A: &Array2<u64>, // full A: (m, full_n)
    b: &Array1<u64>,
) -> Array1<u64> {
    let alpha = 0.005;
    let full_n = s_pred.len();

    let result = Arc::new(Mutex::new(None));
    let found_correct = Arc::new(AtomicBool::new(false));

    let A_small = A.slice(s![0..n, 0..n]).to_owned();
    let b_small = b.slice(s![0..n]).to_owned();

    fn dfs_parallel(
        i: usize,
        n: usize,
        current: &mut Array1<u64>,
        result: &Arc<Mutex<Option<Array1<u64>>>>,
        found_correct: &Arc<AtomicBool>,
        A_full: &Array2<u64>,
        b_full: &Array1<u64>,
        A_small: &Array2<u64>,
        b_small: &Array1<u64>,
        q: u64,
        full_n: usize,
        alpha: f64,
    ) {
        if found_correct.load(Ordering::Relaxed) {
            return;
        }

        if i == n {
            let partial_recovered = solve_mod_linear_system(A_small, current, q);
            if partial_recovered.is_none() {
                return;
            }

            let partial = partial_recovered.unwrap();

            let mut s_full = Array1::<u64>::zeros(full_n);
            for j in 0..n {
                s_full[j] = partial[j];
            }
            for j in n..full_n {
                s_full[j] = current[j];
            }

            if is_key_correct(A_full, b_full, &s_full, q, alpha, 0.01, false) {
                *result.lock().unwrap() = Some(s_full.clone());
                found_correct.store(true, Ordering::Relaxed);
            }

            return;
        }

        let orig = current[i];
        [-1i64, 0, 1].into_par_iter().for_each(|delta| {
            if found_correct.load(Ordering::Relaxed) {
                return;
            }

            let new_val = ((orig as i64 + delta).rem_euclid(q as i64)) as u64;
            let mut next = current.clone();
            next[i] = new_val;

            dfs_parallel(
                i + 1,
                n,
                &mut next,
                result,
                found_correct,
                A_full,
                b_full,
                A_small,
                b_small,
                q,
                full_n,
                alpha,
            );
        });
    }

    let mut current = s_pred.clone();
    dfs_parallel(
        0,
        n,
        &mut current,
        &result,
        &found_correct,
        A,
        b,
        &A_small,
        &b_small,
        q,
        full_n,
        alpha,
    );

    // 成功返回解，否则返回原始猜测
    match Arc::try_unwrap(result).unwrap().into_inner().unwrap() {
        Some(sol) => sol,
        None => s_pred.clone(),
    }
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
        1.5
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
    assert_eq!(
        A.ncols(),
        s.len(),
        "A.ncols() must equal s.len(): {} != {}",
        A.ncols(),
        s.len()
    );
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
