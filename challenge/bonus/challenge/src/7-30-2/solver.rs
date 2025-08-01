use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::lll::{self, VecLinearAlgebra};
use ndarray::{Array1, Array2, Axis, concatenate, s};
use ndarray_linalg::{QR, Scalar};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::rand_distr::{Normal, Uniform};
use std::cmp::Ordering;
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
    let pIm = Array2::from_shape_fn((m, m), |(i, j)| q as f64 * (i == j) as u64 as f64);
    let A_f64 = A.mapv(|x| x as f64);
    let M = concatenate(Axis(1), &[pIm.view(), A_f64.view()]).unwrap();
    let b_f64 = b.mapv(|x| x as f64);
    //let br = babai_nearest_vector(&M, &b_f64);

    // this solution requires solving a tall matrix under modulo q
    // the solutions are not unique, so we take the one that is closest to the original
    // vector b, which is the result of the LWE problem

    // Enumeration to approximate the original secret vector
    //let s_pred = enumerate_approximate(b, n, q, &A.t().to_owned(), b);
    let s_pred = greedy_approximate(&b, n, q, A, b, q/2 as u64);

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

pub fn greedy_approximate(
    s_pred: &Array1<u64>,
    n: usize,
    q: u64,
    A: &Array2<u64>,
    b: &Array1<u64>,
    max_rounds: u64,
) -> Array1<u64> {
    let alpha = 0.005;          // 与之前保持一致
    let m = b.len();
    let mut s = s_pred.clone(); // 当前向量
    let mut best = s.clone();

    let mut s_recovered = solve_mod_linear_system(A, &s, q).unwrap();

    let mut best_err = compute_error(A, &s_recovered, b, q, alpha);

    // 若初始即正确直接返回
    if is_key_correct(A, b, &s_recovered, q, alpha, 0.01, false) {
        return s;
    }

    // 最多做 max_rounds 次“单维移动”
    for _round in 0..max_rounds {
        let mut improved = false;
        let mut turns = 0;

        // 先计算当前误差
        let base_recovered = solve_mod_linear_system(A, &s, q).unwrap();
        //let base_err = compute_error(A, &s, b, q);
        let base_err = compute_error(A, &base_recovered, b, q, alpha);

        // 记录 (维度索引, delta, 新误差)
        let mut best_move: Option<(usize, i64, f64)> = None;

        // 遍历每一维，三个方向 {-1, 0, 1}
        for i in 0..n {
            let len = (q as f64 * alpha as f64).round() as i64 + 1i64;
            let range_vec: Vec<i64> = (-len..=len).collect();
            for &delta in &range_vec {
                let mut cand = s.clone();
                cand[i] = ((s[i] as i64 + delta).rem_euclid(q as i64)) as u64;
                let cand_recovered = solve_mod_linear_system(A, &cand, q).unwrap();
                //let err = compute_error(A, &cand, b, q);
                let err = compute_error(A, &cand_recovered, b, q, alpha);

                if err < base_err {
                    // 记录误差下降最大的那一步
                    match best_move {
                        None => best_move = Some((i, delta, err)),
                        Some((_pi, _pd, pe)) => {
                            if err < pe {
                                best_move = Some((i, delta, err))
                            }
                        }
                    }
                }
            }
        }

        // 如果有改进，执行最佳移动
        if let Some((i, delta, new_err)) = best_move {
            s[i] = ((s[i] as i64 + delta).rem_euclid(q as i64)) as u64;
            if new_err < best_err {
                best_err = new_err;
                best = s.clone();
                println!(
                    "Round improved: dim {}, delta {}, new error {:.4}",
                    i, delta, new_err
                );
            }
            improved = true;

            let current_recovered = solve_mod_linear_system(A, &best, q).unwrap();
            // 立即验证是否成功
            if is_key_correct(A, b, &current_recovered, q, alpha, 0.01, false) {
                return s;
            }
        }

        // 如果本轮没有任何维度能带来改进，就提前退出
        if !improved {
            let mut rng = thread_rng();
            let sigma = alpha * q as f64 * 0.5; // 使用 alpha * q / 2 作为误差标准差

            let uniform = Uniform::new(0, q);
            let normal = Normal::new(0.0, sigma).unwrap();

            let e: Array1<f64> = Array1::random_using(m, normal, &mut rng);

            let mut b_new = b.clone();

            for (bi, ei) in b_new.iter_mut().zip(e.iter()) {
                let error = ei.round() as i64;
                let val = (*bi as i64 + error).rem_euclid(q as i64);
                *bi = val as u64;
            }

            let current_recovered = solve_mod_linear_system(A, &b_new, q).unwrap();
            let err = compute_error(A, &current_recovered, b, q, alpha);

            if err < best_err {
                best_err = err;
                best = b_new.clone();
                println!("No improvement, but found better candidate with error {:.4}", err);
            }else{
                turns += 1;
                if turns >= 1000*n {
                    break;
                }
            }


            //break;
        }
    }

    best
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
    let s_recovered = solve_mod_linear_system(&A.t().to_owned(), &best, q).unwrap();
    let mut best_err = compute_error(&A.t().to_owned(), &s_recovered, b, q, alpha);
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
            let err = compute_error(&A.t().to_owned(), &current_recovered, b, q, alpha);
            if err < *best_err {
                *best_err = err;
                *best = current.clone();
            }

            if is_key_correct(&A.t().to_owned(), b, &current_recovered, q, alpha, 0.01, false) {
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
