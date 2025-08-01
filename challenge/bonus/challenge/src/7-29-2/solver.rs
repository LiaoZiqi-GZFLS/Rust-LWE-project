use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::lll::{self, VecLinearAlgebra};
use ndarray::{Array1, Array2, Axis, concatenate, s};
use std::ops::Rem;

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
    let br = babai_nearest_vector(&M, &b_f64);

    // this solution requires solving a tall matrix under modulo q
    // the solutions are not unique, so we take the one that is closest to the original
    // vector b, which is the result of the LWE problem
    let s_pred = br.into_iter()
        .map(|x| (x.round() as u64).rem_euclid(q))
        .collect::<Array1<u64>>();

    // Enumeration to approximate the original secret vector
    let s_pred = enumerate_approximate(&s_pred, n, q, &A.t().to_owned(), b);
    println!("Recovered secret vector: {:?}", s_pred);
    let s_recovered = solve_mod_linear_system(A, &s_pred, q).unwrap();
    println!("Correct? {}", is_key_correct(&A, b, &s_recovered, q, alpha));

    s_recovered
}

fn babai_nearest_vector(B: &Array2<f64>, t: &Array1<f64>) -> Array1<f64> {
    let B0 = lll::Lattice::from_array2(B);
    let G = lll::gram_schmidt(&B0.basis);
    let B = lll::int_lll(&B0).unwrap();
    // 枚举校准，搜索半径设为最短向量长度的1.5倍
    let calibrated_result = lll::enumerate_and_calibrate(
        &B0, 
        &B, 
        B.get_min_norm_from_basis() * 1.5
    );

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

fn is_key_correct(
    A: &Array2<u64>,
    b: &Array1<u64>,
    s_pred: &Array1<u64>,
    q: u64,
    alpha: f64,
) -> bool {
    // 计算 A s_pred mod q
    let As_mod_q = A.dot(s_pred).mapv(|x| x % q);

    fn center_mod_q(x: i64, q: i64) -> i64 {
        let r = ((x % q) + q) % q;
        if r > q / 2 { r - q } else { r }
    }

    let e_vec = As_mod_q.iter()
        .zip(b.iter())
        .map(|(&as_i, &b_i)| {
            let diff = b_i as i64 - as_i as i64;
            center_mod_q(diff, q as i64)
        })
        .collect::<Vec<i64>>();

    let l2_norm = (e_vec.iter().map(|&e| (e as f64).powi(2)).sum::<f64>()).sqrt();
    let expected = alpha * q as f64 * (b.len() as f64).sqrt();

    /*println!(
        "解密误差范数 = {:.4}, 期望最大噪声 ≈ {:.4}",
        l2_norm, expected
    );*/

    l2_norm < expected
}


fn center_mod_q(x: i64, q: i64) -> i64 {
    let r = ((x % q) + q) % q;
    if r > q / 2 { r - q } else { r }
}

pub fn compute_error(A: &Array2<u64>, s: &Array1<u64>, b: &Array1<u64>, q: u64) -> f64 {
    let As = A.dot(s).mapv(|x| x % q);
    let mut error = 0.0;
    for i in 0..As.len() {
        let diff = b[i] as i64 - As[i] as i64;
        let centered = center_mod_q(diff, q as i64);
        error += (centered as f64).powi(2);
    }
    error.sqrt()
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