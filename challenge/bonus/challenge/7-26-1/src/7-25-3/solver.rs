use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::lll::{self, VecLinearAlgebra};
use crate::bkz;
use ndarray::{s, Array1, Array2, Axis, concatenate};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::thread;
use num::{BigInt, ToPrimitive, rational::Ratio};

/// Solves the LWE problem using BKZ lattice basis reduction and Babai's nearest plane algorithm.
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
    
    // 构建扩展格基矩阵 M = [qI | A]
    let pIm = Array2::eye(m) * q_f64;
    
    // 将 A 转换为 f64 类型
    let A_f64 = A.mapv(|x| x as f64);
    
    // 构建格基矩阵 M = [qI | A]
    let M = concatenate(Axis(1), &[pIm.view(), A_f64.view()]).unwrap();
    
    // 应用 BKZ 约化
    let mut basis = convert_ndarray_to_rational(&M);
    let block_size = 40; // BKZ 块大小，可以根据需要调整
    let max_iterations = 5; // BKZ 最大迭代次数
    basis = bkz::bkz(&basis, block_size, max_iterations);
    
    // 转换回 ndarray 格式
    let reduced_basis = convert_rational_to_ndarray(&basis);
    
    // 使用 Babai 最近平面算法找到近似向量
    // 构建目标向量 t：长度为 m + n，前 m 个元素为 b 的值，后 n 个元素为 0
    let b_f64 = b.mapv(|x| x as f64); // 长度 m
    let zeros = Array1::zeros(n); // 长度 n
    let t = concatenate(Axis(0), &[b_f64.view(), zeros.view()]).unwrap(); // 拼接为长度 m + n 的向量

    // 使用 Babai 最近平面算法找到近似向量（此时 t 的长度与格基列数匹配）
    let br = babai_nearest_plane(&reduced_basis, &t);
    
    // 提取秘密向量
    let s = extract_secret(&br, m, n, q);
    
    s
}

/// 将 ndarray 矩阵转换为 Rational 类型的向量表示
fn convert_ndarray_to_rational(matrix: &Array2<f64>) -> Vec<Vec<num::rational::Ratio<num::BigInt>>> {
    let (rows, cols) = matrix.dim();
    let mut result = Vec::with_capacity(rows);
    
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            let num = matrix[[i, j]];
            // 转换浮点数为有理数
            let (numerator, denominator) = float_to_rational(num);
            row.push(num::rational::Ratio::new(
                num::BigInt::from(numerator),
                num::BigInt::from(denominator)
            ));
        }
        result.push(row);
    }
    
    result
}

/// 将 Rational 类型的向量表示转换回 ndarray 矩阵
fn convert_rational_to_ndarray(basis: &Vec<Vec<num::rational::Ratio<num::BigInt>>>) -> Array2<f64> {
    let rows = basis.len();
    let cols = if rows > 0 { basis[0].len() } else { 0 };
    
    let mut data = Vec::with_capacity(rows * cols);
    
    for row in basis {
        for value in row {
            data.push(value.to_f64().unwrap_or(0.0));
        }
    }
    
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// 将浮点数转换为有理数表示
fn float_to_rational(num: f64) -> (i64, u64) {
    // 简单实现：将浮点数乘以一个足够大的数，转换为整数
    let precision = 1_000_000_000; // 10^9
    let numerator = (num * precision as f64) as i64;
    let denominator = precision;
    
    (numerator, denominator)
}

/// Babai 最近平面算法实现
fn babai_nearest_plane(B: &Array2<f64>, t: &Array1<f64>) -> Array1<f64> {
    let (n, m) = B.dim();
    assert_eq!(m, t.len());

    // ---------- Gram-Schmidt ----------
    let mut b_gs = B.clone();
    let mut mu   = Array2::<f64>::zeros((n, n));

    for i in 0..n {
    for j in 0..i {
        let dot_ij = b_gs.row(i).dot(&b_gs.row(j));
        let dot_jj = b_gs.row(j).dot(&b_gs.row(j));
        mu[[i, j]] = dot_ij / dot_jj;

        let correction = b_gs.row(j).mapv(|x| x * mu[[i, j]]);
        let row_i = b_gs.row(i).to_owned();   // 先拷出来
        b_gs.row_mut(i).assign(&(row_i - correction));
    }
}

    // ---------- 求解 ----------
    let mut c: Array1<f64> = t.to_owned();             // 关键：显式 f64
    let mut v: Array1<f64> = Array1::<f64>::zeros(m);  // 关键：显式 f64

    for i in (0..n).rev() {
        let ci_dot  = c.dot(&b_gs.row(i));
        let bi_dot  = b_gs.row(i).dot(&b_gs.row(i));

        if bi_dot != 0.0 {
            let coeff = (ci_dot / bi_dot).round();     // 纯 f64
            v = &v + &(&B.row(i)    * coeff);          // 用 &v + &... 避开 += 的推断问题
            c = &c - &(&b_gs.row(i) * coeff);
        }
    }

    v
}

/// 从 Babai 算法结果中提取秘密向量
fn extract_secret(result: &Array1<f64>, m: usize, n: usize, q: u64) -> Array1<u64> {
    // 提取后 n 个元素
    let s_f64 = result.slice(s![m..m+n]);
    
    // 转换为整数并取模 q
    let s_u64: Vec<u64> = s_f64.iter()
        .map(|&x| ((x % q as f64).round() as u64).rem_euclid(q))
        .collect();
    
    Array1::from_vec(s_u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_babai_algorithm() {
        // 测试用例：2D 格基
        let B = array![
            [3.0, 1.0],
            [1.0, 2.0]
        ];
        
        let t = array![2.5, 3.5];
        
        let v = babai_nearest_plane(&B, &t);
        
        // 预期结果应该接近格点
        println!("Babai result: {:?}", v);
        
        // 验证结果是否在格中
        // 这里只是简单检查，实际测试可能需要更严格的验证
        assert!(v.len() == 2);
    }
    
    #[test]
    fn test_float_to_rational() {
        let tests = vec![
            (0.5, (5, 10)),
            (1.25, (125, 100)),
            (0.333333333, (333333333, 1000000000)),
        ];
        
        for (input, expected) in tests {
            let (num, den) = float_to_rational(input);
            assert_eq!((num, den), expected);
        }
    }
}