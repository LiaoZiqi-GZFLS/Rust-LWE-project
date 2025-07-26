use num::rational::Ratio;
use num::{BigInt, Signed}; // 添加了 Signed trait
use std::cmp::{max, min};

// 向量内积
pub fn inner_product(vs: &[Ratio<BigInt>], ws: &[Ratio<BigInt>]) -> Ratio<BigInt> {
    let mut res = Ratio::from_integer(BigInt::from(0));
    for (v, w) in vs.iter().zip(ws.iter()) {
        res += v * w;
    }
    res
}

// 向量减法
pub fn vecminus(vs: &[Ratio<BigInt>], ws: &[Ratio<BigInt>]) -> Vec<Ratio<BigInt>> {
    vs.iter().zip(ws.iter()).map(|(v, w)| v - w).collect()
}

// 向量数乘
pub fn scalar_mul(vs: &[Ratio<BigInt>], a: Ratio<BigInt>) -> Vec<Ratio<BigInt>> {
    vs.iter().map(|v| v * a.clone()).collect()
}

// Gram-Schmidt正交化
pub fn gram_schmidt(vs: &[Vec<Ratio<BigInt>>]) -> (Vec<Vec<Ratio<BigInt>>>, Vec<Vec<Ratio<BigInt>>>) {
    let n = vs.len();
    let mut res = vec![vec![Ratio::from_integer(BigInt::from(0)); vs[0].len()]; n];
    let mut mus = vec![vec![Ratio::from_integer(BigInt::from(1)); n]; n];
    res[0] = vs[0].clone();
    for i in 1..n {
        let mut v = vs[i].clone();
        for j in 0..i {
            mus[i][j] = inner_product(&vs[i], &res[j]) / inner_product(&res[j], &res[j]);
            v = vecminus(&v, &scalar_mul(&res[j], mus[i][j].clone()));
        }
        res[i] = v;
    }
    (res, mus)
}

// LLL约化
pub fn lll(bs: &[Vec<Ratio<BigInt>>]) -> Vec<Vec<Ratio<BigInt>>> {
    let n = bs.len();
    let (bstars, mut mus) = gram_schmidt(bs);
    let mut norm_bstars = vec![Ratio::from_integer(BigInt::from(0)); n];
    for i in 0..n {
        norm_bstars[i] = inner_product(&bstars[i], &bstars[i]);
    }
    let delta = Ratio::from_integer(BigInt::from(3)) / Ratio::from_integer(BigInt::from(4));

    let mut k = 1;
    let mut result = bs.to_vec();
    while k < n {
        for j in (0..k).rev() {
            let mu_kj = mus[k][j].clone();
            if mu_kj.abs() > Ratio::from_integer(BigInt::from(1)) / Ratio::from_integer(BigInt::from(2)) {
                let q = mu_kj.round();
                result[k] = vecminus(&result[k], &scalar_mul(&result[j], q.clone()));
                for l in 0..=j {
                    mus[k][l] = &mus[k][l] - &(q.clone() * &mus[j][l]);
                }
            }
        }

        if norm_bstars[k] >= (&delta - &mus[k][k - 1].pow(2)) * &norm_bstars[k - 1] {
            k += 1;
        } else {
            result.swap(k, k - 1);
            let mup = mus[k][k - 1].clone();
            let B = &norm_bstars[k] + &(mup.pow(2) * &norm_bstars[k - 1]);
            mus[k][k - 1] = &mup * &(&norm_bstars[k - 1] / &B);
            norm_bstars[k] = &norm_bstars[k] * &(&norm_bstars[k - 1] / &B);
            norm_bstars[k - 1] = B;
            for j in 0..k - 1 {
                // 修复：使用临时变量交换
                let temp = mus[k - 1][j].clone();
                mus[k - 1][j] = mus[k][j].clone();
                mus[k][j] = temp;
            }
            for j in k + 1..n {
                let t = mus[j][k].clone();
                mus[j][k] = &mus[j][k - 1] - &(mup.clone() * &t);
                mus[j][k - 1] = &t + &(&mus[k][k - 1] * &mus[j][k]);
            }
            k = max(k - 1, 1);
        }
    }
    result
}

// BKZ块处理
pub fn bkz_block(bs: &[Vec<Ratio<BigInt>>], block_size: usize) -> Vec<Vec<Ratio<BigInt>>> {
    let n = bs.len();
    let mut result = bs.to_vec();
    for start in 0..n {
        let end = min(start + block_size, n);
        if end - start > 1 {
            let sub_basis: Vec<Vec<Ratio<BigInt>>> = result[start..end].to_vec();
            let reduced_sub_basis = lll(&sub_basis);
            for (i, vec) in reduced_sub_basis.into_iter().enumerate() {
                result[start + i] = vec;
            }
        }
    }
    result
}

// BKZ算法
pub fn bkz(bs: &[Vec<Ratio<BigInt>>], block_size: usize, max_iter: usize) -> Vec<Vec<Ratio<BigInt>>> {
    let mut basis = bs.to_vec();
    for _ in 0..max_iter {
        basis = bkz_block(&basis, block_size);
    }
    basis
}

// 简单的筛法示例
pub fn simple_sieving(bs: &[Vec<Ratio<BigInt>>]) -> Vec<Vec<Ratio<BigInt>>> {
    let mut candidates = bs.to_vec();
    let mut new_candidates = Vec::new();
    for _ in 0..10 { // 简单迭代10次
        let mut temp_candidates = Vec::new();
        for i in 0..candidates.len() {
            for j in i + 1..candidates.len() {
                let sum = vecplus(&candidates[i], &candidates[j]);
                let diff = vecminus(&candidates[i], &candidates[j]);
                temp_candidates.push(sum);
                temp_candidates.push(diff);
            }
        }
        new_candidates.extend(temp_candidates);
        candidates = new_candidates;
        new_candidates = Vec::new();
    }
    // 这里可以添加筛选规则，例如选择最短向量等
    candidates
}

// 向量加法
pub fn vecplus(vs: &[Ratio<BigInt>], ws: &[Ratio<BigInt>]) -> Vec<Ratio<BigInt>> {
    vs.iter().zip(ws.iter()).map(|(v, w)| v + w).collect()
}
/*
fn main() {
    let vs = vec![
        vec![Ratio::from_integer(BigInt::from(1)), Ratio::from_integer(BigInt::from(1)), Ratio::from_integer(BigInt::from(1))],
        vec![Ratio::from_integer(BigInt::from(-1)), Ratio::from_integer(BigInt::from(0)), Ratio::from_integer(BigInt::from(2))],
        vec![Ratio::from_integer(BigInt::from(3)), Ratio::from_integer(BigInt::from(5)), Ratio::from_integer(BigInt::from(6))]
    ];
    let block_size = 2;
    let max_iter = 5;
    let bkz_result = bkz(&vs, block_size, max_iter);
    let sieving_result = simple_sieving(&bkz_result);
    println!("BKZ result: {:?}", bkz_result);
    println!("Sieving result: {:?}", sieving_result);
}
    */