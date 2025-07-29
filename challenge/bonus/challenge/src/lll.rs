use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Zero, One, Signed, ToPrimitive};
use std::collections::HashSet;

// Lattice结构体定义假设
pub struct Lattice {
    basis: Vec<Vec<BigInt>>, // 整数基向量矩阵
}
impl Lattice {
    pub fn from_integral_basis(basis: Vec<Vec<BigInt>>) -> Self {
        Self { basis }
    }
    pub fn get_basis_as_integer(&self) -> Result<Vec<Vec<BigInt>>, String> {
        Ok(self.basis.clone())
    }
}

// ---------------------
// int_lll 主函数
// ---------------------
pub fn int_lll(lat: &Lattice) -> Result<Lattice, String> {
    // 预处理基向量
    let mut basis = preprocess_basis(lat)?;
    let n = basis.len();

    // 初始化参数
    let mut k: usize = 1;
    let mut k_max: usize = 0;
    let mut mu_matrix = vec![vec![BigRational::zero(); n]; n];
    let mut d: Vec<BigRational> = vec![BigRational::zero(); n + 1];

    // 历史状态用于检测循环
    let mut state_history = HashSet::new();
    let max_iterations = 10000 * n;
    let mut iterations = 0;
    let mut consecutive_swaps = 0;
    let lambda = BigRational::new(BigInt::from(3), BigInt::from(4)); // 0.75

    d[0] = BigRational::one();
    d[1] = norm_squared_bigint(&basis[0]);

    while k < n {
        iterations += 1;
        if iterations > max_iterations {
            println!("达到最大迭代次数 {}", max_iterations);
            break;
        }

        if iterations % 100 == 0 {
            let state_hash = hash_current_state_bigint(&basis, &mu_matrix, k);
            if !state_history.insert(state_hash) {
                println!("检测到重复状态，可能陷入循环");
                break;
            }
        }

        if k > k_max {
            k_max = k;
            for j in 0..=k {
                // u = <b_k, b_j> - sum_i mu_k_i * mu_j_i * d_i
                let mut u = inner_product_bigint(&basis[k], &basis[j]);
                for i in 0..j {
                    let prod = &mu_matrix[k][i] * &mu_matrix[j][i] * &d[i];
                    let sub = BigRational::from_integer(u.clone()) - prod;
                    u = sub.to_integer();
                }
                if j < k {
                    mu_matrix[k][j] = BigRational::from_integer(u);
                } else {
                    d[k + 1] = BigRational::from_integer(u.clone());
                    if u.is_zero() {
                        println!("检测到线性相关向量");
                    }
                }
            }
        }

        // 调用LLL的约减和交换步骤
        int_lll_red(k, k - 1, &mut mu_matrix, &mut basis, &mut d)?;

        // Lovász条件判断
        if lovasz_condition(
            &d[k + 1],
            &d[k],
            &d[k - 1],
            &mu_matrix[k][k - 1],
            &lambda,
        ) {
            int_lll_swap(k, k_max, &mut mu_matrix, &mut basis, &mut d)?;
            k = k.saturating_sub(1).max(1);
            consecutive_swaps += 1;
        } else {
            consecutive_swaps = 0;
            // k-1之前所有向量约减
            for l in (0..k - 1).rev() {
                int_lll_red(k, l, &mut mu_matrix, &mut basis, &mut d)?;
            }
            k += 1;
        }
    }

    Ok(Lattice::from_integral_basis(basis))
}

// -----------------------------------------
// Lovász 条件判断，使用有理数
fn lovasz_condition(
    d_k_plus_1: &BigRational,
    d_k: &BigRational,
    d_k_minus_1: &BigRational,
    mu_k_k_minus_1: &BigRational,
    lambda: &BigRational,
) -> bool {
    let lhs = d_k_plus_1 * d_k_minus_1;
    let rhs = (lambda - mu_k_k_minus_1.pow(2)) * d_k.pow(2);
    lhs < rhs
}

// -----------------------------------------
// 约减操作，令mu_kl = mu_k_l向量的系数化简
fn int_lll_red(
    k: usize,
    l: usize,
    mu_matrix: &mut Vec<Vec<BigRational>>,
    basis: &mut Vec<Vec<BigInt>>,
    d: &mut Vec<BigRational>,
) -> Result<(), String> {
    if mu_matrix[k][l].abs() > BigRational::from_integer(BigInt::one()) {
        let r = mu_matrix[k][l].round();
        let r_int = r.to_integer();
        // basis[k] -= r * basis[l]
        for i in 0..basis[k].len() {
            basis[k][i] -= &basis[l][i] * &r_int;
        }
        mu_matrix[k][l] -= BigRational::from_integer(r_int);
    }
    Ok(())
}

// -----------------------------------------
// 交换基向量k和k-1，并更新相关参数
fn int_lll_swap(
    k: usize,
    k_max: usize,
    mu_matrix: &mut Vec<Vec<BigRational>>,
    basis: &mut Vec<Vec<BigInt>>,
    d: &mut Vec<BigRational>,
) -> Result<(), String> {
    basis.swap(k, k - 1);
    mu_matrix.swap(k, k - 1);
    for row in mu_matrix.iter_mut() {
        row.swap(k, k - 1);
    }
    d.swap(k, k - 1);
    Ok(())
}

// -----------------------------------------
// 预处理基向量：移除线性相关和（可选）缩放
fn preprocess_basis(lat: &Lattice) -> Result<Vec<Vec<BigInt>>, String> {
    let mut basis = lat.get_basis_as_integer()?;

    basis = remove_linear_dependencies(&basis)?;

    // 可选缩放，保持整数精度一般不缩放
    //basis = scale_basis(&basis);

    // 按范数排序（用BigInt计算范数平方）
    basis.sort_by_key(|v| norm_squared_bigint(v));

    Ok(basis)
}

// -----------------------------------------
// 计算平方范数(返回BigRational方便后续运算)
fn norm_squared_bigint(v: &[BigInt]) -> BigRational {
    let sum = v.iter().fold(BigInt::zero(), |acc, x| acc + x * x);
    BigRational::from_integer(sum)
}

// -----------------------------------------
// 向量内积
fn inner_product_bigint(a: &[BigInt], b: &[BigInt]) -> BigInt {
    a.iter()
        .zip(b.iter())
        .fold(BigInt::zero(), |acc, (x, y)| acc + x * y)
}

// -----------------------------------------
// 检测线性相关：用有理数高斯消元实现
fn remove_linear_dependencies(basis: &[Vec<BigInt>]) -> Result<Vec<Vec<BigInt>>, String> {
    let mut filtered = Vec::new();
    for v in basis {
        if !is_linear_combination(&filtered, v)? {
            filtered.push(v.clone());
        }
    }
    Ok(filtered)
}

fn is_linear_combination(basis: &[Vec<BigInt>], target: &[BigInt]) -> Result<bool, String> {
    if basis.is_empty() {
        return Ok(false);
    }

    let n = basis.len();
    let m = basis[0].len();

    if target.len() != m {
        return Err("向量维度不一致".into());
    }

    // 构建矩阵，行是基向量，最后一列是target
    let mut mat: Vec<Vec<BigRational>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<BigRational> = basis[i]
            .iter()
            .map(|x| BigRational::from_integer(x.clone()))
            .collect();
        row.push(BigRational::from_integer(target[i].clone()));
        mat.push(row);
    }

    // 有理数高斯消元
    let rank_before = rank_of_matrix(&mat, m)?;
    for row in &mut mat {
        row.pop();
    }
    let rank_after = rank_of_matrix(&mat, m)?;

    Ok(rank_before == rank_after)
}

// 计算矩阵秩，使用有理数
fn rank_of_matrix(mat: &[Vec<BigRational>], ncols: usize) -> Result<usize, String> {
    let mut mat = mat.to_vec();
    let nrows = mat.len();
    let mut rank = 0;

    for col in 0..ncols {
        let mut pivot = None;
        for r in rank..nrows {
            if !mat[r][col].is_zero() {
                pivot = Some(r);
                break;
            }
        }
        if let Some(pivot_row) = pivot {
            mat.swap(rank, pivot_row);
            let pivot_val = mat[rank][col].clone();
            for c in col..ncols {
                mat[rank][c] = mat[rank][c].clone() / pivot_val.clone();
            }
            for r in 0..nrows {
                if r != rank && !mat[r][col].is_zero() {
                    let factor = mat[r][col].clone();
                    for c in col..ncols {
                        mat[r][c] = mat[r][c].clone() - factor.clone() * mat[rank][c].clone();
                    }
                }
            }
            rank += 1;
        }
    }
    Ok(rank)
}

// -----------------------------------------
// 哈希当前状态，用于循环检测
fn hash_current_state_bigint(
    basis: &[Vec<BigInt>],
    mu_matrix: &[Vec<BigRational>],
    k: usize,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for i in 0..std::cmp::min(k + 2, basis.len()) {
        basis[i].iter().take(5).for_each(|x| {
            x.to_signed_bytes_le().hash(&mut hasher);
        });
    }

    if k < mu_matrix.len() {
        mu_matrix[k].iter().take(5).for_each(|x| {
            let (num, den) = (x.numer(), x.denom());
            num.to_signed_bytes_le().hash(&mut hasher);
            den.to_signed_bytes_le().hash(&mut hasher);
        });
    }

    k.hash(&mut hasher);
    hasher.finish()
}


