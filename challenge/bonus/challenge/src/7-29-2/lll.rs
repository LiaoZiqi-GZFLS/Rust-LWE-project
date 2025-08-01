use ndarray::Array2;
use std::collections::HashSet;


#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub basis: Vec<Vec<f64>>,
}

impl Lattice {
    pub fn new(basis: Vec<Vec<f64>>) -> Self {
        Self { basis }
    }

    pub fn from_array2(basis: &Array2<f64>) -> Self {
        Self {
            basis: basis
                .outer_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f64>>>(),
        }
    }

    pub fn from_integral_basis(basis: Vec<Vec<i128>>) -> Self {
        Self {
            basis: basis
                .iter()
                .map(|v| v.iter().map(|x| *x as f64).collect::<Vec<f64>>())
                .collect::<Vec<Vec<f64>>>(),
        }
    }

    pub fn is_integral(&self) -> bool {
        self.basis.iter().flatten().fold(0.0, |_acc, x| x.fract()) == 0.0
    }

    pub fn get_basis_as_integer(&self) -> Result<Vec<Vec<i128>>, ()> {
        if !self.is_integral() {
            return Err(());
        }

        Ok(self
            .basis
            .iter()
            .map(|v| v.iter().map(|x| *x as i128).collect::<Vec<i128>>())
            .collect::<Vec<Vec<i128>>>())
    }

    pub fn get_min_norm_from_basis(&self) -> f64 {
        self.basis
            .iter()
            .map(|x| x.norm())
            .collect::<Vec<f64>>()
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b))
    }
}



fn lovasz_condition(k: usize, lambda: f64, norm_vector: &[f64], mu_matrix: &[Vec<f64>]) -> bool {
    norm_vector[k] < (lambda - mu_matrix[k][k - 1] * mu_matrix[k][k - 1]) * norm_vector[k - 1]
}

/// The Lenstra, Lenstra and Lovasz (LLL) algorithm. It can be used to reduce a Lattice basis and to try to solve the SVP problem.
/// Implementation based on Alg 2.6.3 from Henri Cohen - A Course in Computational Algebraic Number Theory.
///
/// # Example
///
/// ```
/// # use lattice_cryptanalysis::lattice::{lll,Lattice};
/// let lat = Lattice::new(vec![vec![1.0, 1.0, 1.0],vec![-1.0, 0.0, 2.0],vec![3.0, 5.0, 6.0],]);
/// let ans = Lattice::new(vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 1.0], vec![-2.0, 0.0, 1.0]]);
/// assert_eq!(ans, lll(&lat).unwrap());
/// ```


fn int_lll_red(
    k: usize,
    l: usize,
    mu_matrix: &mut [Vec<i128>],
    basis: &mut [Vec<i128>],
    d: &mut [i128],
) {
    if 2 * mu_matrix[k][l].abs() > d[l + 1] {
        let q = ((mu_matrix[k][l] as f64) / (d[l + 1] as f64)).round() as i128;
        basis[k] = basis[k].sub(&basis[l].scalar_mult(q));
        mu_matrix[k][l] -= q * d[l + 1];
        for i in 0..l {
            mu_matrix[k][i] -= q * mu_matrix[l][i];
        }
    }
}

fn int_lll_swap(
    k: usize,
    k_max: usize,
    mu_matrix: &mut [Vec<i128>],
    basis: &mut [Vec<i128>],
    d: &mut [i128],
) {
    let aux = basis[k].clone();
    basis[k] = basis[k - 1].clone();
    basis[k - 1] = aux;

    if k > 1 {
        for j in 0..(k - 1) {
            let aux = mu_matrix[k][j];
            mu_matrix[k][j] = mu_matrix[k - 1][j];
            mu_matrix[k - 1][j] = aux;
        }
    }

    let m = mu_matrix[k][k - 1];
    let new_value = (d[k + 1] * d[k - 1] + m * m) / d[k];

    for v in mu_matrix.iter_mut().take(k_max + 1).skip(k + 1) {
        let t = v[k];
        v[k] = (v[k - 1] * d[k + 1] - m * t) / d[k];
        v[k - 1] = (new_value * t + m * v[k]) / d[k + 1];
    }

    d[k] = new_value;
}

/// The Lenstra, Lenstra and Lovasz (LLL) algorithm when we have a basis only with integers.
/// It can be used to reduce a Lattice basis and to try to solve the SVP problem.
/// Implementation based on Alg 2.6.7 from Henri Cohen - A Course in Computational Algebraic Number Theory.
/// Original algorithm belongs to B.M.M. de Weger - Algorithms for diophantine equations (1988)
/*pub fn int_lll(lat: &Lattice) -> Result<Lattice, ()> {
    let mut k: usize = 1;
    let mut k_max: usize = 0;
    let n = lat.basis.len();
    let mut basis = lat.get_basis_as_integer()?;
    let mut mu_matrix = vec![vec![0; n]; n];
    let mut d: Vec<i128> = vec![0; n + 1];

    d[0] = 1;
    d[1] = basis[0].norm_squared();

    while k < n {
        if k > k_max {
            k_max = k;
            for j in 0..(k + 1) {
                let mut u = basis[k].dot(&basis[j]);
                for i in 0..j {
                    u = (d[i + 1] * u - mu_matrix[k][i] * mu_matrix[j][i]) / d[i];
                }
                if j < k {
                    mu_matrix[k][j] = u;
                } else {
                    d[k + 1] = u;
                }
            }
            if d[k + 1] == 0 {
                return Err(());
            }
        }

        int_lll_red(k, k - 1, &mut mu_matrix, &mut basis, &mut d);

        if d[k + 1] * d[k - 1] < (3 * d[k] * d[k]) / 4 - mu_matrix[k][k - 1] * mu_matrix[k][k - 1] {
            int_lll_swap(k, k_max, &mut mu_matrix, &mut basis, &mut d);
            k = std::cmp::max(1, k - 1);
        } else {
            (0..(k - 1)).rev().for_each(|l| {
                int_lll_red(k, l, &mut mu_matrix, &mut basis, &mut d);
            });
            k += 1;
            println!("k: {}, k_max: {}", k, k_max);
        }
    }

    Ok(Lattice::from_integral_basis(basis))
}*/


pub fn int_lll(lat: &Lattice) -> Result<Lattice, String> {
    // 预处理基向量：缩放并检查线性相关性
    let mut basis = preprocess_basis(lat)?;
    let n = basis.len();
    
    // 初始化参数
    let mut k: usize = 1;
    let mut k_max: usize = 0;
    let mut mu_matrix = vec![vec![0; n]; n];
    let mut d: Vec<i128> = vec![0; n + 1];
    
    // 记录历史状态以检测循环
    let mut state_history = HashSet::new();
    let MAX_ITERATIONS: usize = 10000 * n; // 迭代次数上限
    let mut iterations = 0;
    let mut consecutive_swaps = 0;
    let mut lambda = 0.75; // 初始 Lovász 参数
    
    d[0] = 1;
    d[1] = basis[0].norm_squared();
    
    while k < n {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            println!("达到最大迭代次数: {}", MAX_ITERATIONS);
        }
        
        // 每100次迭代检测重复状态
        if iterations % 100 == 0 {
            let state_hash = hash_current_state(&basis, &mu_matrix, k);
            if !state_history.insert(state_hash) {
                println!("检测到重复状态，可能陷入循环");
            }
        }
        
        if k > k_max {
            k_max = k;
            for j in 0..(k + 1) {
                let mut u = basis[k].dot(&basis[j]);
                for i in 0..j {
                    // 安全整数运算，防止溢出
                    u = safe_div(
                        d[i + 1].checked_mul(u).ok_or("整数溢出")? 
                        - mu_matrix[k][i].checked_mul(mu_matrix[j][i]).ok_or("整数溢出")?,
                        d[i]
                    )?;
                }
                if j < k {
                    mu_matrix[k][j] = u;
                } else {
                    d[k + 1] = u;
                    
                    // 检查行列式是否为零（线性相关）
                    if u == 0 {
                        println!("检测到线性相关向量");
                    }
                }
            }
        }
        
        int_lll_red(k, k - 1, &mut mu_matrix, &mut basis, &mut d);
        
        // 连续多次交换后动态调整 Lovász 参数
        if consecutive_swaps > 10 {
            lambda = 0.70; // 放宽条件以跳出循环
        }
        
        // 检查 Lovász 条件，使用安全整数运算
        let lovasz_condition = d[k + 1].checked_mul(d[k - 1]).ok_or("整数溢出")?
            < (3 * d[k] * d[k]) / 4 - mu_matrix[k][k - 1].checked_mul(mu_matrix[k][k - 1]).ok_or("整数溢出")?;
            
        if lovasz_condition {
            int_lll_swap(k, k_max, &mut mu_matrix, &mut basis, &mut d);
            k = std::cmp::max(1, k - 1);
            consecutive_swaps += 1;
        } else {
            consecutive_swaps = 0;
            lambda = 0.75; // 恢复默认参数
            
            // 对k-1之前的所有向量进行约减
            (0..(k - 1)).rev().for_each(|l| {
                int_lll_red(k, l, &mut mu_matrix, &mut basis, &mut d);
            });
            k += 1;
        }
    }
    
    Ok(Lattice::from_integral_basis(basis))
}

// 预处理基向量：缩放并检查线性相关性
fn preprocess_basis(lat: &Lattice) -> Result<Vec<Vec<i128>>, String> {
    let mut basis = lat.get_basis_as_integer()
        .map_err(|_| "无法获取整数基向量")?;
    
    // 移除线性相关向量
    basis = remove_linear_dependencies(&basis)?;
    
    // 缩放基向量以改善条件数
    basis = scale_basis(&basis);
    
    // 按范数排序向量
    basis.sort_by_key(|v| v.iter().map(|&x| x.saturating_mul(x)).sum::<i128>());
    
    Ok(basis)
}

// 安全整数除法，处理除以零的情况
fn safe_div(a: i128, b: i128) -> Result<i128, String> {
    if b == 0 {
        println!("除以零错误");
    }
    Ok(a / b)
}

// 移除线性相关向量
fn remove_linear_dependencies(basis: &[Vec<i128>]) -> Result<Vec<Vec<i128>>, String> {
    let mut result = Vec::new();
    for v in basis {
        if !is_linear_combination(&result, v)? {
            result.push(v.clone());
        }
    }
    if result.len() < basis.len() {
        println!("警告：移除了 {} 个线性相关向量", basis.len() - result.len());
    }
    Ok(result)
}

// 检查向量是否为其他向量的线性组合
fn is_linear_combination(basis: &[Vec<i128>], target: &[i128]) -> Result<bool, String> {
    if basis.is_empty() {
        return Ok(false);
    }
    
    let n = basis.len();
    let m = basis[0].len();
    
    // 确保所有向量维度一致
    if target.len() != m {
        return Err("向量维度不一致".to_string());
    }
    
    // 创建增广矩阵 [basis | target]
    let mut matrix = Vec::with_capacity(n);
    for i in 0..n {
        if basis[i].len() != m {
            return Err("基向量维度不一致".to_string());
        }
        let mut row = basis[i].clone();
        row.push(target[i]);
        matrix.push(row);
    }
    
    // 高斯消元法
    let mut row_ptr = 0;
    
    for col in 0..m {
        // 寻找主元
        let mut pivot_row = None;
        for r in row_ptr..n {
            if matrix[r][col] != 0 {
                pivot_row = Some(r);
                break;
            }
        }
        
        if let Some(pivot) = pivot_row {
            // 交换到当前行
            if pivot != row_ptr {
                matrix.swap(pivot, row_ptr);
            }
            
            // 主元归一化（使用分数避免精度损失）
            let pivot_val = matrix[row_ptr][col];
            
            // 消元
            for r in 0..n {
                if r != row_ptr {
                    let factor = matrix[r][col];
                    if factor != 0 {
                        // 逐元素相减：row[r] -= factor * row[row_ptr] / pivot_val
                        for c in col..=m {
                            let numerator = matrix[r][c] * pivot_val - matrix[row_ptr][c] * factor;
                            let denominator = pivot_val;
                            
                            if numerator % denominator != 0 {
                                // 非整数结果，无法在整数环中表示
                                return Ok(false);
                            }
                            
                            matrix[r][c] = numerator / denominator;
                        }
                    }
                }
            }
            
            row_ptr += 1;
            if row_ptr >= n {
                break;
            }
        }
    }
    
    // 检查增广部分是否全为零
    for r in 0..n {
        if matrix[r][m] != 0 {
            return Ok(false);
        }
    }
    
    Ok(true)
}

// 缩放基向量以改善条件数
fn scale_basis(basis: &[Vec<i128>]) -> Vec<Vec<i128>> {
    // 计算每个向量的范数
    let norms: Vec<f64> = basis.iter()
        .map(|v| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt())
        .collect();
    
    // 计算平均范数
    let avg_norm = norms.iter().sum::<f64>() / norms.len() as f64;
    
    // 缩放每个向量
    basis.iter().zip(norms.iter())
        .map(|(v, norm)| {
            let scale = avg_norm / norm;
            v.iter().map(|&x| (x as f64 * scale).round() as i128).collect()
        })
        .collect()
}

// 计算当前状态的哈希值，用于检测循环
fn hash_current_state(basis: &[Vec<i128>], mu_matrix: &[Vec<i128>], k: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    
    // 哈希前k个基向量的关键部分
    for i in 0..std::cmp::min(k+2, basis.len()) {
        basis[i].iter().take(5).for_each(|&x| x.hash(&mut hasher));
    }
    
    // 哈希相关的mu系数
    if k < mu_matrix.len() {
        mu_matrix[k].iter().take(5).for_each(|&x| x.hash(&mut hasher));
    }
    
    k.hash(&mut hasher);
    hasher.finish()
}


///
/// 在LLL约简结果的基础上进行枚举校准，寻找更精确的解
/*pub fn enumerate_and_calibrate(lat: &Lattice, lll_result: &Lattice, radius: f64) -> Result<Lattice, String> {
    let basis = lll_result.get_basis_as_integer()
        .map_err(|_| "LLL结果不是整数基向量")?;
    
    let n = basis.len();
    let mut best_vector = basis[0].clone();
    let mut min_norm = best_vector.norm_squared();
    
    // 构建Gram矩阵和Gram-Schmidt正交化基
    let gram_matrix = compute_gram_matrix(&basis)?;
    let (gs_basis, mu) = gram_schmidt_integer(&basis)?;
    
    // 枚举参数：搜索半径和步长
    let enum_radius = radius;
    let step = 1;
    
    // 定义递归枚举函数
    fn enumerate_recursive(
        level: usize,
        coef: &mut Vec<i128>,
        current_sum: &mut Vec<i128>,
        best_vector: &mut Vec<i128>,
        min_norm: &mut i128,
        basis: &[Vec<i128>],
        gs_basis: &[Vec<i128>],
        mu: &[Vec<i128>],
        d: &[i128],
        radius: f64,
        step: i128,
    ) {
        if level >= basis.len() {
            // 计算当前组合的范数
            let norm = current_sum.norm_squared();
            if norm < *min_norm {
                *min_norm = norm;
                *best_vector = current_sum.clone();
            }
            return;
        }
        
        // 计算当前维度的搜索范围
        let max_coef = (radius / (gs_basis[level].norm() + 1e-10)) as i128;
        
        // 尝试当前维度的可能系数
        for c in (-max_coef..=max_coef).step_by(step as usize) {
            // 更新当前和
            let prev_sum = current_sum.clone();
            coef[level] = c;
            
            // 累加当前基向量的贡献
            for i in 0..current_sum.len() {
                current_sum[i] += basis[level][i] * c;
            }
            
            // 递归处理下一维度
            enumerate_recursive(
                level + 1,
                coef,
                current_sum,
                best_vector,
                min_norm,
                basis,
                gs_basis,
                mu,
                d,
                radius,
                step,
            );
            
            // 回溯
            *current_sum = prev_sum;
        }
    }
    
    // 初始化递归枚举
    let mut coef = vec![0; n];
    let mut current_sum = vec![0; basis[0].len()];
    
    // 计算Gram-Schmidt正交化的d数组
    let mut d = vec![0; n];
    for i in 0..n {
        d[i] = gs_basis[i].norm_squared();
    }
    
    // 执行枚举
    enumerate_recursive(
        0,
        &mut coef,
        &mut current_sum,
        &mut best_vector,
        &mut min_norm,
        &basis,
        &gs_basis,
        &mu,
        &d,
        enum_radius,
        step,
    );
    
    // 使用找到的最优向量替换基中的第一个向量
    let mut new_basis = basis.clone();
    new_basis[0] = best_vector;
    
    // 重新执行LLL约简以传播更改
    let refined_lat = Lattice::from_integral_basis(new_basis);
    int_lll(&refined_lat)
}*/

pub fn enumerate_and_calibrate(
    lat: &Lattice,
    lll_result: &Lattice,
    radius: f64,
) -> Result<Lattice, String> {
    let basis = lll_result.get_basis_as_integer()
        .map_err(|_| "LLL结果不是整数基向量")?;
    let n = basis.len();
    let dim = basis[0].len();

    // 用大整数范数表示半径的平方界限，避免浮点误差
    let radius_squared = (radius * radius).ceil() as u128;

    // 计算Gram-Schmidt正交化基与mu系数（整数版本），d[i]为正交基向量范数平方（u128）
    let (gs_basis, mu) = gram_schmidt_integer(&basis)?;
    let mut d = vec![0u128; n];
    for i in 0..n {
        d[i] = gs_basis[i].norm_squared() as u128;
        if d[i] == 0 {
            return Err("Gram-Schmidt向量范数为0，基不满秩".into());
        }
    }

    // 递归枚举状态
    struct EnumState<'a> {
        best_vector: Vec<i128>,
        min_norm: u128,
        coef: Vec<i128>,
        current_sum: Vec<i128>,
        basis: &'a [Vec<i128>],
        mu: &'a [Vec<i128>],
        d: &'a [u128],
        n: usize,
        dim: usize,
        radius_squared: u128,
    }

    impl<'a> EnumState<'a> {
        fn norm_squared(v: &[i128]) -> u128 {
            v.iter().map(|&x| (x as i128).pow(2) as u128).sum()
        }

        // 更新current_sum加上系数c * basis[level]
        fn add_basis(&mut self, level: usize, c: i128) {
            for i in 0..self.dim {
                self.current_sum[i] += self.basis[level][i] * c;
            }
        }

        // 递归枚举主函数
        fn enumerate_recursive(&mut self, level: usize) {
            if level == self.n {
                let norm = Self::norm_squared(&self.current_sum);
                if norm < self.min_norm {
                    self.min_norm = norm;
                    self.best_vector = self.current_sum.clone();
                }
                return;
            }

            // 根据Gram-Schmidt长度控制枚举范围（避免用浮点）
            let gs_len = (self.d[level] as f64).sqrt().max(1.0);
            let max_coef = (self.radius_squared as f64).sqrt() / gs_len;
            let max_c = max_coef.ceil() as i128;

            for c in -max_c..=max_c {
                self.coef[level] = c;
                self.add_basis(level, c);

                // 剪枝: 当前范数超限则回溯
                let partial_norm = Self::norm_squared(&self.current_sum);
                if partial_norm <= self.radius_squared {
                    self.enumerate_recursive(level + 1);
                }

                // 回溯
                self.add_basis(level, -c);
            }
        }
    }

    // 初始化枚举状态
    let mut state = EnumState {
        best_vector: basis[0].clone(),
        min_norm: EnumState::norm_squared(&basis[0]),
        coef: vec![0; n],
        current_sum: vec![0; dim],
        basis: &basis,
        mu: &mu,
        d: &d,
        n,
        dim,
        radius_squared,
    };

    state.enumerate_recursive(0);

    // 用找到的最优向量替换基向量
    let mut new_basis = basis.clone();
    new_basis[0] = state.best_vector;

    // 重新LLL约简传播更改
    let refined_lat = Lattice::from_integral_basis(new_basis);
    int_lll(&refined_lat)
}


/// 计算整数基的Gram矩阵
fn compute_gram_matrix(basis: &[Vec<i128>]) -> Result<Vec<Vec<i128>>, String> {
    let n = basis.len();
    let mut gram = vec![vec![0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            gram[i][j] = basis[i].dot(&basis[j]);
        }
    }
    
    Ok(gram)
}

/// 计算整数基的Gram-Schmidt正交化
fn gram_schmidt_integer(basis: &[Vec<i128>]) -> Result<(Vec<Vec<i128>>, Vec<Vec<i128>>), String> {
    let n = basis.len();
    let mut gs_basis = vec![vec![0; basis[0].len()]; n];
    let mut mu = vec![vec![0; n]; n];
    
    for i in 0..n {
        gs_basis[i] = basis[i].clone();
        for j in 0..i {
            let numerator = basis[i].dot(&gs_basis[j]);
            let denominator = gs_basis[j].norm_squared();
            
            if denominator == 0 {
                return Err("Gram-Schmidt正交化过程中出现零向量".to_string());
            }
            
            // 计算整数近似的mu值
            mu[i][j] = (numerator * 2 / denominator) / 2; // 四舍五入
            
            // 减去投影
            for k in 0..gs_basis[i].len() {
                gs_basis[i][k] -= (gs_basis[j][k] * mu[i][j]) / denominator;
            }
        }
    }
    
    Ok((gs_basis, mu))
}
///



/// The Gram Schmidt algorithm computes an orthogonal basis given an arbitrary basis.
///
/// # Examples
/// ```
/// # use lattice_cryptanalysis::linear_algebra::{gram_schmidt, VecLinearAlgebra};
/// let basis = vec![vec![1.0, 2.0],vec![3.0, 7.0]];
/// let orth_basis = gram_schmidt(&basis);
/// assert_eq!(orth_basis[0].dot(&orth_basis[1]).round(), 0.0);
/// ```
pub fn gram_schmidt(basis: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut new_basis = vec![basis[0].clone()];
    for v in basis.iter().skip(1) {
        new_basis.push(v.sub(&v.projection(&new_basis)));
    }

    new_basis
}

///This trait is designed to implement basic linear algebra functionalities to base types.
pub trait VecLinearAlgebra<T> {
    ///The dot product between two vectors.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let a: Vec<f64> = vec![1.0, 2.0];
    /// let b: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(a.dot(&b), 11.0);
    /// ~~~
    fn dot(&self, v: &[T]) -> T;

    ///Computes the squared norm of the vector.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let v: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(v.norm_squared(), 25.0);
    /// ~~~
    fn norm_squared(&self) -> T;

    ///Computes the norm of the vector.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let v: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(v.norm(), 5.0);
    /// ~~~
    fn norm(&self) -> f64;

    ///Adds two vectors.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let a: Vec<f64> = vec![1.0, 2.0];
    /// let b: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(a.add(&b), vec![4.0, 6.0]);
    /// ~~~
    fn add(&self, v: &[T]) -> Vec<T>;

    ///Adds two vectors.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let a: Vec<f64> = vec![1.0, 2.0];
    /// let b: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(a.sub(&b), vec![-2.0, -2.0]);
    /// ~~~
    fn sub(&self, v: &[T]) -> Vec<T>;

    ///Multiplies a vector by a scalar.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let v: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(v.scalar_mult(5.0), vec![15.0, 20.0]);
    /// ~~~
    fn scalar_mult(&self, a: T) -> Vec<T>;

    ///Computes the projection of the vector into a space spanned by some basis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let basis = vec![vec![1.0, 2.0, 2.0], vec![2.0, 1.0, -2.0]];
    /// let v = vec![2.0, 9.0, -4.0];
    /// println!("{:?}", v.projection(&basis));
    /// ```
    fn projection(&self, basis: &[Vec<T>]) -> Vec<f64>;
}

pub trait MatLinearAlgebra<T> {
    /// Compute the transpose of a matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lattice_cryptanalysis::linear_algebra::MatLinearAlgebra;
    /// let m = vec![vec![1.0, 2.0, 3.0],vec![4.0, 5.0, 6.0],];
    /// let mt = vec![vec![1.0,4.0],vec![2.0,5.0],vec![3.0,6.0],];
    /// assert_eq!(mt, m.transpose());
    /// ```
    fn transpose(&self) -> Vec<Vec<T>>;

    fn mat_mult(&self, m: &Vec<Vec<T>>) -> Vec<Vec<T>>;
}

//Implementation of basic linear algebra methods for f64.
impl VecLinearAlgebra<f64> for Vec<f64> {
    fn dot(&self, v: &[f64]) -> f64 {
        self.iter().zip(v.iter()).map(|(x, y)| x * y).sum::<f64>()
    }

    fn norm_squared(&self) -> f64 {
        self.iter().map(|x| x * x).sum::<f64>()
    }

    fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    fn add(&self, v: &[f64]) -> Vec<f64> {
        self.iter().zip(v.iter()).map(|(x, y)| x + y).collect()
    }

    fn sub(&self, v: &[f64]) -> Vec<f64> {
        self.iter().zip(v.iter()).map(|(x, y)| x - y).collect()
    }

    fn scalar_mult(&self, a: f64) -> Vec<f64> {
        self.iter().map(|x| x * a).collect()
    }

    fn projection(&self, basis: &[Vec<f64>]) -> Vec<f64> {
        let mut new_vec = vec![0.0; self.len()];
        for v in basis.iter() {
            new_vec = new_vec.add(&v.scalar_mult(self.dot(v) / v.norm_squared()));
        }
        new_vec
    }
}

impl VecLinearAlgebra<i128> for Vec<i128> {
    fn dot(&self, v: &[i128]) -> i128 {
        self.iter().zip(v.iter()).map(|(x, y)| x * y).sum::<i128>()
    }

    fn norm_squared(&self) -> i128 {
        self.iter().map(|x| x * x).sum::<i128>()
    }

    fn norm(&self) -> f64 {
        (self.norm_squared() as f64).sqrt()
    }

    fn add(&self, v: &[i128]) -> Vec<i128> {
        self.iter().zip(v.iter()).map(|(x, y)| x + y).collect()
    }

    fn sub(&self, v: &[i128]) -> Vec<i128> {
        self.iter().zip(v.iter()).map(|(x, y)| x - y).collect()
    }

    fn scalar_mult(&self, a: i128) -> Vec<i128> {
        self.iter().map(|x| x * a).collect()
    }

    fn projection(&self, basis: &[Vec<i128>]) -> Vec<f64> {
        let mut new_vec = vec![0.0; self.len()];
        for v in basis.iter() {
            let vf = v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
            new_vec = new_vec.add(&vf.scalar_mult((self.dot(v) / v.norm_squared()) as f64));
        }
        new_vec
    }
}

impl MatLinearAlgebra<f64> for Vec<Vec<f64>> {
    fn transpose(&self) -> Vec<Vec<f64>> {
        let mut t = vec![Vec::with_capacity(self.len()); self[0].len()];
        for r in self {
            for i in 0..r.len() {
                t[i].push(r[i]);
            }
        }
        t
    }

    fn mat_mult(&self, m: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mt = m.transpose();
        let mut r = vec![Vec::with_capacity(self.len()); self[0].len()];
        for (i, x) in self.iter().enumerate() {
            for y in mt.iter() {
                r[i].push(x.dot(y));
            }
        }
        r
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lll_test_silverman_example() {
        let answer = Lattice::from_integral_basis(vec![
            vec![5, 2, 33, 0, 15, -9],
            vec![-20, 4, -9, 16, 13, 16],
            vec![-9, -19, 8, 6, -29, 10],
            vec![15, 42, 11, 0, 3, 24],
            vec![28, -27, -11, 24, 1, -8],
        ]);

        let silverman_lat = Lattice::from_integral_basis(vec![
            vec![19, 2, 32, 46, 3, 33],
            vec![15, 42, 11, 0, 3, 24],
            vec![43, 15, 0, 24, 4, 16],
            vec![20, 44, 44, 0, 18, 15],
            vec![0, 48, 35, 16, 31, 31],
        ]);

        assert_eq!(answer, lll(&silverman_lat).unwrap());
        assert_eq!(answer, int_lll(&silverman_lat).unwrap());
    }
}
