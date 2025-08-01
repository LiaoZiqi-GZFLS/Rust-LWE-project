use ndarray::Array2;

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

// 在 lll.rs 中添加以下代码

/// BKZ 算法实现
pub fn bkz(lat: &Lattice, block_size: usize) -> Result<Lattice, ()> {
    let mut basis = lat.basis.clone();
    let n = basis.len();

    for k in 1..n {
        if k >= block_size {
            // 进行局部 SVP 求解，这里简单使用 LLL 作为示例，实际中可以使用更高效的方法
            let block_start = k - block_size + 1;
            let block_end = k + 1;
            let block_basis = basis[block_start..block_end].to_vec();
            let block_lattice = Lattice::new(block_basis);
            let reduced_block = lll(&block_lattice)?;
            for i in 0..block_size {
                basis[block_start + i] = reduced_block.basis[i].clone();
            }
        }
        // 进行 LLL 约化
        let lattice = Lattice::new(basis.clone());
        let reduced_lattice = lll(&lattice)?;
        basis = reduced_lattice.basis;
    }

    Ok(Lattice::new(basis))
}

// 在 lll.rs 中添加以下代码

/// Sieving 算法实现
pub fn sieving(lat: &Lattice, num_iterations: usize) -> Vec<f64> {
    let mut vectors = lat.basis.clone();
    let n = vectors.len();

    for _ in 0..num_iterations {
        // 随机生成新向量
        let new_vector = vectors.iter().fold(vec![0.0; vectors[0].len()], |acc, v| {
            let coeff = rand::random::<f64>() - 0.5;
            acc.add(&v.scalar_mult(coeff))
        });

        // 筛选操作
        let mut new_vectors = vec![];
        for v in vectors {
            if new_vector.norm() < v.norm() {
                new_vectors.push(new_vector.clone());
            } else {
                new_vectors.push(v);
            }
        }
        vectors = new_vectors;
    }

    // 找到最短向量
    let shortest_vector = vectors.iter().min_by_key(|v| (v.norm() * 1000.0) as u64).unwrap();
    shortest_vector.clone()
}

fn lll_red(k: usize, l: usize, mu_matrix: &mut [Vec<f64>], basis: &mut [Vec<f64>]) {
    if mu_matrix[k][l].abs() > 0.5 {
        let q = mu_matrix[k][l].round();
        basis[k] = basis[k].sub(&basis[l].scalar_mult(q));
        mu_matrix[k][l] -= q;
        for i in 0..l {
            mu_matrix[k][i] -= q * mu_matrix[l][i];
        }
    }
}

fn lll_swap(
    k: usize,
    k_max: usize,
    mu_matrix: &mut [Vec<f64>],
    basis: &mut [Vec<f64>],
    basis_star: &mut [Vec<f64>],
    inner_vector: &mut [f64],
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
    let new_value = inner_vector[k] + m * m * inner_vector[k - 1];
    mu_matrix[k][k - 1] = m * inner_vector[k - 1] / new_value;
    let b = basis_star[k - 1].clone();
    basis_star[k - 1] = basis_star[k].add(&b.scalar_mult(m));
    basis_star[k] = b
        .scalar_mult(inner_vector[k] / new_value)
        .sub(&basis_star[k].scalar_mult(mu_matrix[k][k - 1]));

    inner_vector[k] = inner_vector[k - 1] * inner_vector[k] / new_value;
    inner_vector[k - 1] = new_value;

    for i in (k + 1)..(k_max + 1) {
        let t = mu_matrix[i][k];
        mu_matrix[i][k] = mu_matrix[i][k - 1] - m * t;
        mu_matrix[i][k - 1] = t + mu_matrix[k][k - 1] * mu_matrix[i][k];
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
pub fn lll(lat: &Lattice) -> Result<Lattice, ()> {
    let mut k: usize = 1;
    let mut k_max: usize = 0;
    let n = lat.basis.len();
    let mut basis = lat.basis.clone();
    let mut basis_star = lat.basis.clone();
    let mut mu_matrix = vec![vec![0.0; n]; n];
    let mut inner_vector: Vec<f64> = vec![0.0; n];

    inner_vector[0] = basis[0].norm_squared();

    while k < n {
        if k > k_max {
            k_max = k;
            basis_star[k] = basis[k].clone();
            for j in 0..k {
                mu_matrix[k][j] = basis[k].dot(&basis_star[j]) / inner_vector[j];
                basis_star[k] = basis_star[k].sub(&basis_star[j].scalar_mult(mu_matrix[k][j]));
            }
            inner_vector[k] = basis_star[k].norm_squared();
            if inner_vector[k] == 0.0 {
                return Err(());
            }
        }

        lll_red(k, k - 1, &mut mu_matrix, &mut basis);

        if lovasz_condition(k, 0.75, &inner_vector, &mu_matrix) {
            lll_swap(
                k,
                k_max,
                &mut mu_matrix,
                &mut basis,
                &mut basis_star,
                &mut inner_vector,
            );
            k = std::cmp::max(1, k - 1);
        } else {
            (0..(k - 1)).rev().for_each(|l| {
                lll_red(k, l, &mut mu_matrix, &mut basis);
            });
            k += 1;
        }
    }

    Ok(Lattice::new(basis))
}

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
pub fn int_lll(lat: &Lattice) -> Result<Lattice, ()> {
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
        }
    }

    Ok(Lattice::from_integral_basis(basis))
}

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
