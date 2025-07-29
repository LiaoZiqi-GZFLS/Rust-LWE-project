use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::rand_distr::{Normal, Uniform};

pub(crate) struct LWEInstance {
    pub a: Array2<u64>,
    pub b: Array1<u64>,
    pub s: Array1<u64>,
}

pub(crate) fn generate_lwe_instance(n: usize, m: usize, q: u64, alpha: f64) -> LWEInstance {
    let mut rng = thread_rng();
    let sigma = alpha * q as f64;

    let uniform = Uniform::new(0, q);
    let normal = Normal::new(0.0, sigma).unwrap();

    let a = Array2::random_using((m, n), uniform, &mut rng);
    let s = Array1::random_using(n, uniform, &mut rng);
    let e: Array1<f64> = Array1::random_using(m, normal, &mut rng);

    let mut b = a.dot(&s);
    for (bi, ei) in b.iter_mut().zip(e.iter()) {
        let error = ei.round() as i64;
        let val = (*bi as i64 + error).rem_euclid(q as i64);
        *bi = val as u64;
    }

    LWEInstance { a, b, s }
}
