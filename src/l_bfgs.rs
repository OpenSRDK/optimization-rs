use rayon::prelude::*;

pub fn l_bfgs(
    initial: &[f64],
    grad: impl Fn(&[f64]) -> Vec<f64>,
    finishing_grad_error: f64,
    max_memory: usize,
) -> Vec<f64> {
    let mut x = initial.to_vec();
    let mut k = 0;
    let mut m = 0;

    loop {
        k += 1;
        m += 1;
        let mut finish = false;

        let q = grad(&x);
    }

    x
}
