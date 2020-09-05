use opensrdk_linear_algebra::*;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::error::Error;

/// # Stochastic Gradient Descent Adam
/// `alpha = 0.001`
/// `beta1 = 0.9`
/// `beta2 = 0.999`
/// `epsilon = 0.00000001`
pub fn sgd_adam(
    grad: &(dyn Fn(&[usize], &[f64]) -> Result<Vec<f64>, Box<dyn Error>> + Send + Sync),
    x: &[f64],
    grad_error_goal: f64,
    batch: usize,
    total: usize,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut batch_index = (0..total).into_iter().collect::<Vec<_>>();
    let mut w = x.to_vec().col_mat();
    let mut m = Matrix::new(w.rows(), 1);
    let mut v = Matrix::new(w.rows(), 1);
    let mut t = 0;
    let mut rng: StdRng = SeedableRng::seed_from_u64(1);

    loop {
        let g_max = grad(&(0..total).into_iter().collect::<Vec<_>>(), w.elems_ref())?
            .iter()
            .fold(0.0 / 0.0, |max: f64, wi| wi.abs().max(max.abs()));
        if g_max < grad_error_goal {
            break;
        }

        batch_index.shuffle(&mut rng);

        for minibatch in batch_index.chunks(batch) {
            let minibatch_grad = grad(&minibatch, w.elems_ref())?.col_mat();

            m = beta1 * m + (1.0 - beta1) * minibatch_grad.clone();
            v = beta2 * v + (1.0 - beta2) * minibatch_grad.clone().hadamard_prod(&minibatch_grad);

            let m_hat = m.clone() * (1.0 / (1.0 - beta1.powi(t + 1)));
            let v_hat = v.clone() * (1.0 / (1.0 - beta2.powi(t + 1)));
            let v_hat_sqrt_e_inv = v_hat
                .elems()
                .iter()
                .map(|vi| 1.0 / (vi.sqrt() + epsilon))
                .collect::<Vec<_>>()
                .col_mat();

            w = w - alpha * m_hat.hadamard_prod(&v_hat_sqrt_e_inv);
        }
        t += 1;
    }

    Ok(w.elems())
}
