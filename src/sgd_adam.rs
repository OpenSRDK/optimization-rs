use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;
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
    let mut w = x.to_vec();
    let mut m = vec![0.0; w.len()];
    let mut v = vec![0.0; w.len()];
    let mut t = 0;
    let mut rng: StdRng = SeedableRng::seed_from_u64(1);

    loop {
        if grad(&(0..total).into_iter().collect::<Vec<_>>(), &w)?
            .iter()
            .fold(0.0 / 0.0, |max: f64, wi| wi.abs().max(max.abs()))
            < grad_error_goal
        {
            break;
        }

        batch_index.shuffle(&mut rng);

        for minibatch in batch_index.chunks(batch) {
            let minibatch_grad: Vec<f64> = grad(&minibatch, &w)?;

            m.par_iter_mut()
                .zip(minibatch_grad.par_iter())
                .for_each(|(mi, gi)| *mi = beta1 * *mi + (1.0 - beta1) * gi);
            v.par_iter_mut()
                .zip(minibatch_grad.par_iter())
                .for_each(|(vi, gi)| *vi = beta2 * *vi + (1.0 - beta1) * gi.powi(2));

            let m_hat = m
                .par_iter()
                .map(|me| me / (1.0 - beta1.powi(t + 1)))
                .collect::<Vec<_>>();
            let v_hat = v
                .par_iter()
                .map(|ve| ve / (1.0 - beta1.powi(t + 1)))
                .collect::<Vec<_>>();

            let m_v = m_hat.into_par_iter().zip(v_hat.into_par_iter());

            w.par_iter_mut()
                .zip(m_v)
                .for_each(|(wi, (mi, vi))| *wi = *wi - alpha * mi / (vi.sqrt() + epsilon));
        }
        t += 1;
    }

    Ok(w)
}
