use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;

/// # Stochastic Gradient Descent Adam
/// `alpha = 0.001`
/// `beta1 = 0.9`
/// `beta2 = 0.999`
/// `epsilon = 0.00000001`
pub fn sgd_adam(
    initial: &[f64],
    grad: &dyn Fn(&[f64]) -> Result<Vec<f64>, String>,
    grad_terms: Vec<Box<dyn Fn(&[f64]) -> Result<Vec<f64>, String> + Send + Sync>>,
    batch: usize,
    grad_error_goal: f64,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
) -> Result<Vec<f64>, String> {
    let mut grad_terms_mut = grad_terms;
    let mut w = initial.to_vec();
    let mut m = vec![0.0; w.len()];
    let mut v = vec![0.0; w.len()];
    let mut t = 0;
    let mut rng: StdRng = SeedableRng::seed_from_u64(1);

    loop {
        if grad(&w)?
            .iter()
            .fold(0.0 / 0.0, |max: f64, wi| wi.abs().max(max.abs()))
            < grad_error_goal
        {
            break;
        }

        grad_terms_mut.shuffle(&mut rng);

        for grad_terms_batch in grad_terms_mut.chunks(batch) {
            let batch_sum_grad: Vec<f64> = grad_terms_batch
                .par_iter()
                .map(|grad_term| grad_term(&w))
                .try_reduce(
                    || vec![0.0; w.len()],
                    |mut sum, g| {
                        let g = g;
                        sum.par_iter_mut()
                            .zip(g.into_par_iter())
                            .for_each(|(sumi, gi)| *sumi += gi);
                        Ok(sum)
                    },
                )?;

            m.par_iter_mut()
                .zip(batch_sum_grad.par_iter())
                .for_each(|(mi, gi)| *mi = beta1 * *mi + (1.0 - beta1) * gi);
            v.par_iter_mut()
                .zip(batch_sum_grad.par_iter())
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
