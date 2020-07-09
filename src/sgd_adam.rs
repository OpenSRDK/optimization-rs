use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;

/// # Stochastic Gradient Descent Adam
/// `alpha = 0.001`
/// `beta1 = 0.9`
/// `beta2 = 0.999`
/// `epsilon = 0.00000001`
pub fn sgd_adam(
    initial: &[f64],
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    grad_terms: Vec<Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>>,
    batch: usize,
    finishing_grad_error: f64,
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
) -> Vec<f64> {
    let mut grad_terms_mut = grad_terms;
    let mut w = initial.to_vec();
    let mut m = vec![0.0; w.len()];
    let mut v = vec![0.0; w.len()];
    let mut t = 0;
    let mut rng: StdRng = SeedableRng::seed_from_u64(1);

    loop {
        if grad(&w)
            .iter()
            .fold(0.0 / 0.0, |m: f64, v| v.abs().max(m.abs()))
            < finishing_grad_error
        {
            break;
        }

        grad_terms_mut.shuffle(&mut rng);

        for grad_terms_batch in grad_terms_mut.chunks(batch) {
            let batch_sum_grad: Vec<f64> = grad_terms_batch
                .par_iter()
                .map(|grad_term| grad_term(&w))
                .reduce(
                    || vec![0.0; w.len()],
                    |mut sum, g| {
                        sum.par_iter_mut()
                            .zip(g.into_par_iter())
                            .for_each(|(sum_e, g_e)| *sum_e += g_e);
                        sum
                    },
                );

            m.par_iter_mut()
                .zip(batch_sum_grad.par_iter())
                .for_each(|(m_e, g_e)| *m_e = beta1 * *m_e + (1.0 - beta1) * g_e);
            v.par_iter_mut()
                .zip(batch_sum_grad.par_iter())
                .for_each(|(v_e, g_e)| *v_e = beta2 * *v_e + (1.0 - beta1) * g_e.powi(2));

            let m_hat = m
                .par_iter()
                .map(|m_e| m_e / (1.0 - beta1.powi(t + 1)))
                .collect::<Vec<_>>();
            let v_hat = v
                .par_iter()
                .map(|v_e| v_e / (1.0 - beta1.powi(t + 1)))
                .collect::<Vec<_>>();

            let m_v = m_hat.into_par_iter().zip(v_hat.into_par_iter());

            w.par_iter_mut()
                .zip(m_v)
                .for_each(|(w_e, (m_e, v_e))| *w_e = *w_e - alpha * m_e / (v_e.sqrt() + epsilon));
        }
        t += 1;
    }

    w
}
