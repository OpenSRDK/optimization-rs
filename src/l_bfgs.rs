use crate::line_search::line_search;
use rayon::prelude::*;

/// # Limited memory BFGS
/// for minimization
/// https://en.wikipedia.org/wiki/Limited-memory_BFGS
pub fn l_bfgs(
    initial: &[f64],
    func_grad: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
    finishing_grad_error: f64,
    max_memory: usize,
    initial_step_width: f64,
) -> Vec<f64> {
    let mut x = initial.to_vec();
    let mut x_before = initial.to_vec();
    let mut g_before = func_grad(&x).1;
    let mut k = 0;

    let mut s = vec![vec![0.0; 0]; 0];
    let mut y = vec![vec![0.0; 0]; 0];
    let mut rho = vec![0.0; 0];

    loop {
        let mut g = func_grad(&x).1;
        if g.iter().fold(0.0 / 0.0, |m: f64, v| v.abs().max(m.abs())) < finishing_grad_error {
            break;
        }

        s.push(
            x.par_iter()
                .zip(x_before.par_iter())
                .map(|(x_e, x_before_e)| x_e - x_before_e)
                .collect(),
        );

        y.push(
            g.par_iter()
                .zip(g_before.par_iter())
                .map(|(g_e, g_before_e)| g_e - g_before_e)
                .collect(),
        );

        let s_k = &s[k];
        let y_k = &y[k];

        rho.push(
            1.0 / y_k
                .par_iter()
                .zip(s_k.par_iter())
                .map(|(y_e, s_e)| y_e * s_e)
                .sum::<f64>(),
        );

        let mut alpha = vec![0.0; k as usize];

        for i in ((k - max_memory.min(k))..k).rev() {
            alpha[i] = rho[i]
                * s[i]
                    .par_iter()
                    .zip(g.par_iter())
                    .map(|(s_e, g_e)| s_e * g_e)
                    .sum::<f64>();
            let alpha_i = alpha[i];
            let y_i = &y[i];
            g.par_iter_mut()
                .zip(y_i.par_iter())
                .for_each(|(g_e, y_e)| *g_e = *g_e - alpha_i * y_e);
        }

        let gamma = s_k
            .par_iter()
            .zip(y_k.par_iter())
            .map(|(s_e, y_e)| s_e * y_e)
            .sum::<f64>()
            / y_k.par_iter().map(|y_e| *y_e * *y_e).sum::<f64>();

        let mut z = g.par_iter().map(|q_e| gamma * q_e).collect::<Vec<_>>();

        for i in (k - max_memory.min(k))..k {
            let s_i = &s[i];
            let alpha_i = alpha[i];
            let beta_i = rho[i]
                * y[i]
                    .par_iter()
                    .zip(z.par_iter())
                    .map(|(y_e, z_e)| y_e * z_e)
                    .sum::<f64>();
            z.par_iter_mut()
                .zip(s_i.par_iter())
                .for_each(|(z_e, s_e)| *z_e = *z_e + s_e * (alpha_i - beta_i))
        }

        z.par_iter_mut().for_each(|z_e| {
            *z_e = -*z_e;
        });

        let alpha = line_search(&x, func_grad, &z, initial_step_width);

        g_before = g;

        x.par_iter_mut()
            .zip(x_before.par_iter_mut())
            .zip(z.par_iter())
            .for_each(|((x_e, x_before_e), z_e)| {
                *x_before_e = *x_e;
                *x_e = *x_e + alpha * z_e;
            });

        k += 1;
    }

    x
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
