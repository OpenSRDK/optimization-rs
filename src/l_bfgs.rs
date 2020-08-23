use crate::line_search::line_search;
use rayon::prelude::*;

/// # Limited memory BFGS
/// for minimization
/// https://en.wikipedia.org/wiki/Limited-memory_BFGS
pub fn l_bfgs(
    initial: &[f64],
    func_grad: &dyn Fn(&[f64]) -> Result<(f64, Vec<f64>), String>,
    grad_error_goal: f64,
    max_memory: usize,
    initial_step_width: f64,
) -> Result<Vec<f64>, String> {
    let mut x = initial.to_vec();
    let mut x_prev = initial.to_vec();
    let mut g_prev = func_grad(&x)?.1;
    let mut k = 0;

    let mut s = vec![vec![0.0; 0]; 0];
    let mut y = vec![vec![0.0; 0]; 0];
    let mut rho = vec![0.0; 0];

    loop {
        let mut g = func_grad(&x)?.1;
        if g.iter()
            .fold(0.0 / 0.0, |max: f64, gi| gi.abs().max(max.abs()))
            < grad_error_goal
        {
            break;
        }

        s.push(
            x.par_iter()
                .zip(x_prev.par_iter())
                .map(|(xi, xi_prev)| xi - xi_prev)
                .collect(),
        );

        y.push(
            g.par_iter()
                .zip(g_prev.par_iter())
                .map(|(gi, gi_prev)| gi - gi_prev)
                .collect(),
        );

        let sk = &s[k];
        let yk = &y[k];

        rho.push(
            1.0 / yk
                .par_iter()
                .zip(sk.par_iter())
                .map(|(ykj, skj)| ykj * skj)
                .sum::<f64>(),
        );

        let mut alpha = vec![0.0; k as usize];

        for i in ((k - max_memory.min(k))..k).rev() {
            alpha[i] = rho[i]
                * s[i]
                    .par_iter()
                    .zip(g.par_iter())
                    .map(|(sij, gj)| sij * gj)
                    .sum::<f64>();
            let alphai = alpha[i];
            let yi = &y[i];
            g.par_iter_mut()
                .zip(yi.par_iter())
                .for_each(|(gj, yij)| *gj = *gj - alphai * yij);
        }

        let gamma = sk
            .par_iter()
            .zip(yk.par_iter())
            .map(|(skj, ykj)| skj * ykj)
            .sum::<f64>()
            / yk.par_iter().map(|ykj| *ykj * *ykj).sum::<f64>();

        let mut z = g.par_iter().map(|gi| gamma * gi).collect::<Vec<_>>();

        for i in (k - max_memory.min(k))..k {
            let si = &s[i];
            let alphai = alpha[i];
            let betai = rho[i]
                * y[i]
                    .par_iter()
                    .zip(z.par_iter())
                    .map(|(yij, zj)| yij * zj)
                    .sum::<f64>();
            z.par_iter_mut()
                .zip(si.par_iter())
                .for_each(|(zj, sij)| *zj = *zj + sij * (alphai - betai))
        }

        z.par_iter_mut().for_each(|zi| {
            *zi = -*zi;
        });

        let alpha = line_search(&x, func_grad, &z, initial_step_width)?;

        g_prev = g;

        x.par_iter_mut()
            .zip(x_prev.par_iter_mut())
            .zip(z.par_iter())
            .for_each(|((xi, xi_prev), zi)| {
                *xi_prev = *xi;
                *xi = *xi + alpha * zi;
            });

        k += 1;
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
