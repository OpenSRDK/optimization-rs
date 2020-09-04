use crate::line_search::*;
use rayon::prelude::*;
use std::collections::LinkedList;
use std::error::Error;

/// # Limited memory BFGS
/// https://en.wikipedia.org/wiki/Limited-memory_BFGS
#[derive(Clone, Debug)]
pub struct LBFGS {
    grad_error_goal: f64,
    max_memory: usize,
    line_search: LineSearch,
}

impl LBFGS {
    pub fn new(grad_error_goal: f64, max_memory: usize, line_search: LineSearch) -> Self {
        Self {
            grad_error_goal,
            max_memory,
            line_search,
        }
    }

    pub fn minimize(
        &self,
        func_grad: &dyn Fn(&[f64]) -> Result<(f64, Vec<f64>), Box<dyn Error>>,
        x: &[f64],
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        let mut x = x.to_vec();
        let mut x_prev = vec![0.0; x.len()];
        let mut g_prev = vec![0.0; x.len()];
        let mut k = 0;

        let mut s_y_rho = LinkedList::<(Vec<f64>, Vec<f64>, f64)>::new();

        loop {
            let mut g = func_grad(&x)?.1;
            let g_max = g
                .iter()
                .fold(0.0 / 0.0, |max: f64, gi| gi.abs().max(max.abs()));
            if g_max < self.grad_error_goal {
                break;
            }

            let sk = x
                .par_iter()
                .zip(x_prev.par_iter())
                .map(|(xi, xi_prev)| xi - xi_prev)
                .collect::<Vec<_>>();
            let yk = g
                .par_iter()
                .zip(g_prev.par_iter())
                .map(|(gi, gi_prev)| gi - gi_prev)
                .collect::<Vec<_>>();
            let rhok = 1.0
                / yk.par_iter()
                    .zip(sk.par_iter())
                    .map(|(ykj, skj)| ykj * skj)
                    .sum::<f64>();

            if rhok.is_infinite() {
                break;
            }

            let mut alpha = vec![0.0; k as usize];

            for (i, (si, yi, rhoi)) in s_y_rho.iter().enumerate().rev() {
                alpha[i] = rhoi
                    * si.par_iter()
                        .zip(g.par_iter())
                        .map(|(sij, gj)| sij * gj)
                        .sum::<f64>();

                let alphai = alpha[i];
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

            for (i, (si, yi, rhoi)) in s_y_rho.iter().enumerate() {
                let alphai = alpha[i];
                let betai = rhoi
                    * yi.par_iter()
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
            s_y_rho.push_back((sk, yk, rhok));
            if s_y_rho.len() > self.max_memory {
                s_y_rho.pop_front();
            }

            let step_size = self.line_search.search(func_grad, &x, &z)?;

            g_prev = g;

            x.par_iter_mut()
                .zip(x_prev.par_iter_mut())
                .zip(z.par_iter())
                .for_each(|((xi, xi_prev), zi)| {
                    *xi_prev = *xi;
                    *xi = *xi + step_size * zi;
                });

            k += 1;
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::lbfgs::*;
    use std::error::Error;
    #[test]
    fn it_works() {
        result().unwrap();
    }

    fn result() -> Result<(), Box<dyn Error>> {
        let lbfgs = LBFGS::new(0.001 * 0.001, 32usize, LineSearch::default());
        let func_grad = |x: &[f64]| {
            let func = 1.0 + x[0].powi(2) + 2.0 * x[1].powi(4);
            let grad = vec![2.0 * x[0], 8.0 * x[1].powi(3)];

            Ok((func, grad))
        };
        let x = lbfgs.minimize(&func_grad, &[-1000.0, 10.0])?;

        println!("{:#?}", x);

        Ok(())
    }
}
