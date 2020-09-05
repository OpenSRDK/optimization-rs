use crate::line_search::*;
use opensrdk_linear_algebra::*;
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
        let mut x = x.to_vec().col_mat();
        let mut x_prev = Matrix::new(x.rows(), 1);
        let mut g_prev = Matrix::new(x.rows(), 1);
        let mut k = 0;

        let mut s_y_rho = LinkedList::<(Matrix, Matrix, f64)>::new();

        loop {
            let mut g = func_grad(x.elems_ref())?.1.col_mat();
            let g_max = g
                .elems_ref()
                .iter()
                .fold(0.0 / 0.0, |max: f64, gi| gi.abs().max(max.abs()));
            if g_max < self.grad_error_goal {
                break;
            }

            let sk = x.clone() - x_prev;
            let yk = g.clone() - g_prev;
            let rhok = 1.0 / (yk.t() * &sk)[0][0];

            if rhok.is_infinite() {
                break;
            }

            let mut alpha = vec![0.0; k as usize];

            for (i, (si, yi, rhoi)) in s_y_rho.iter().enumerate().rev() {
                alpha[i] = rhoi * (si.t() * &g)[0][0];

                let alphai = alpha[i];

                g = g - alphai * yi.clone();
            }

            let gamma = (sk.t() * &yk)[0][0] / (yk.t() * &yk)[0][0];

            let mut z = gamma * g.clone();

            for (i, (si, yi, rhoi)) in s_y_rho.iter().enumerate() {
                let alphai = alpha[i];
                let betai = rhoi * (yi.t() * &z)[0][0];

                z = z + (alphai - betai) * si.clone();
            }

            z = -1.0 * z;

            s_y_rho.push_back((sk, yk, rhok));
            if s_y_rho.len() > self.max_memory {
                s_y_rho.pop_front();
            }

            let step_size = self
                .line_search
                .search(func_grad, x.elems_ref(), z.elems_ref())?;

            g_prev = g;

            x_prev = x.clone();
            x = x + step_size * z;

            k += 1;
        }

        Ok(x.elems())
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
