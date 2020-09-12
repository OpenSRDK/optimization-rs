use crate::vec::Vector as _;
use crate::{line_search::*, Status};
use opensrdk_linear_algebra::*;
use std::collections::LinkedList;
use std::error::Error;

/// # Limited memory BFGS
/// https://en.wikipedia.org/wiki/Limited-memory_BFGS
pub struct Lbfgs {
    memory: usize,
    delta: f64,
    epsilon: f64,
    max_iter: usize,
    line_search: LineSearch,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self {
            memory: 8,
            delta: 1e-6,
            epsilon: 1e-6,
            max_iter: 0,
            line_search: LineSearch::default(),
        }
    }
}

impl Lbfgs {
    pub fn with_memory(mut self, memory: usize) -> Self {
        self.memory = memory;

        self
    }

    pub fn with_delta(mut self, delta: f64) -> Self {
        assert!(delta.is_sign_positive(), "delta must be positive");

        self.delta = delta;

        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        assert!(epsilon.is_sign_positive(), "epsilon must be positive");

        self.epsilon = epsilon;

        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;

        self
    }

    pub fn with_line_search(mut self, line_search: LineSearch) -> Self {
        self.line_search = line_search;

        self
    }

    pub fn minimize(
        &self,
        x: &mut [f64],
        fx_gfx: &dyn Fn(&[f64]) -> Result<(f64, Vec<f64>), Box<dyn Error>>,
    ) -> Result<Status, Box<dyn Error>> {
        let mut x_prev = Matrix::new(x.len(), 1);
        let fx_prev = 0.0;
        let mut gfx_prev = Matrix::new(x.len(), 1);

        let mut s_y_rho_inv = LinkedList::<(Matrix, Matrix, f64)>::new();

        for k in 1.. {
            if self.max_iter != 0 && self.max_iter <= k {
                return Ok(Status::MaxIter);
            }

            let (fx, gfx) = fx_gfx(x)?;
            let dfx = fx - fx_prev;

            if dfx.abs() / fx < self.delta {
                return Ok(Status::Delta);
            }

            if gfx.l2_norm() < self.epsilon {
                return Ok(Status::Epsilon);
            }

            let sk = x.to_vec().col_mat() - x_prev;
            let yk = gfx.to_vec().col_mat() - gfx_prev;
            let rhok_inv = (yk.t() * &sk)[0][0];

            let mut h = gfx.clone().col_mat();

            let mut alpha = vec![0.0; k as usize];

            for (i, (si, yi, rhoi_inv)) in s_y_rho_inv.iter().enumerate().rev() {
                alpha[i] = (si.t() * &h)[0][0] / rhoi_inv;

                let alphai = alpha[i];

                h = h - alphai * yi.clone();
            }

            let gamma = (sk.t() * &yk)[0][0] / (yk.t() * &yk)[0][0];

            let mut z: Matrix = gamma * h.clone();

            for (i, (si, yi, rhoi_inv)) in s_y_rho_inv.iter().enumerate() {
                let alphai = alpha[i];
                let betai = (yi.t() * &z)[0][0] / rhoi_inv;

                z = z + (alphai - betai) * si.clone();
            }

            z = -1.0 * z;

            s_y_rho_inv.push_back((sk, yk, rhok_inv));
            if s_y_rho_inv.len() > self.memory {
                s_y_rho_inv.pop_front();
            }

            let step_size = self.line_search.search(fx_gfx, x, z.elems_ref())?;
            let dx = step_size * z;

            if !dx.elems_ref().l2_norm().is_finite() {
                return Ok(Status::NaN);
            }

            gfx_prev = h;
            x_prev = x.to_vec().col_mat();
            x.clone_from_slice((&x_prev + dx).elems_ref());
        }

        Ok(Status::Success)
    }
}

#[cfg(test)]
mod tests {
    use crate::{lbfgs::*, numerical_diff};
    use std::{error::Error, f64::consts::E, f64::consts::PI};
    #[test]
    fn it_works() {
        result().unwrap();
    }

    #[test]
    fn result() -> Result<(), Box<dyn Error>> {
        let mut x = vec![1.2, 3.456789];

        let fx_gfx = |x: &[f64]| {
            let fx = 1.0 + 2.0 * x[0].powi(2) + 3.0 * x[1].powi(4);
            let gx = vec![4.0 * x[0], 12.0 * x[1].powi(3)];

            Ok((fx, gx))
        };
        Lbfgs::default().minimize(&mut x, &fx_gfx)?;

        println!("x: {:#?}", x);

        Ok(())
    }

    #[test]
    fn result_() -> Result<(), Box<dyn Error>> {
        let mut x = vec![30.0, -20.0];

        let fx = |x: &[f64]| {
            let fx = 20.0
                - 20.0
                    * (-0.2
                        * (1.0 / x.len() as f64 * x.into_iter().map(|xi| xi.powi(2)).sum::<f64>())
                            .sqrt())
                    .exp()
                + E
                - (1.0 / x.len() as f64
                    * x.into_iter().map(|xi| (2.0 * PI * xi).cos()).sum::<f64>())
                .exp();

            Ok(fx)
        };
        let fx_gfx = |x: &[f64]| {
            let gfx = numerical_diff(&fx, x, 1e-3)?;
            println!("{:#?}", gfx);

            Ok((fx(x)?, gfx))
        };
        Lbfgs::default().minimize(&mut x, &fx_gfx)?;

        println!("x: {:#?}", x);

        Ok(())
    }
}
