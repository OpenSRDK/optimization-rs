use opensrdk_linear_algebra::*;
use std::error::Error;

#[derive(Clone, Debug)]
pub struct LineSearch {
    pub initial_step_size: f64,
    pub step_update_rate: f64,
    pub armijo_param: f64,
    pub curvature_param: f64,
}

impl Default for LineSearch {
    fn default() -> Self {
        Self::new(1.0, 0.1, 0.01, 0.99)
    }
}

impl LineSearch {
    pub fn new(
        initial_step_size: f64,
        step_update_rate: f64,
        armijo_param: f64,
        curvature_param: f64,
    ) -> Self {
        Self {
            initial_step_size,
            step_update_rate,
            armijo_param,
            curvature_param,
        }
    }

    pub fn search(
        &self,
        func_grad: &dyn Fn(&[f64]) -> Result<(f64, Vec<f64>), Box<dyn Error>>,
        x: &[f64],
        direction: &[f64],
    ) -> Result<f64, Box<dyn Error>> {
        let mut step_size = self.initial_step_size;
        let x = x.to_vec().col_mat();
        let d = direction.to_vec().col_mat();

        loop {
            let xad = x.clone() + step_size * d.clone();

            let (fx, dfx_dx) = func_grad(x.elems_ref())?;
            let (fxad, dfxad_dx) = func_grad(xad.elems_ref())?;

            let dfx_dx_d = (dfx_dx.row_mat() * &d)[0][0];

            // Armijo condition
            let armijo_left = fxad;
            let armijo_right = fx + self.armijo_param * step_size * dfx_dx_d;

            if armijo_left > armijo_right {
                step_size *= 1.0 - self.step_update_rate;

                continue;
            }

            // Curvature condition
            let curvature_left = self.curvature_param * dfx_dx_d;
            let curvature_right = (dfxad_dx.row_mat() * &d)[0][0];

            if curvature_left > curvature_right {
                step_size *= 1.0 + self.step_update_rate;

                continue;
            }

            break;
        }

        Ok(step_size)
    }
}
