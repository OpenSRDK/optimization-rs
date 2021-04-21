use opensrdk_linear_algebra::*;

#[derive(Clone, Debug)]
pub struct LineSearch {
  initial_step_size: f64,
  step_update_rate: f64,
  armijo_param: f64,
  curvature_param: f64,
}

impl Default for LineSearch {
  fn default() -> Self {
    Self {
      initial_step_size: 1.0,
      step_update_rate: 0.1,
      armijo_param: 0.1,
      curvature_param: 0.9,
    }
  }
}
impl LineSearch {
  pub fn with_initial_step_size(mut self, initial_step_size: f64) -> Self {
    self.initial_step_size = initial_step_size;

    self
  }

  pub fn with_step_update_rate(mut self, step_update_rate: f64) -> Self {
    self.step_update_rate = step_update_rate;

    self
  }

  pub fn with_armijo_param(mut self, armijo_param: f64) -> Self {
    self.armijo_param = armijo_param;

    self
  }

  pub fn with_curvature_param(mut self, curvature_param: f64) -> Self {
    self.curvature_param = curvature_param;

    self
  }

  pub fn search(
    &self,
    fx_gfx: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
    x: &[f64],
    direction: &[f64],
  ) -> f64 {
    let mut step_size = self.initial_step_size;
    let x = x.to_vec().col_mat();
    let d = direction.to_vec().col_mat();

    loop {
      let xad = x.clone() + step_size * d.clone();

      let (fx, dfx_dx) = fx_gfx(x.slice());
      let (fxad, dfxad_dx) = fx_gfx(xad.slice());

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

    step_size
  }
}
