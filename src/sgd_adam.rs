use crate::{vec::Vector as _, Status};
use opensrdk_linear_algebra::*;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

/// # Stochastic Gradient Descent Adam
pub struct SgdAdam {
  epsilon: f64,
  max_iter: usize,
  alpha: f64,
  beta1: f64,
  beta2: f64,
  e: f64,
}

impl Default for SgdAdam {
  fn default() -> Self {
    Self {
      epsilon: 1e-6,
      max_iter: 0,
      alpha: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      e: 0.00000001,
    }
  }
}

impl SgdAdam {
  pub fn new(epsilon: f64, max_iter: usize, alpha: f64, beta1: f64, beta2: f64, e: f64) -> Self {
    Self {
      epsilon,
      max_iter,
      alpha,
      beta1,
      beta2,
      e,
    }
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

  pub fn with_alpha(mut self, alpha: f64) -> Self {
    assert!(alpha.is_sign_positive(), "delta must be positive");

    self.alpha = alpha;

    self
  }

  pub fn with_beta1(mut self, beta1: f64) -> Self {
    assert!(beta1.is_sign_positive(), "beta1 must be positive");

    self.beta2 = beta1;

    self
  }

  pub fn with_beta2(mut self, beta2: f64) -> Self {
    assert!(beta2.is_sign_positive(), "beta2 must be positive");

    self.beta2 = beta2;

    self
  }

  pub fn with_e(mut self, e: f64) -> Self {
    assert!(e.is_sign_positive(), "e must be positive");

    self.e = e;

    self
  }

  pub fn minimize(
    &self,
    x: &mut [f64],
    grad: &dyn Fn(&[usize], &[f64]) -> Vec<f64>,
    batch: usize,
    total: usize,
  ) -> Status {
    let mut batch_index = (0..total).into_iter().collect::<Vec<_>>();
    let mut w = x.to_vec().col_mat();
    let mut m = Matrix::new(w.rows(), 1);
    let mut v = Matrix::new(w.rows(), 1);
    let mut rng: StdRng = SeedableRng::seed_from_u64(1);

    for k in 0.. {
      if self.max_iter != 0 && self.max_iter <= k {
        return Status::MaxIter;
      }

      let gfx = grad(&(0..total).into_iter().collect::<Vec<_>>(), w.slice());
      if gfx.l2_norm() < self.epsilon + x.l2_norm() {
        return Status::Epsilon;
      }

      batch_index.shuffle(&mut rng);

      for minibatch in batch_index.chunks(batch) {
        let minibatch_grad = grad(&minibatch, w.slice()).col_mat();

        m = self.beta1 * m + (1.0 - self.beta1) * minibatch_grad.clone();
        v = self.beta2 * v
          + (1.0 - self.beta2) * minibatch_grad.clone().hadamard_prod(&minibatch_grad);

        let m_hat = m.clone() * (1.0 / (1.0 - self.beta1.powi(k as i32 + 1)));
        let v_hat = v.clone() * (1.0 / (1.0 - self.beta2.powi(k as i32 + 1)));
        let v_hat_sqrt_e_inv = v_hat
          .vec()
          .iter()
          .map(|vi| 1.0 / (vi.sqrt() + self.e))
          .collect::<Vec<_>>()
          .col_mat();

        w = w - self.alpha * m_hat.hadamard_prod(&v_hat_sqrt_e_inv);

        if !w.slice().l2_norm().is_finite() {
          return Status::NaN;
        }
      }

      x.clone_from_slice(w.slice());
    }

    Status::Success
  }
}
