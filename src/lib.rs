#[cfg(test)]
extern crate blas_src;
#[cfg(test)]
extern crate lapack_src;
extern crate opensrdk_linear_algebra;
extern crate rand;
extern crate rayon;
extern crate thiserror;

pub mod lbfgs;
pub mod line_search;
pub mod numerical_diff;
pub mod sgd_adam;
pub mod vec;

pub use crate::lbfgs::*;
pub use crate::line_search::*;
pub use crate::numerical_diff::*;
pub use crate::sgd_adam::*;

pub enum Status {
  Success,
  Delta,
  Epsilon,
  MaxIter,
  NaN,
}
