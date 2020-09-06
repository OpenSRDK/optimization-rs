use rayon::prelude::*;

pub trait Vector {
    fn l2_norm(&self) -> f64;
}

impl Vector for [f64] {
    fn l2_norm(&self) -> f64 {
        self.into_par_iter()
            .map(|xi| xi.powi(2))
            .sum::<f64>()
            .sqrt()
    }
}
