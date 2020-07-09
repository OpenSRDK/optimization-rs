extern crate rand;
extern crate rayon;

pub mod l_bfgs;
pub mod line_search;
pub mod prelude;
pub mod sgd_adam;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
