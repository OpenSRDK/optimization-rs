use rayon::prelude::*;

pub fn numerical_diff(
    func: &(dyn Fn(&[f64]) -> Result<f64, String> + Send + Sync),
    position: &[f64],
) -> Result<Vec<f64>, String> {
    const H: f64 = 0.001 * 0.001;
    const H2: f64 = 2.0 * H;

    let res = (0..position.len())
        .into_par_iter()
        .map(|i| -> Result<f64, String> {
            let mut add = position.to_vec();
            let mut sub = position.to_vec();
            add[i] += H;
            sub[i] -= H;

            Ok((func(&add)? - func(&sub)?) / H2)
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(res)
}
