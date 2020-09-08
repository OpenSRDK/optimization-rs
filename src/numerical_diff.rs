use std::error::Error;

pub fn numerical_diff(
    func: &(dyn Fn(&[f64]) -> Result<f64, Box<dyn Error>>),
    x: &[f64],
    h: f64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let res = (0..x.len())
        .into_iter()
        .map(|i| -> Result<f64, Box<dyn Error>> {
            let mut add = x.to_vec();
            let mut sub = x.to_vec();
            add[i] += h;
            sub[i] -= h;

            Ok((func(&add)? - func(&sub)?) / 2.0 * h)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(res)
}
