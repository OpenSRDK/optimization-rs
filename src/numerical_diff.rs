use std::error::Error;

pub fn numerical_diff(
    func: &(dyn Fn(&[f64]) -> Result<f64, Box<dyn Error>>),
    x: &[f64],
) -> Result<Vec<f64>, Box<dyn Error>> {
    const H: f64 = 0.001 * 0.001;
    const H2: f64 = 2.0 * H;

    let res = (0..x.len())
        .into_iter()
        .map(|i| -> Result<f64, Box<dyn Error>> {
            let mut add = x.to_vec();
            let mut sub = x.to_vec();
            add[i] += H;
            sub[i] -= H;

            Ok((func(&add)? - func(&sub)?) / H2)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(res)
}
