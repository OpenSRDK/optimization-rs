pub fn numerical_diff(func: &dyn Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
  let res = (0..x.len())
    .into_iter()
    .map(|i| -> f64 {
      let mut add = x.to_vec();
      let mut sub = x.to_vec();
      let h = if x[i] == 0.0 { h } else { x[i].abs() * h };
      add[i] += h;
      sub[i] -= h;

      (func(&add) - func(&sub)) / 2.0 * h
    })
    .collect::<Vec<_>>();

  res
}
