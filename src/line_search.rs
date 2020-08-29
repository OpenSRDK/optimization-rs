use rayon::prelude::*;

pub fn line_search(
    func_grad: &dyn Fn(&[f64]) -> Result<(f64, Vec<f64>), String>,
    x: &[f64],
    direction: &[f64],
    initial_step_size: f64,
) -> Result<f64, String> {
    let mut step_size = initial_step_size;
    let armijo_param = 0.4;
    let curvature_param = 0.6;
    let mut i = 1;

    loop {
        i += 1;
        let moved_position = x
            .par_iter()
            .zip(direction.par_iter())
            .map(|(x_e, direction_e)| x_e + step_size * direction_e)
            .collect::<Vec<_>>();

        let tmp1 = func_grad(x)?;
        let tmp2 = func_grad(&moved_position)?;

        let grad_dot_direction = tmp1
            .1
            .par_iter()
            .zip(direction.par_iter())
            .map(|(x_e, direction_e)| x_e * direction_e)
            .sum::<f64>();

        // Armijo condition
        let armijo_left = tmp2.0;
        let armijo_right = tmp1.0 + armijo_param * step_size * grad_dot_direction;

        if armijo_left <= armijo_right {
            step_size -= initial_step_size / i as f64;

            continue;
        }

        // Curvature condition
        let curvature_left = curvature_param * grad_dot_direction;
        let curvature_right = tmp2
            .1
            .par_iter()
            .zip(direction.par_iter())
            .map(|(x_e, direction_e)| x_e * direction_e)
            .sum();

        if curvature_left < curvature_right {
            step_size += initial_step_size / i as f64;

            continue;
        }

        break;
    }

    Ok(step_size)
}
