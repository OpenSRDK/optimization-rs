use rayon::prelude::*;

pub fn line_search(
    position: &[f64],
    func: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    direction: &[f64],
    initial_step_width: f64,
) -> f64 {
    let mut step_width = initial_step_width;
    let armijo_param = 0.4;
    let curvature_param = 0.6;
    let mut i = 1;

    loop {
        i += 1;
        let moved_position = position
            .par_iter()
            .zip(direction.par_iter())
            .map(|(x_e, direction_e)| x_e + step_width * direction_e)
            .collect::<Vec<_>>();

        let grad_dot_direction = grad(position)
            .par_iter()
            .zip(direction.par_iter())
            .map(|(x_e, direction_e)| x_e * direction_e)
            .sum::<f64>();

        // Armijo condition
        let armijo_left = func(&moved_position);
        let armijo_right = func(position) + armijo_param * step_width * grad_dot_direction;

        if armijo_left <= armijo_right {
            step_width -= initial_step_width / i as f64;

            continue;
        }

        // Curvature condition
        let curvature_left = curvature_param * grad_dot_direction;
        let curvature_right = grad(&moved_position)
            .par_iter()
            .zip(direction.par_iter())
            .map(|(x_e, direction_e)| x_e * direction_e)
            .sum();

        if curvature_left < curvature_right {
            step_width += initial_step_width / i as f64;

            continue;
        }

        break;
    }

    step_width
}
