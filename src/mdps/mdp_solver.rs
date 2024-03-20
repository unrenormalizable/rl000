use std::option::*;

pub trait MdpSolver<T> {
    fn v_star(&self, state: usize) -> f32;

    fn q_star(&self, s: usize, a: usize) -> Option<f32>;

    fn pi_star(&self, state: usize) -> Option<usize>;

    fn exec(&mut self, theta: f32, num_iterations: Option<usize>) -> (T, usize);
}
