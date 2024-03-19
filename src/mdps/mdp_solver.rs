use std::*;

pub trait MdpSolver {
    fn v_star(&self, state: usize) -> f32;

    fn q_star(&self, s: usize, a: usize) -> Option<f32>;

    fn pi_star(&self, state: usize) -> Option<usize>;
}
