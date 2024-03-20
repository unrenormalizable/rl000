use crate::math;
use gymnasium::*;

pub fn q_for_all_actions(
    transitions: &Transitions,
    n_a: usize,
    gamma: f32,
    v: &[f32],
    s: usize,
) -> (usize, Option<f32>) {
    let qs: Vec<_> = (0..n_a).map(|a| q(transitions, gamma, v, s, a)).collect();

    math::max(&qs)
}

pub fn q(transitions: &Transitions, gamma: f32, v: &[f32], s: usize, a: usize) -> Option<f32> {
    if let Some(ts) = transitions.get(&(s, a)) {
        let q = ts
            .iter()
            .map(|t| t.probability * (t.reward + gamma * v[t.next_state]))
            .sum();
        Some(q)
    } else {
        None
    }
}
