use super::math;
use super::mdp::*;
use super::mdp_solver::*;
use gymnasium::*;

pub struct ValueIteration<'a> {
    n_s: usize,
    n_a: usize,
    transitions: &'a Transitions,
    gamma: f32,
    v_init: f32,
    theta: f32,
    values: Vec<f32>,
    values_prev: Vec<f32>,
}

impl<'a> MdpSolver for ValueIteration<'a> {
    fn v_star(&self, state: usize) -> f32 {
        self.values[state]
    }

    fn q_star(&self, s: usize, a: usize) -> Option<f32> {
        self.q(&self.values, s, a)
    }

    fn pi_star(&self, state: usize) -> Option<usize> {
        let max = self.q_for_all_actions(&self.values, state);
        max.1.map_or_else(|| None, |_| Some(max.0))
    }
}

impl<'a> ValueIteration<'a> {
    pub fn new(mdp: &'a dyn Mdp<'a>, v_init: f32, theta: f32) -> Self {
        let n_s = mdp.get_n_s();
        let n_a = mdp.get_n_a();
        let transitions = mdp.get_transitions();
        let gamma = mdp.get_gamma();
        let values = vec![v_init; n_s];
        let values_prev = vec![v_init; n_s];
        Self {
            n_s,
            n_a,
            transitions,
            v_init,
            gamma,
            theta,
            values,
            values_prev,
        }
    }

    pub fn value_iteration(&mut self, num_iterations: usize) -> (f32, usize) {
        self.values.fill(self.v_init);
        self.values_prev.fill(self.v_init);
        let mut delta = 0.;
        let mut iter_done = num_iterations;
        for i in 0..num_iterations {
            self.values_prev.copy_from_slice(&self.values);
            self.values.fill(self.v_init);
            delta = self.one_value_iteration();
            if self.theta > delta {
                iter_done = i;
                break;
            }
        }

        self.values_prev.fill(self.v_init);
        (delta, iter_done)
    }

    pub fn values(&'a self) -> &'a [f32] {
        &self.values
    }

    fn one_value_iteration(&mut self) -> f32 {
        let mut delta = 0.;

        for s in 0..self.n_s {
            let v_new = self
                .q_for_all_actions(&self.values_prev, s)
                .1
                .unwrap_or_default();
            delta = f32::max(delta, f32::abs(self.values_prev[s] - v_new));
            self.values[s] = v_new;
        }

        delta
    }

    fn q_for_all_actions(&self, v: &Vec<f32>, s: usize) -> (usize, Option<f32>) {
        let qs: Vec<_> = (0..self.n_a).map(|a| self.q(v, s, a)).collect();

        math::max(&qs)
    }

    fn q(&self, v: &Vec<f32>, s: usize, a: usize) -> Option<f32> {
        if let Some(ts) = self.transitions.get(&(s, a)) {
            let q = ts
                .iter()
                .map(|t| t.probability * (t.reward + self.gamma * v[t.next_state]))
                .sum();
            Some(q)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simple_golf_mdp::*;
    use float_eq::assert_float_eq;

    #[test]
    fn first_iteration() {
        let mdp = SimpleGolfMdp::new(0.9);
        let mut vi = ValueIteration::new(&mdp, 0., 0.00001);

        vi.value_iteration(1);

        assert_float_eq!(Vec::from(vi.values()), vec![0., 9., 0.], rmax_all <= 0.001);
    }

    #[test]
    fn nth_iteration() {
        let mdp = SimpleGolfMdp::new(0.9);
        let mut vi = ValueIteration::new(&mdp, 0., 0.0001);

        let (delta, iterations) = vi.value_iteration(6);

        assert_eq!(iterations, 6);
        assert_float_eq!(delta, 0.00239, rmax <= 0.001);
        assert_float_eq!(
            Vec::from(vi.values()),
            vec![8.80299, 9.89010, 0.],
            rmax_all <= 0.0001
        );
    }

    #[test]
    fn test_policy() {
        let mdp = SimpleGolfMdp::new(0.9);
        let mut vi = ValueIteration::new(&mdp, 0., 0.0001);

        vi.value_iteration(6);

        assert_eq!(vi.pi_star(0), Some(0));
        assert_eq!(vi.pi_star(1), Some(2));
        assert_eq!(vi.pi_star(2), None);
    }

    // check for q_star
}
