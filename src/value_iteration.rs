use super::mdp::*;

// TODO: extract MDP trait
// TODO: Separate out domain terminology between Grid, MDP, algorithms.

pub struct ValueIteration<'a> {
    mdp: &'a dyn MarkovDecisionProcess<'a>,
    v_init: f32,
    theta: f32,
    values: Vec<f32>,
    values_prev: Vec<f32>,
}

impl<'a> ValueIteration<'a> {
    pub fn new(mdp: &'a dyn MarkovDecisionProcess<'a>, v_init: f32, theta: f32) -> Self {
        let sn = mdp.get_states().len();
        let values = vec![v_init; sn];
        let values_prev = vec![v_init; sn];
        Self {
            mdp,
            v_init,
            theta,
            values,
            values_prev,
        }
    }

    pub fn value_iteration(&mut self, num_iterations: usize) {
        self.values.fill(self.v_init);
        self.values_prev.fill(self.v_init);
        for _ in 0..num_iterations {
            self.values_prev.copy_from_slice(&self.values);
            self.values.fill(self.v_init);
            let delta = self.one_value_iteration(self.mdp);
            if self.theta > delta {
                break;
            }
        }
    }

    pub fn get_values(&'a self) -> &'a [f32] {
        &self.values
    }

    fn one_value_iteration(&mut self, mdp: &'a dyn MarkovDecisionProcess<'a>) -> f32 {
        let mut delta = 0.;

        let states = mdp.get_states();
        for (sn, s) in states.iter().enumerate() {
            let v_prev = self.values_prev[sn];
            let v_new = Self::max(&self.expected_utility_for_all_actions(mdp, sn, s))
                .map(|x| x.1)
                .unwrap_or_default();
            delta = f32::max(delta, f32::abs(v_prev - v_new));
            self.values[sn] = v_new;
        }

        delta
    }

    fn expected_utility_for_all_actions(
        &self,
        mdp: &'a dyn MarkovDecisionProcess<'a>,
        sn: usize,
        s0: &'a State,
    ) -> Vec<f32> {
        mdp.get_actions(s0)
            .into_iter()
            .map(|a| self.expected_utility_for_one_action(mdp, sn, s0, a))
            .collect()
    }

    fn expected_utility_for_one_action(
        &self,
        mdp: &'a dyn MarkovDecisionProcess<'a>,
        sn: usize,
        s0: &'a State,
        a: &'a Action,
    ) -> f32 {
        let x = mdp.get_transitions(s0, a);

        let y: Vec<f32> = x
            .iter()
            .map(|&(_, p, r)| p * (r + self.mdp.get_gamma() * self.values_prev[sn]))
            .collect();

        y.iter().sum()
    }

    fn max(xs: &[f32]) -> Option<(usize, f32)> {
        let x = xs
            .iter()
            .rev()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b));

        // TODO: Add tests.
        x.map(|xx| (xs.len() - 1 - xx.0, *xx.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simple_golf_env::*;
    use float_eq::assert_float_eq;

    #[test]
    fn first_iteration() {
        let mdp = SimpleGolfEnv::new(0.9);
        let mut vi = ValueIteration::new(&mdp, 0., 0.00001);

        vi.value_iteration(1);

        assert_float_eq!(
            Vec::from(vi.get_values()),
            vec![0., 9., 0.],
            rmax_all <= 0.001
        );
    }

    #[test]
    fn nth_iteration() {
        let mdp = SimpleGolfEnv::new(0.9);
        let mut vi = ValueIteration::new(&mdp, 0., 0.0001);

        vi.value_iteration(6);

        assert_float_eq!(
            Vec::from(vi.get_values()),
            vec![8.80299, 9.89010, 0.],
            rmax_all <= 0.0001
        );
    }
}
