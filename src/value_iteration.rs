use super::mdp::*;

pub struct ValueIteration<'a> {
    mdp: &'a dyn MarkovDecisionProcess<'a>,
    states: Vec<&'a State<'a>>,
    v_init: f32,
    theta: f32,
    values: Vec<f32>,
    values_prev: Vec<f32>,
}

impl<'a> ValueIteration<'a> {
    pub fn new(mdp: &'a dyn MarkovDecisionProcess<'a>, v_init: f32, theta: f32) -> Self {
        let mut states = mdp.get_states();
        states.sort_by(|a, b| a.id.cmp(&b.id));

        let values = vec![v_init; states.len()];
        let values_prev = vec![v_init; states.len()];
        Self {
            mdp,
            states,
            v_init,
            theta,
            values,
            values_prev,
        }
    }

    pub fn value_iteration(&mut self, num_iterations: usize) -> f32 {
        self.values.fill(self.v_init);
        self.values_prev.fill(self.v_init);
        let mut delta = 0.;
        for _ in 0..num_iterations {
            self.values_prev.copy_from_slice(&self.values);
            self.values.fill(self.v_init);
            delta = self.one_value_iteration(self.mdp);
            if self.theta > delta {
                break;
            }
        }

        delta
    }

    pub fn get_values(&'a self) -> &'a [f32] {
        &self.values
    }

    fn one_value_iteration(&mut self, mdp: &'a dyn MarkovDecisionProcess<'a>) -> f32 {
        let mut delta = 0.;

        for s in self.states.iter() {
            let v_prev = self.values_prev[s.id];
            let v_new = Self::max(&self.expected_utility_for_all_actions(mdp, s))
                .map(|x| x.1)
                .unwrap_or_default();
            delta = f32::max(delta, f32::abs(v_prev - v_new));
            self.values[s.id] = v_new;
        }

        delta
    }

    fn expected_utility_for_all_actions(
        &self,
        mdp: &'a dyn MarkovDecisionProcess<'a>,
        s: &'a State,
    ) -> Vec<f32> {
        mdp.get_actions(s)
            .into_iter()
            .map(|a| self.expected_utility_for_one_action(mdp, s, a))
            .collect()
    }

    fn expected_utility_for_one_action(
        &self,
        mdp: &'a dyn MarkovDecisionProcess<'a>,
        s: &'a State,
        a: &'a Action,
    ) -> f32 {
        mdp.get_transitions(s, a)
            .iter()
            .map(|&(s_dst, p, r)| p * (r + self.mdp.get_gamma() * self.values_prev[s_dst.id]))
            .sum()
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

        let delta = vi.value_iteration(6);

        assert_float_eq!(delta, 0.00239, rmax <= 0.001);
        assert_float_eq!(
            Vec::from(vi.get_values()),
            vec![8.80299, 9.89010, 0.],
            rmax_all <= 0.0001
        );
    }
}
