use super::mdp::*;
use gymnasium::*;

pub struct GymMdpAdapter<'a> {
    name: String,
    env: &'a Environment,
    gamma: f32,
    transitions: Transitions,
}

impl<'a> GymMdpAdapter<'a> {
    pub fn new(env: &'a Environment, gamma: f32) -> Self {
        let transitions = env.get_transitions().unwrap();

        Self {
            name: env.get_name().to_string(),
            env,
            gamma,
            transitions,
        }
    }
}

impl<'a> MarkovDecisionProcess<'a> for GymMdpAdapter<'a> {
    fn get_n_s(&self) -> usize {
        if let Space::Discrete { n } = self.env.get_observation_space() {
            *n
        } else {
            panic!("'{}' is not an MDP.", self.name)
        }
    }

    fn get_n_a(&self) -> usize {
        if let Space::Discrete { n } = self.env.get_action_space() {
            *n
        } else {
            panic!("'{}' is not an MDP.", self.name)
        }
    }

    fn get_transitions(&self) -> &Transitions {
        &self.transitions
    }

    fn get_gamma(&self) -> f32 {
        self.gamma
    }
}
