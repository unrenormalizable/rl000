use crate::Mdp;
use gymnasium::*;

pub struct GymMdpAdapter<'a> {
    name: String,
    env: &'a Environment,
    gamma: f32,
    transitions: Transitions,
}

impl<'a> GymMdpAdapter<'a> {
    pub fn new(env: &'a Environment, gamma: f32) -> Self {
        let transitions = env.transitions().unwrap();

        Self {
            name: env.name().to_string(),
            env,
            gamma,
            transitions,
        }
    }
}

impl<'a> Mdp<'a> for GymMdpAdapter<'a> {
    fn n_s(&self) -> usize {
        if let ObsActSpace::Discrete { n } = self.env.observation_space() {
            *n
        } else {
            panic!("'{}' is not an MDP.", self.name)
        }
    }

    fn n_a(&self) -> usize {
        if let ObsActSpace::Discrete { n } = self.env.action_space() {
            *n
        } else {
            panic!("'{}' is not an MDP.", self.name)
        }
    }

    fn transitions(&self) -> &Transitions {
        &self.transitions
    }

    fn gamma(&self) -> f32 {
        self.gamma
    }
}
