use std::hash::Hash;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct State<'a> {
    pub id: usize,
    pub name: &'a str,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Action<'a> {
    pub id: usize,
    pub name: &'a str,
}

pub type Probability = f32;

pub type Reward = f32;

/// Markov Decision Process - Sutton & Barto 2018.
pub trait MarkovDecisionProcess<'a> {
    fn get_states(&'a self) -> Vec<&'a State<'a>>;

    fn get_actions(&'a self, state: &State) -> Vec<&'a Action<'a>>;

    fn get_transitions(
        &'a self,
        state: &'a State,
        action: &'a Action,
    ) -> Vec<(&'a State<'a>, Probability, Reward)>;

    fn get_gamma(&self) -> f32;
}
