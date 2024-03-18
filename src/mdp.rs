use gymnasium::*;
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

/// Markov Decision Process - Sutton & Barto 2018.
pub trait MarkovDecisionProcess<'a> {
    fn get_n_s(&self) -> usize;

    fn get_n_a(&self) -> usize;

    fn get_transitions(&'a self) -> &'a Transitions;

    fn get_gamma(&self) -> f32;
}
