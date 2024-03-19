use gymnasium::*;

/// Markov Decision Process - Sutton & Barto 2018.
pub trait Mdp<'a> {
    fn get_n_s(&self) -> usize;

    fn get_n_a(&self) -> usize;

    fn get_transitions(&'a self) -> &'a Transitions;

    fn get_gamma(&self) -> f32;
}
