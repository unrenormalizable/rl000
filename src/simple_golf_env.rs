use super::mdp::*;
use std::collections::{HashMap, HashSet};

type Transition<'a> = (&'a State<'a>, Probability, Reward);
type TransitionWithId<'a> = (usize, Probability, Reward);
type TransitionMap<'a> = HashMap<(usize, usize), Vec<TransitionWithId<'a>>>;

/// https://towardsdatascience.com/reinforcement-learning-an-easy-introduction-to-value-iteration-e4cfe0731fd5
pub struct SimpleGolfEnv<'a> {
    gamma: f32,
    states: HashMap<usize, State<'a>>,
    actions: HashMap<usize, Action<'a>>,
    transitions: TransitionMap<'a>,
}

impl<'a> SimpleGolfEnv<'a> {
    pub fn new(gamma: f32) -> Self {
        let states = HashMap::<usize, State>::from([
            (
                0,
                State {
                    id: 0,
                    name: "s0: fairway",
                },
            ),
            (
                1,
                State {
                    id: 1,
                    name: "s1: green",
                },
            ),
            (
                2,
                State {
                    id: 2,
                    name: "s2: hole",
                },
            ),
        ]);
        let actions = HashMap::<usize, Action>::from([
            (
                0,
                Action {
                    id: 0,
                    name: "hit to green",
                },
            ),
            (
                1,
                Action {
                    id: 1,
                    name: "hit to fairway",
                },
            ),
            (
                2,
                Action {
                    id: 2,
                    name: "hit in hole",
                },
            ),
        ]);

        let transitions = HashMap::from([
            (
                (states[&0].id, actions[&0].id),
                vec![(states[&1].id, 0.9, 0.), (states[&0].id, 0.1, 0.)],
            ),
            (
                (states[&1].id, actions[&1].id),
                vec![(states[&0].id, 0.9, 0.), (states[&1].id, 0.1, 0.)],
            ),
            (
                (states[&1].id, actions[&2].id),
                vec![(states[&2].id, 0.9, 10.), (states[&1].id, 0.1, 0.)],
            ),
        ]);

        Self {
            gamma,
            states,
            actions,
            transitions,
        }
    }
}

impl<'a> MarkovDecisionProcess<'a> for SimpleGolfEnv<'a> {
    fn get_states(&'a self) -> Vec<&'a State<'a>> {
        self.states.values().collect()
    }

    fn get_actions(&'a self, state: &State) -> Vec<&'a Action<'a>> {
        self.transitions
            .keys()
            .filter(|k| k.0 == state.id)
            .map(|k| &self.actions[&k.1])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    }

    fn get_transitions(&'a self, state: &'a State, action: &'a Action) -> Vec<Transition<'a>> {
        if self.transitions.contains_key(&(state.id, action.id)) {
            return self.transitions[&(state.id, action.id)]
                .iter()
                .map(|&(id, p, r)| (&self.states[&id], p, r))
                .collect();
        }

        vec![]
    }

    fn get_gamma(&self) -> f32 {
        self.gamma
    }
}
