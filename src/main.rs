mod gym_mdp_adapter;
mod mdp;
mod simple_golf_mdp;
mod value_iteration;

use gym_mdp_adapter::*;
use gymnasium::*;
use simple_golf_mdp::*;
use value_iteration::*;

fn main() {
    let mdp = SimpleGolfMdp::new(0.9);
    let mut vi = ValueIteration::new(&mdp, 0., 0.0001);
    let delta = vi.value_iteration(2);
    println!("delta = {delta:?} => {:?}", vi.get_values());

    let gc = GymClient::new("http://localhost", 5000);
    let env = gc.make_env("FrozenLake-v1", Some("ansi")).unwrap();
    let mdp = GymMdpAdapter::new(&env, 0.9);
    let mut vi = ValueIteration::new(&mdp, 0., 0.0001);
    let delta = vi.value_iteration(1000);
    println!("delta = {delta:?} => {:?}", vi.get_values());
}
