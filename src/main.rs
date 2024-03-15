mod mdp;
mod simple_golf_env;
mod value_iteration;

use simple_golf_env::*;
use value_iteration::*;

fn main() {
    let mdp = SimpleGolfEnv::new(0.9);
    let mut vi = ValueIteration::new(&mdp, 0., 0.0001);
    let delta = vi.value_iteration(1000);

    println!("delta = {delta} => {:?}", vi.get_values());
}
