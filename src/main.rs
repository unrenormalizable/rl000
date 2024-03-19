mod gym_mdp_adapter;
mod math;
mod mdp;
mod mdp_solver;
mod simple_golf_mdp;
mod value_iteration;

use gym_mdp_adapter::*;
use gymnasium::*;
use mdp::*;
use mdp_solver::*;
use simple_golf_mdp::*;
use value_iteration::*;

fn main() {
    let mdp = &SimpleGolfMdp::new(0.9) as &dyn Mdp;
    let mut vi = ValueIteration::new(mdp, 0., 0.0001);
    let delta = vi.value_iteration(2);
    println!("delta = {delta:?} => {:?}", vi.values());

    let gc = GymClient::new("http://localhost", 5000);
    let env = gc.make_env("FrozenLake-v1", Some("ansi")).unwrap();
    let mdp = &GymMdpAdapter::new(&env, 0.9) as &dyn Mdp;
    let mut vi = ValueIteration::new(mdp, 0., 0.0001);
    let delta = vi.value_iteration(1000);
    println!("delta = {delta:?} => {:?}", vi.values());

    let mdp_solver = &vi as &dyn MdpSolver;
    let mut total_reward = 0.;
    let mut current_state = 0;
    let x = env.reset(None).unwrap();

    loop {
        let action = mdp_solver.pi_star(current_state).unwrap();
        //println!(">>>> {action}");
        let step_info = env.step(vec![SampleItem::USize(action)]).unwrap();
        let render_frame = env.render().unwrap();
        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
        println!("{}", render_frame);
        assert_eq!(
            step_info.observation.len(),
            env.get_observation_space().sample().len()
        );
        total_reward += step_info.reward;

        if step_info.truncated || step_info.terminated {
            break;
        }

        current_state = mdp.get_transitions().get(&(current_state, action)).unwrap().iter().max_by(|a, b| a.probability.total_cmp(&b.probability)).unwrap().next_state;
        println!("Next state {current_state}, total rewards {total_reward}");
        //std::thread::sleep(std::time::Duration::from_millis(2000));
    }
}
