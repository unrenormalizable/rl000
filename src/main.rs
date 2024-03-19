mod environments;
mod math;
mod mdps;

use environments::gym_mdp_adapter::*;
use environments::simple_golf_mdp::*;
use gymnasium::*;
use mdps::mdp::*;
use mdps::mdp_solver::*;
use mdps::value_iteration::*;

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

    let v_star = (0..mdp.n_s()).map(|s| vi.v_star(s)).collect::<Vec<_>>();
    println!("{v_star:?}");
    let pi_star = (0..mdp.n_s()).map(|s| vi.pi_star(s)).collect::<Vec<_>>();
    println!("{pi_star:?}");
    let mut q_star = Vec::new();
    for s in 0..mdp.n_s() {
        for a in 0..mdp.n_a() {
            q_star.push(vi.q_star(s, a))
        }
    }
    println!("{q_star:?}");

    let mut total_reward = 0.;
    let mut curr_state = env.reset(None).unwrap()[0].discrete_value();

    for i in 0.. {
        let curr_action = mdp_solver.pi_star(curr_state).unwrap();
        let step_info = env
            .step(vec![ObsActSpaceItem::Discrete(curr_action)])
            .unwrap();
        let render_frame = env.render().unwrap();
        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
        println!("{}", render_frame);
        assert_eq!(
            step_info.observation.len(),
            env.observation_space().sample().len()
        );
        total_reward += step_info.reward;

        if step_info.truncated || step_info.terminated {
            break;
        }

        let next_state = step_info.observation[0].discrete_value();
        println!("Info: {i}\n  Action taken : {curr_action}\n  Current state: {curr_state}\n  Next state   : {next_state}\n  Total rewards: {total_reward}");
        curr_state = next_state;
        std::thread::sleep(std::time::Duration::from_millis(2000));
    }
}
