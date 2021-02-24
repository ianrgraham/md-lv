pub mod simulation;
pub mod config;

use config::CompressConfig;
use simulation::Simulation;

fn main() {

    // parse command line options
    let config = CompressConfig::new();

    // initialize simulation box
    let mut sim = Simulation::new_from_config(config);

    let initial_l = sim.length_from_vol(&config.vol);
    let final_l = sim.length_from_vol(&config.fvol2);
    let final_sigma = initial_l/final_l;


    let mut step = 0;

    for sigma in itertools_num::linspace(1.0, final_sigma, config.step_max) {

        // run MD step
        let w = sim.rand_force_vector();
        let forces = sim.f_system_hertz_sigma(&sigma);
        sim.langevin_step_with_forces_w(&forces, &w);

        // write data to file
        if step % config.write_step == 0 {
            sim.dump_xyz();
        }

        // print to terminal
        if step % config.stdout_step == 0 {
            println!("{}", step);
        }

        step += 1;
    }
}