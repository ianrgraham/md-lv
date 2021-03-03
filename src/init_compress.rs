pub mod simulation;
pub mod config;

use config::InitCompressConfig;
use simulation::Simulation;

fn main() {

    // parse command line options
    let config = InitCompressConfig::new();

    // initialize simulation box
    let mut sim = Simulation::new_from_config(config);

    let initial_l = sim.length_from_vol(&config.vol);
    let final_l = sim.length_from_vol(&config.fvol2);
    let sigma = initial_l/final_l;

    for step in 0..config.step_max {

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
    }
}