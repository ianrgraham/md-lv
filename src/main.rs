
pub mod simulation;
pub mod config;

use config::Config;
use simulation::Simulation;

fn main() {

    // parse command line options
    let config = Config::new();
    
    // initialize simulation box
    let mut sim = Simulation::new_from_config(&config);

    for step in 0..(config.step_max) {

        // run MD step
        sim.langevin_step();

        // write data to file
        if step % 50 == 0 {
            sim.dump_xyz();
        }
        if step % 100000 == 0 {
            println!("{}", step);
        }
    }
}