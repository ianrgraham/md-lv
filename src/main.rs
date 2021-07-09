pub mod simulation;
pub mod config;

use config::Config;
use simulation::Simulation;

fn main() {

    // parse command line options
    let config = Config::new();
    
    // initialize simulation box
    let mut sim = Simulation::new_from_config(config); // consumes config

    for step in 1..(sim.config.step_max+1) {

        // run MD step
        sim.langevin_step();

        // write data to file
        if step % sim.config.write_step == 0 {
            sim.dump_xyz();
        }

        // print to terminal
        if step % sim.config.stdout_step == 0 {
            println!("{}", step);
        }
    }
}