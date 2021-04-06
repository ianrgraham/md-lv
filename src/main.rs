pub mod simulation;
pub mod config;

use config::StdConfig;
use simulation::Simulation;

fn main() {

    // parse command line options
    let config = StdConfig::new();
    
    // initialize simulation box
    let mut sim = Simulation::new_from_config(config);

    for step in 0..(config.step_max) {

        // run MD step
        sim.langevin_step();

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