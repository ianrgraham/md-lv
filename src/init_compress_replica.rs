pub mod simulation;
pub mod config;

use config::InitCompressReplicaConfig;
use simulation::Simulation;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

fn main() {

    // parse command line options
    let config = InitCompressReplicaConfig::new();

    // init I_b log buffer
    let mut log_file = init_log_buffer(&config);
    
    // initialize simulation box
    let mut sim = Simulation::new_from_config(config);

    // some final configuration before going through simulation steps
    let final_sigma = get_final_sigma(&sim, &config);
    let mut integration_factor = 0.0f64;
    let sigma_b = final_sigma;

    // run through simulation steps while slowly compressing the system
    for step in 0..config.step_max {

        // run MD step
        let w = sim.rand_force_vector();
        let forces_a = sim.f_system_hertz();
        let forces_b = sim.f_system_hertz_sigma(&sigma_b);
        let mut force_bias = forces_a.to_vec();
        for i in 0..force_bias.len() {
            for j in 0..sim.dim {
                force_bias[i][j] = forces_a[i][j] - forces_b[i][j];
            }
        }

        integration_factor += sim.integration_factor(&force_bias, &w);

        sim.langevin_step_with_forces_w(&forces_a, &w);

        // write data to file
        if step % config.write_step == 0 {
            sim.dump_xyz();
            dump_log(&mut log_file, format!("{}", integration_factor));
        }

        // print to terminal
        if step % config.stdout_step == 0 {
            println!("{}", step);
        }
    }
}

fn dump_log(file: &mut std::io::BufWriter<std::fs::File>, message: String) {
    writeln!(file, "{}", message).expect("FILE IO ERROR!");
}

fn get_final_sigma(sim: &Simulation, config: &InitCompressReplicaConfig) -> f64 {
    let initial_l = sim.length_from_vol(&config.vol);
    let final_l = sim.length_from_vol(&config.fvol2);
    let final_sigma = initial_l/final_l;
    final_sigma
}

fn init_log_buffer(config: &InitCompressReplicaConfig) -> std::io::BufWriter<std::fs::File> {
    let log_file = OpenOptions::new()
    .write(true)
    .create(true)
    .open(format!("log_Ib_{}.dat", config.format_file_suffix()))
    .unwrap();
    let log_file = BufWriter::new(log_file);
    log_file
}