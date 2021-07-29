
pub mod simulation;
pub mod config;

use crate::config::{Config, ProgramMode};
use crate::simulation::Simulation;
use ndarray::*;

fn main() {

    // parse command line options
    let config = Config::new();
    
    // initialize the simulation
    let mut sim = Simulation::new_from_config(&config);

    match &config.mode {
        ProgramMode::Standard => {
            for step in 0..(config.step_max+1) {

                // run MD step
                sim.langevin_step();
        
                // write data to file
                if step % config.write_step == 0 {
                    sim.dump_xyz();
                }
                
                // print to terminal
                if config.stdout_step.is_some() {
                    if step % config.stdout_step.unwrap() == 0 {
                        println!("{}", step);
                    }
                }
            }
        },
        ProgramMode::Variant(variant_config, realizations) => {

            let init_x = sim.get_positions();

            let write_outputs = config.step_max/config.write_step;
            let variants = variant_config.len();

            sim.dump_hdf5_meta(&config, &init_x, &variant_config);

            let group = sim.create_hdf5_group(String::from("realizations"));


            for real in 0..(*realizations) {
                if config.stdout_step.is_some() {
                    if real%100 == 0 {
                        println!("Realization {}", real);
                    }
                }
                let mut output_integration_factors =
                    Array2::<f64>::zeros((write_outputs, variants));
                let mut integration_factors = Array1::<f64>::zeros(variants);
                let mut time = Array1::<f64>::zeros(write_outputs);
                let mut output_positions =
                    Array3::<f64>::zeros((write_outputs, init_x.len(), 3));

                let mut output_idx: usize = 0;

                for step in 1..(config.step_max+1) {

                    // write data to file
                    if step % config.write_step == 0 {
                        time[output_idx] = config.dt*((step-1) as f64);
                        output_integration_factors.index_axis_mut(Axis(0), output_idx)
                            .assign(&integration_factors);
                        let tmp_pos = sim.get_positions();
                        for i in 0..tmp_pos.len() {
                            for j in 0..3 {
                                output_positions[[output_idx, i, j]] = tmp_pos[i][j];
                            }
                        }
                        output_idx += 1;
                    }

                    // run MD step
                    let w = sim.rand_force_vector();
                    let (forces_a, forces_b) = sim.f_system_hertz_variants(&variant_config);

                    for i in 0..variants {
                        let bias_force = forces_a.iter().zip(&forces_b[i])
                            .map(|(a, b)| [a[0]-b[0], a[1]-b[1], a[2]-b[2]]).collect();
                        integration_factors[i] += sim.integration_factor(&bias_force, &w);
                    }
            
                    // print to terminal
                    if config.stdout_step.is_some() {
                        if step % config.stdout_step.unwrap() == 0 {
                            println!("{}", step);
                        }
                    }

                    sim.langevin_step_with_forces_w(&forces_a, &w);

                }
                sim.dump_hdf5_to_group(&real, &time, &output_integration_factors, &output_positions, &group);
                sim.set_positions(&init_x);
            }
        },
        ProgramMode::Equilibrate(max_dr, max_f) => {
            for step in 0..(config.step_max+1) {

                // run MD step
                let (cur_max_dr, cur_max_f) = sim.gd_step();
        
                // print to terminal
                if config.stdout_step.is_some() {
                    if step % config.stdout_step.unwrap() == 0 {
                        println!("{}", step);
                    }
                }

                if cur_max_dr/config.dt < *max_dr && cur_max_f < *max_f {
                    break;
                }
            }
            sim.dump_json();
        }
    }
}

struct KahanAdder {
    accum: f64,
    comp: f64
}

impl KahanAdder {

    fn new() -> Self {
        KahanAdder{accum: 0.0, comp: 0.0}
    }

    fn add(&mut self, num: &f64) {
        let y = num - self.comp;
        let t = self.accum + y;
        self.comp = (t - self.accum) - y;
        self.accum = t;
    }

    fn result(&self) -> f64 {
        self.accum.clone()
    }

    fn reset(&mut self) {
        self.accum = 0.0;
        self.comp = 0.0;
    }
}