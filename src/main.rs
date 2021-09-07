
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

            // let group = sim.create_hdf5_group(String::from("realizations"));
            let data_col = sim.create_hdf5_dataset_collection(
                "data",
                &[write_outputs],
                &[*realizations, write_outputs, variants],
                &[*realizations, write_outputs, init_x.len(), 3]
            );


            for real in 0..(*realizations) {
                if config.stdout_step.is_some() {
                    if real%1000 == 0 {
                        println!("Realization {}", real);
                    }
                }
                let mut output_integration_factors =
                    Array2::<f64>::zeros((write_outputs, variants));
                // let mut integration_factors = Array1::<f64>::zeros(variants);
                let mut integration_factors = vec![KahanAdder::new(); variants];
                let mut time = Array1::<f64>::zeros(write_outputs);
                let mut output_positions =
                    Array3::<f64>::zeros((write_outputs, init_x.len(), 3));

                let mut output_idx: usize = 0;

                for step in 1..(config.step_max+1) {

                    // write data to file
                    if step % config.write_step == 0 {
                        time[output_idx] = config.dt*((step-1) as f64);
                        let factors: Array1<f64> = integration_factors.iter().map(|x| x.result()).collect::<Vec<f64>>().into();
                        output_integration_factors.index_axis_mut(Axis(0), output_idx)
                            .assign(&factors);
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
                // sim.dump_hdf5_to_group(&real, &time, &output_integration_factors, &output_positions, &group);
                sim.dump_hdf5_slices_to_dataset(
                    &real,
                    time.view(),
                    output_integration_factors.view(),
                    output_positions.view(),
                    &data_col
                );
                sim.set_positions(&init_x);
            }
        },
        ProgramMode::GenVariant(realizations) => {

            let init_x = sim.get_positions();

            let write_outputs = config.step_max/config.write_step;
            let chunk_size: usize = 1000;

            sim.dump_hdf5_meta_gen(&config, &init_x);

            let data_col = sim.create_hdf5_dataset_collection(
                "data",
                &[write_outputs],
                &[*realizations, write_outputs, 2],
                &[*realizations, write_outputs, init_x.len(), 3]
            );

            let mut start = 0;
            let mut end = 0;
            let mut dump = false;


            for real in 0..(*realizations) {

                let chunk_idx = real%chunk_size;
                if chunk_idx == 0 {
                    start = real;
                    dump = false;
                }
                else if chunk_idx == chunk_size - 1 || real == realizations - 1 {
                    end = real + 1;
                    dump = true;
                }

                if config.stdout_step.is_some() {
                    if real%1000 == 0 {
                        println!("Realization {}", real);
                    }
                }
                let mut output_integration_factors =
                    Array3::<f64>::zeros((chunk_size, write_outputs, 2));
                // let mut integration_factors = Array1::<f64>::zeros(variants);
                let mut integration_factors = vec![KahanAdder::new(); 2];
                let mut time = Array1::<f64>::zeros(write_outputs);
                let mut output_positions =
                    Array4::<f64>::zeros((chunk_size, write_outputs, init_x.len(), 3));

                let mut output_idx: usize = 0;

                for step in 1..(config.step_max+1) {

                    // write data to file
                    if step % config.write_step == 0 {
                        time[output_idx] = config.dt*((step-1) as f64);
                        let factors: Array1<f64> = integration_factors.iter().map(|x| x.result()).collect::<Vec<f64>>().into();
                        output_integration_factors.index_axis_mut(Axis(0), chunk_idx)
                            .index_axis_mut(Axis(0), output_idx)
                            .assign(&factors);
                        let tmp_pos = sim.get_positions();
                        for i in 0..tmp_pos.len() {
                            for j in 0..3 {
                                output_positions[[chunk_idx, output_idx, i, j]] = tmp_pos[i][j];
                            }
                        }
                        output_idx += 1;
                    }

                    // run MD step
                    let w = sim.rand_force_vector();
                    let forces_a = sim.f_system_hertz();

                    for (int_factor_term, term) in integration_factors
                        .iter_mut()
                        .zip(sim.integration_factor_gen(&forces_a, &w)) 
                    {
                        *int_factor_term += term;
                    }
                    // integration_factors[i] += ;
            
                    // print to terminal
                    if config.stdout_step.is_some() {
                        if step % config.stdout_step.unwrap() == 0 {
                            println!("{}", step);
                        }
                    }

                    sim.langevin_step_with_forces_w(&forces_a, &w);

                }
                // sim.dump_hdf5_to_group(&real, &time, &output_integration_factors, &output_positions, &group);
                if dump {
                    let out_slice = Slice::from(0..=chunk_idx);
                    sim.dump_hdf5_large_slices_to_dataset(
                        &start,
                        &end,
                        time.view(),
                        output_integration_factors.slice_axis(Axis(0), out_slice),
                        output_positions.slice_axis(Axis(0), out_slice),
                        &data_col
                    );
                }
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

use std::ops::AddAssign;

#[derive(Default, Debug, Copy, Clone, PartialEq)]
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

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.accum = 0.0;
        self.comp = 0.0;
    }
}

impl AddAssign<f64> for KahanAdder {
    fn add_assign(&mut self, other: f64) {
        self.add(&other);
    }
}