
pub mod simulation;
pub mod config;

use crate::config::{Config, ProgramMode};
use crate::simulation::*;
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
        ProgramMode::GenVariant(realizations, variants) => {

            if variants.is_none() {

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

                let mut output_integration_factors =
                    Array3::<f64>::zeros((chunk_size, write_outputs, 2));
                let mut time = Array1::<f64>::zeros(write_outputs);
                let mut output_positions =
                    Array4::<f64>::zeros((chunk_size, write_outputs, init_x.len(), 3));


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

                    let mut integration_factors = [KahanAdder::new(); 2];
                    let mut output_idx: usize = 0;

                    for step in 1..(config.step_max+1) {

                        // write data to file
                        if step % config.write_step == 0 {
                            if real == 0 { time[output_idx] = config.dt*((step-1) as f64); }
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
                        let forces_a = sim.f_system();

                        integration_factors
                            .iter_mut()
                            .zip(sim.integration_factor_gen(&forces_a, &w))
                            .for_each(|(int_factor_term, term)| {
                                *int_factor_term += term;
                            });
                
                        // print to terminal
                        if config.stdout_step.is_some() {
                            if step % config.stdout_step.unwrap() == 0 {
                                println!("{}", step);
                            }
                        }

                        sim.langevin_step_with_forces_w(&forces_a, &w);

                    }

                    if dump {
                        let out_slice = Slice::from(..=chunk_idx);
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
            }
            else {
                let variants = variants.as_ref().unwrap();
                let init_x = sim.get_positions();

                let write_outputs = config.step_max/config.write_step;

                sim.dump_hdf5_meta_gen(&config, &init_x);

                let mut names = Vec::new();
                let mut shapes: Vec<Vec<usize>> = Vec::new();

                let len_vars = variants.0.len();
                let vars = &variants.0;
                let option_q_as = variants.3.as_ref();

                let file = match &sim.file {
                    OutputWriter::HDF5File(file) => file,
                    _ => panic!()
                };
                let dataset_realizations = file.new_dataset::<usize>().create("real", 1).unwrap();
                let param_group = file.create_group("param").unwrap();
                let dataset_dvs = param_group.new_dataset::<f64>().create("dVs", vars.len()).unwrap();
                dataset_dvs.write(vars).unwrap();
                if let Some(q_as) = option_q_as {
                    let dataset_q_as = param_group.new_dataset::<f64>().create("Q_As", q_as.len()).unwrap();
                    dataset_q_as.write(q_as).unwrap();
                }


                names.push("time");
                shapes.push(vec![write_outputs]);

                names.push("Norm");
                shapes.push(vec![len_vars, write_outputs]);
                let mut norm_data = Array2::<f64>::zeros((len_vars, write_outputs));
                
                names.push("Norm2");
                shapes.push(vec![len_vars, write_outputs]);
                let mut norm2_data = Array2::<f64>::zeros((len_vars, write_outputs));


                let mut msd_data = if variants.1 {
                    names.push("MSD");
                    shapes.push(vec![len_vars, write_outputs]);

                    Some(Array2::<f64>::zeros((len_vars, write_outputs)))
                }
                else {
                    None
                };

                let mut pos_data = if variants.2 {
                    names.push("POS");
                    shapes.push(vec![len_vars, write_outputs, init_x.len(), 3]);
                    Some(Array4::<f64>::zeros((len_vars, write_outputs, init_x.len(), 3)))
                }
                else {
                    None
                };

                let mut q_data = if let Some(qs) = &variants.3 {
                    names.push("Q");
                    shapes.push(vec![len_vars, qs.len(), write_outputs]);
                    Some(Array3::<f64>::zeros((len_vars, qs.len(), write_outputs)))
                }
                else {
                    None
                };

                let (_, data_col) = sim.create_gen_hdf5_dataset_collection(
                    "data",
                    &names,
                    &shapes
                );

                let mut time_data = Array1::<f64>::zeros(write_outputs);
                let mut output_integration_factors =
                    Array2::<f64>::zeros((write_outputs, 2));
                let mut tmp_ib =
                    Array2::<f64>::zeros((len_vars, write_outputs));
                let mut output_positions =
                    Array3::<f64>::zeros((write_outputs, init_x.len(), 3));

                let init_x_arr: Array2<f64> = init_x.clone().into();

                for real in 0..(*realizations) {

                    if config.stdout_step.is_some() {
                        if real%1000 == 0 {
                            println!("Realization {}", real);
                        }
                    }

                    let mut integration_factors = [KahanAdder::new(); 2];
                    let mut output_idx: usize = 0;

                    for step in 0..(config.step_max+1) {

                        // write data to file
                        if step % config.write_step == 0 && step != 0 {
                            time_data[output_idx] = config.dt*(step as f64);
                            let factors: Array1<f64> = integration_factors.iter().map(|x| x.result()).collect::<Vec<f64>>().into();
                            output_integration_factors
                                .index_axis_mut(Axis(0), output_idx)
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
                        let forces_a = sim.f_system();

                        for (int_factor_term, term) in integration_factors
                            .iter_mut()
                            .zip(sim.integration_factor_gen(&forces_a, &w)) 
                        {
                            *int_factor_term += term;
                        }
                
                        // print to terminal
                        if config.stdout_step.is_some() {
                            if step % config.stdout_step.unwrap() == 0 {
                                println!("{}", step);
                            }
                        }

                        sim.langevin_step_with_forces_w(&forces_a, &w);

                    }

                    let traj_msd: Array1<f64> = output_positions
                        .axis_iter(Axis(0)).map(|x| (&x - &init_x_arr).fold(0.0, |accum, a| accum + a.powi(2))/(init_x_arr.nrows() as f64)).collect();

                    let part_disp: Array2<f64> = (&output_positions - &init_x_arr)
                        .fold_axis(Axis(2), 0.0, |accum, x| accum + x*x).mapv(|x| x.sqrt());

                    let mut recent_qs = Vec::new();
                    if let Some(q_as) = option_q_as {
                        for j in 0..q_as.len() {
                            let a = q_as[j];
                            let this_q = part_disp.mapv(|x| heaviside(a-x)).mean_axis(Axis(1)).unwrap();
                            recent_qs.push(this_q)
                        }
                    }

                    // Compute integration factor norm
                    for i in 0..len_vars {
                        let dv = vars[i];
                        let output: Array1<f64> = output_integration_factors.axis_iter(Axis(0)).map(|x| (-sim.beta*(dv*dv*x[0] + dv*x[1])).exp()).collect();
                        tmp_ib.index_axis_mut(Axis(0), i)
                            .assign(&output);

                        if let Some(msd) = msd_data.as_mut() {
                            let mut msd_slice = msd.slice_mut(s![i, ..]);
                            msd_slice += &(&traj_msd*&output);
                        }
                        if let Some(pos) = pos_data.as_mut() {
                            let mut pos_slice = pos.slice_mut(s![i, .., .., ..]);
                            let mut tmp_pos = Array3::<f64>::zeros(output_positions.dim());
                            for j in 0..output.len() {
                                let biased_pos = &output_positions.slice(s![j, .., ..])*output[[j]];
                                tmp_pos.slice_mut(s![j, .., ..]).assign(&biased_pos);
                            }
                            pos_slice += &tmp_pos;

                        }
                        if let Some(q) = q_data.as_mut() {
                            let q_as = option_q_as.unwrap();
                            for j in 0..q_as.len() {
                                let mut q_slice = q.slice_mut(s![i, j, ..]);
                                q_slice += &(&recent_qs[j]*&output)
                            }
                        }
                    }

                    azip!((a in &mut norm_data, &b in &tmp_ib) *a += b);
                    azip!((a in &mut norm2_data, &b in &tmp_ib) *a += b*b);

                    
                    // store data at regular intervals incase of process interuption
                    if real%10000 == 0 {
                        dataset_realizations.write(&[real+1]).unwrap();
                        sim.dump_gen_hdf5_dataset_collection(
                            &data_col,
                            time_data.view(),
                            norm_data.view(),
                            norm2_data.view(),
                            &msd_data,
                            &pos_data,
                            &q_data
                        );
                    }
                    sim.set_positions(&init_x);
                }
                // final write
                dataset_realizations.write(&[*realizations]).unwrap();
                sim.dump_gen_hdf5_dataset_collection(
                    &data_col,
                    time_data.view(),
                    norm_data.view(),
                    norm2_data.view(),
                    &msd_data,
                    &pos_data,
                    &q_data
                );
            }
        },
        ProgramMode::Equilibrate(max_dr, max_f, melt) => {
            if *melt {
                for step in 0..(config.step_max+1) {

                    // run MD step
                    sim.langevin_step();
            
                    // print to terminal
                    if config.stdout_step.is_some() {
                        if step % config.stdout_step.unwrap() == 0 {
                            println!("{}", step);
                        }
                    }
                }
            }

            let mut step = 0;
            loop {

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
                step += 1;
            }
            sim.dump_json();
        }
    }
}