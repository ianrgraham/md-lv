use rand::prelude::*;
use rand_pcg::Pcg64;
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write, BufReader};
use std::process;
use serde::*;
use ndarray::*;

use crate::config::{Config, ProgramMode, VariantConfigs, VariantConfig};

#[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct HDF5OutputMeta {
    num: usize,
    len: f64,
    temp: f64,
    time: f64,
    dim: usize,
    visc: f64,
    seed: u64,
    rscale: f64,
    vscale: f64
}

impl HDF5OutputMeta {
    
    fn new(config: &Config) -> HDF5OutputMeta {
        HDF5OutputMeta{
            num: config.num, len: config.len, dim: config.dim, time: config.time, temp: config.temp,
            visc: config.visc, seed: config.seed, rscale: config.rscale, vscale: config.vscale
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct System {
    x: Vec<[f64; 3]>,
    types: Vec<usize>,
    b: [f64; 3],
    bh: [f64; 3],
    sigmas: [f64; 2],
    dim: usize,
    vscale: f64
}

pub enum OutputWriter {
    XYZBuffer(std::io::BufWriter<std::fs::File>),
    HDF5File(hdf5::File),
    JSONFile(std::io::BufWriter<std::fs::File>)
}

// TODO
pub enum OutputMetaData {
    XYZMeta(),
    HDF5Meta(),
    JSONMeta()
}


pub struct Simulation {
    pub sys: System,
    rng: rand_pcg::Lcg128Xsl64,
    normal: rand_distr::Normal<f64>,
    a_term: f64,  // dt/visc
    b_term: f64,  // (2.0/(visc*beta)).sqrt()
    pub file: OutputWriter,
    unwrap: Option<Vec<[f64; 3]>>,
    //comp: Option<Vec<[f64; 3]>>
}


impl Simulation {

    // initialize system from Config struct
    pub fn new_from_config(config: &Config) -> Simulation {

        let seed = config.seed;
        let beta = 1./config.temp;
        let dt = config.dt;
        let visc = config.visc;
        let l: f64;

        let mut rng = Pcg64::seed_from_u64(seed);
        let normal = Normal::new(0.0f64, dt.sqrt()).unwrap();

        let sys = match &config.init_config {
            Some(path) => {
                println!("{}", path);
                let reader = BufReader::new(File::open(path).unwrap());
                let mut old_system: System = serde_json::from_reader(reader).unwrap();
                l = old_system.b[0];
                // replace potential parameters
                old_system.sigmas = [0.5*config.rscale, 0.7*config.rscale];
                old_system.vscale = config.vscale;
                old_system
            },
            None => {
                let num = config.num;
                let sigmas: [f64; 2] = [0.5*config.rscale, 0.7*config.rscale];
                let dim = config.dim;

                let mut b: [f64; 3] = [1., 1., 1.];
                let mut bh: [f64; 3] = [0.5, 0.5, 0.5];

                // compute l from volume respecting dimension of the box
                l = config.len;
                let l2 = l/2.0;
                for i in 0..dim {
                    b[i] = l;
                    bh[i] = l2;
                }

                let mut x = Vec::<[f64; 3]>::with_capacity(num);
                let mut types = Vec::<usize>::with_capacity(num);

                if dim == 3 {
                    for i in 0..num {
                        x.push([rng.gen::<f64>()*l - l2, 
                            rng.gen::<f64>()*l - l2, 
                            rng.gen::<f64>()*l - l2]);
                        if i < num/2 {
                            types.push(0);
                        }
                        else {
                            types.push(1);
                        }
                    }
                }
                else if dim == 2 {
                    for i in 0..(config.num) {
                        x.push([(rng.gen::<f64>()*l - l2), 
                        (rng.gen::<f64>()*l - l2), 0.0]);
                        if i < num/2 {
                            types.push(0);
                        }
                        else {
                            types.push(1);
                        }
                    }
                }
                else {
                    panic!("Incorrect dimension! Must be 2 or 3!");
                }
                let new_system = System{x: x, types: types, b: b, bh: bh, sigmas: sigmas, dim : dim, vscale: config.vscale};
                new_system
            }
        };
 

        let coeff: f64 = match sys.dim {
            2 => std::f64::consts::PI,
            3 => 4.0/3.0*std::f64::consts::PI,
            _ => panic!("Dimensions other than 2 and 3 should have already been ruled out!")
        };

        let vol = l.powi(sys.dim as i32);
        let part_vol = sys.types.iter().fold(0.0, |acc, idx| acc + coeff*sys.sigmas[*idx].powi(sys.dim as i32));
        let phi = part_vol/vol;

        let path = match config.mode {
            ProgramMode::Standard => {
                format!("{}/traj_{}_phi-{:.4}_rA-{:.4}_rB-{:.4}_vs-{}.xyz",
                    config.dir, config.file_suffix(), phi, sys.sigmas[0], sys.sigmas[1], sys.vscale)
            },
            ProgramMode::Variant(_, _) => {
                format!("{}/variants_n-{}_l-{}_t-{}_time-{}_dt-{:e}_visc-{}_seed-{}_phi-{:.4}_rA-{:.4}_rB-{:.4}_vs-{}.h5",
                    config.dir, sys.x.len(), l, config.temp, config.time, dt, visc, seed, phi, sys.sigmas[0], sys.sigmas[1], sys.vscale)
            },
            ProgramMode::Equilibrate(_, _) => {
                format!("{}/equil_n-{}_l-{}_dt-{:e}_visc-{}_seed-{}_phi-{:.4}_rA-{:.4}_rB-{:.4}_vs-{}.json",
                    config.dir, config.num, l, dt, visc, seed, phi, sys.sigmas[0], sys.sigmas[1], sys.vscale)
            }
        };

        if config.dryprint {
            println!("A simulation was intended to be processed with the following name, but the dryprint command was used:\n{}", path);
            process::exit(0);
        }

        let file = match config.mode {
            ProgramMode::Standard => {
                let file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .open(path)
                    .unwrap();
                OutputWriter::XYZBuffer(BufWriter::new(file))
            },
            ProgramMode::Variant(_, _) => 
                OutputWriter::HDF5File(hdf5::File::create(path).unwrap()),
            ProgramMode::Equilibrate(_, _) => {
                let file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .open(path)
                    .unwrap();
                OutputWriter::JSONFile(BufWriter::new(file))
            }
        };

        let unwrap = {
            if config.unwrap {
                Some(sys.x.to_vec())
            }
            else {
                None
            }
        };

        let sim = Simulation{sys: sys, rng: rng, normal: normal, a_term: dt/visc, b_term: (2.0/(visc*beta)).sqrt(), file: file, unwrap: unwrap};
        sim
    }

    pub fn f_system_hertz(&mut self) -> Vec<[f64; 3]> {
        let num = self.sys.x.len();
        let mut comp: f64;
        let mut f_hertz_all = Vec::<[f64; 3]>::with_capacity(num);
        for _ in 0..num {
            f_hertz_all.push([0.0, 0.0, 0.0]);
        }
        let mut dr: [f64; 3];
        let mut norm: f64;
        let mut mag: f64;
        let mut sigma: f64;
        let vscale = self.sys.vscale;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec(&i, &j);
                norm = (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]).sqrt();
                sigma = self.sys.sigmas[self.sys.types[i]] + self.sys.sigmas[self.sys.types[j]];
                if norm > sigma {
                    continue;
                }
                mag = (vscale/sigma)*(1.0-norm/sigma).powf(1.5);
                for k in 0..(self.sys.dim) {
                    comp = mag*dr[k]/norm;
                    f_hertz_all[i][k] += comp;
                    f_hertz_all[j][k] -= comp;
                }
            }

        }
        f_hertz_all
    }


    pub fn f_system_hertz_variants(&mut self, variants: &VariantConfigs) -> (Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>) {
        let num = self.sys.x.len();
        let variant_num = variants.len();
        let mut comp: f64;
        let mut f_hertz_all = Vec::<[f64; 3]>::with_capacity(num);
        let mut variant_f_hertz = Vec::<Vec<[f64; 3]>>::with_capacity(variant_num);
        for _ in 0..num {
            f_hertz_all.push([0.0, 0.0, 0.0]);
        }
        for _ in 0..variant_num {
            variant_f_hertz.push(f_hertz_all.to_vec());
        }
        let mut dr: [f64; 3];
        let mut norm: f64;
        let mut mag: f64;
        let mut sigma: f64;
        let vscale = self.sys.vscale;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec(&i, &j);
                unsafe {
                    norm = (dr.get_unchecked(0)*dr.get_unchecked(0)
                        + dr.get_unchecked(1)*dr.get_unchecked(1)
                        + dr.get_unchecked(2)*dr.get_unchecked(2)
                    ).sqrt();
                }
                sigma = self.sys.sigmas[self.sys.types[i]] + self.sys.sigmas[self.sys.types[j]];

                mag = (vscale/sigma)*((1.0-norm/sigma).max(0.0)).powf(1.5);
                for k in 0..(self.sys.dim) {
                    comp = mag*dr[k]/norm;
                    f_hertz_all[i][k] += comp;
                    f_hertz_all[j][k] -= comp;
                }
                for (l, var) in variants.configs.iter().enumerate() {
                    let tmp_sigma = sigma*var.rscale;
                    let tmp_vscale = vscale*var.vscale;
                    mag = (tmp_vscale/tmp_sigma)*((1.0-norm/tmp_sigma).max(0.0)).powf(1.5);
                    for k in 0..(self.sys.dim) {
                        comp = mag*dr[k]/norm;
                        variant_f_hertz[l][i][k] += comp;
                        variant_f_hertz[l][j][k] -= comp;
                    }
                }
            }

        }
        (f_hertz_all, variant_f_hertz)
    }

    // calculate forces with different sigma
    pub fn f_system_hertz_rescale(&mut self, rscale: &f64, vscale: &f64) -> Vec<[f64; 3]> {
        let num = self.sys.x.len();
        let mut comp: f64;
        let mut f_hertz_all = Vec::<[f64; 3]>::with_capacity(num);
        for _ in 0..num {
            f_hertz_all.push([0.0, 0.0, 0.0]);
        }
        let mut dr: [f64; 3];
        let mut norm: f64;
        let mut mag: f64;
        let mut sigma: f64;
        let c_vscale = vscale*self.sys.vscale;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec(&i, &j);
                norm = (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]).sqrt();
                sigma = (self.sys.sigmas[self.sys.types[i]] + self.sys.sigmas[self.sys.types[j]])*rscale;
                if norm > sigma {
                    continue;
                }
                mag = (c_vscale/sigma)*(1.0-norm/sigma).powf(1.5);
                for k in 0..(self.sys.dim) {
                    comp = mag*dr[k]/norm;
                    f_hertz_all[i][k] += comp;
                    f_hertz_all[j][k] -= comp;
                }
            }
        }
        f_hertz_all
    }

    pub fn integration_factor(
            &self, 
            force_bias: &Vec<[f64; 3]>, 
            w: &Vec<f64>) -> f64 {
        let mut factor = 0.0f64;
        let mut index = 0;
        for i in 0..self.sys.x.len() {
            for j in 0..self.sys.dim {
                factor += force_bias[i][j]*(
                    0.25*self.a_term*force_bias[i][j] + 
                    0.5*self.b_term*w[index]);
                index += 1;
            }
        }
        factor
    }

    fn pbc_vdr_vec(&self, i: &usize, j: &usize) -> [f64; 3] {
        let mut mdr: [f64; 3] = [0.0, 0.0, 0.0];
        let mut dr: [f64; 3] = [0.0, 0.0, 0.0];
        let x1 = self.sys.x[*i];
        let x2 = self.sys.x[*j];
    
        for i in 0..(self.sys.dim) {
            dr[i] = x1[i] - x2[i];
    
            if dr[i] >= self.sys.bh[i] {
                dr[i] -= self.sys.b[i];
            }
            else if dr[i] < -self.sys.bh[i] {
                dr[i] += self.sys.b[i];
            }
            mdr[i] += dr[i]
        }
        mdr
    }

    // use internal random number generator to fetch an index
    #[allow(dead_code)]
    fn rand_index(&mut self) -> usize {
        let i = self.rng.gen_range(0..(self.sys.x.len()));
        i
    }

    pub fn langevin_step(&mut self) {
        // calculate forces
        let forces = self.f_system_hertz();
        let dim = self.sys.dim;

        // sample normal distribution 
        let w: Vec<f64> = self.normal
            .sample_iter(&mut self.rng)
            .take(self.sys.dim*self.sys.x.len())
            .collect();
        
        // apply Euler–Maruyama method to update postions
        let mut index = 0;
        for i in 0..(self.sys.x.len()) {
            for k in 0..dim {
                self.sys.x[i][k] += 
                    self.a_term*forces[i][k] + self.b_term*w[index];
                if self.unwrap.is_some() {
                    self.unwrap.as_mut().unwrap()[i][k] += 
                        self.a_term*forces[i][k] + self.b_term*w[index];
                }
                if self.sys.x[i][k] >= self.sys.bh[k] {
                    self.sys.x[i][k] -= self.sys.b[k]
                }
                else if self.sys.x[i][k] < -self.sys.bh[k] {
                    self.sys.x[i][k] += self.sys.b[k]
                }
                index += 1;
            }
        }
    }

    pub fn gd_step(&mut self) -> (f64, f64) {
        // calculate forces
        let forces = self.f_system_hertz();
        let max_f = forces.iter().fold(
            0.0_f64,
            |acc, f| {
                let x = f[0]*f[0] + f[1]*f[1] + f[2]*f[2];
                acc.max(x)
            }
        ).sqrt();
        let dim = self.sys.dim;
        let mut max_dr = 0.0f64;
        let mut dr = 0.0f64;

        // apply forces
        for i in 0..(self.sys.x.len()) {
            for k in 0..dim {
                let tmp = self.a_term*forces[i][k];
                dr += tmp*tmp;
                self.sys.x[i][k] += tmp;
                if self.unwrap.is_some() {
                    self.unwrap.as_mut().unwrap()[i][k] += tmp;
                }
                if self.sys.x[i][k] >= self.sys.bh[k] {
                    self.sys.x[i][k] -= self.sys.b[k]
                }
                else if self.sys.x[i][k] < -self.sys.bh[k] {
                    self.sys.x[i][k] += self.sys.b[k]
                }
            }
            if dr > max_dr { max_dr = dr; }
            dr = 0.0;
        }

        (max_dr.sqrt(), max_f)
    }

    pub fn langevin_step_with_forces_w(
            &mut self, 
            forces: &Vec<[f64; 3]>, 
            w: &Vec<f64>) {

        let dim = self.sys.dim;
        
        // apply Euler–Maruyama method to update postions
        let mut index = 0;
        for i in 0..(self.sys.x.len()) {
            for k in 0..dim {
                self.sys.x[i][k] += self.a_term*forces[i][k] + self.b_term*w[index];
                if self.unwrap.is_some() {
                    self.unwrap.as_mut().unwrap()[i][k] += 
                        self.a_term*forces[i][k] + self.b_term*w[index];
                }
                if self.sys.x[i][k] >= self.sys.bh[k] {
                    self.sys.x[i][k] -= self.sys.b[k]
                }
                else if self.sys.x[i][k] < -self.sys.bh[k] {
                    self.sys.x[i][k] += self.sys.b[k]
                }
                index += 1;
            }
        }
    }

    // dump simulation state to xyz file
    pub fn dump_xyz(&mut self) {
        match &mut self.file {
            OutputWriter::XYZBuffer(file) => {

                let pos = {
                    if self.unwrap.is_some() {
                        self.unwrap.as_ref().unwrap()
                    }
                    else {
                        &self.sys.x
                    }
                };
            
                writeln!(file, "{}\n", pos.len()).expect("FILE IO ERROR!");
                
                for (x, typeid) in pos.iter().zip(&self.sys.types) {
                    writeln!(
                        file, 
                        "{} {} {} {} {}", 
                        if *typeid == 0 { "A" } else { "B" },
                        x[0], 
                        x[1], 
                        x[2],
                        self.sys.sigmas[*typeid]
                    ).expect("FILE IO ERROR!");
                }
            },
            _ => panic!("Unexpected!"),
        }
    }

    pub fn dump_hdf5_meta(&mut self, config: &Config, init_x: &Vec<[f64; 3]>, variants: &VariantConfigs) {
        let file = match &self.file {
            OutputWriter::HDF5File(file) => file,
            _ => panic!()
        };
        let group = file.create_group("meta").unwrap();
        let meta = HDF5OutputMeta::new(config);
        let meta_dataset = group.new_dataset::<HDF5OutputMeta>().create("meta", 1).unwrap();
        meta_dataset.write(&[meta.clone()]).unwrap();
        let x_arr = arr2(&(init_x.to_vec())[..]);
        let init_x_dataset = group.new_dataset::<f64>().create("init_x", x_arr.shape()).unwrap();
        init_x_dataset.write(&x_arr).unwrap();
        let vars = arr1(&(variants.configs.to_vec())[..]);
        let var_dataset = group.new_dataset::<VariantConfig>().create("variants", vars.shape()).unwrap();
        var_dataset.write(&vars).unwrap();
    }

    pub fn dump_hdf5(
            &mut self,
            realization: &usize,
            time: &Array1<f64>,
            integration_factors: &Array2<f64>,
            positions: &Array3<f64>
    ) {
        let file = match &self.file {
            OutputWriter::HDF5File(file) => file,
            _ => panic!()
        };
        let group_name = format!("{}", realization);
        let group = file.create_group(&group_name[..]).unwrap();
        let time_data = group.new_dataset::<f64>().create("time", time.shape()).unwrap();
        time_data.write(time).unwrap();
        let integration_data = group.new_dataset::<f64>().create("Ib", integration_factors.shape()).unwrap();
        integration_data.write(integration_factors).unwrap();
        let position_data = group.new_dataset::<f64>().create("pos", positions.shape()).unwrap();
        position_data.write(positions).unwrap();
    }

    pub fn dump_hdf5_to_group(
        &mut self,
        realization: &usize,
        time: &Array1<f64>,
        integration_factors: &Array2<f64>,
        positions: &Array3<f64>,
        group: &hdf5::Group
    ) {
        let group_name = format!("{}", realization);
        let group = group.create_group(&group_name[..]).unwrap();
        let time_data = group.new_dataset::<f64>().create("time", time.shape()).unwrap();
        time_data.write(time).unwrap();
        let integration_data = group.new_dataset::<f64>().create("Ib", integration_factors.shape()).unwrap();
        integration_data.write(integration_factors).unwrap();
        let position_data = group.new_dataset::<f64>().create("pos", positions.shape()).unwrap();
        position_data.write(positions).unwrap();
    }

    pub fn create_hdf5_group(
            &mut self,
            group_name: String
    ) -> hdf5::Group {
        let file = match &self.file {
            OutputWriter::HDF5File(file) => file,
            _ => panic!()
        };

        let group = file.create_group(&group_name[..]).unwrap();
        group
    }

    pub fn dump_json(&mut self) {
        let out = serde_json::to_string(&self.sys).unwrap();
        match &mut self.file {
            OutputWriter::JSONFile(file) => writeln!(file, "{}", out).expect("FILE IO ERROR!"),
            _ => panic!()
        }
    }

    // return a random vector with the the dimensions of configuration space
    pub fn rand_force_vector(&mut self) -> Vec<f64> {
        let w: Vec<f64> = self.normal.sample_iter(&mut self.rng)
            .take(self.sys.dim*self.sys.x.len())
            .collect();
        w
    }

    pub fn get_positions(&self) -> Vec<[f64; 3]> {
        let pos = {
            if self.unwrap.is_some() {
                self.unwrap.as_ref().unwrap()
            }
            else {
                &self.sys.x
            }
        };
        return pos.to_vec()
    }

    pub fn set_positions(&mut self, new_x: &Vec<[f64; 3]>) {
        if self.sys.x.len() != new_x.len() {
            return
        }
        else {
            self.sys.x = new_x.to_vec();
            if self.unwrap.is_some() {
                *self.unwrap.as_mut().unwrap() = new_x.to_vec();
            }
            return
        }
    }
}