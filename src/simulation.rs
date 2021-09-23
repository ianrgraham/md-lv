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

use Potential::*;

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

pub struct HDF5VarDatasetCol {
    time: hdf5::Dataset,
    position: hdf5::Dataset,
    integration: hdf5::Dataset
}

impl HDF5OutputMeta {
    
    fn new(config: &Config) -> HDF5OutputMeta {
        HDF5OutputMeta{
            num: config.num, len: config.len, dim: config.dim, time: config.time, temp: config.temp,
            visc: config.visc, seed: config.seed, rscale: config.rscale, vscale: config.vscale
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum Potential {
    Hertz,
    LJ
}

impl Potential {

    pub fn from_str(string: &str) -> Option<Self> {
        match string {
            "hertz" => { Some(Hertz) },
            "lj" => { Some(LJ) },
            _ => { None }
        }
    }

    pub fn to_str(&self) -> &str {
        let strs = Potential::valid_strs();
        match self {
            Hertz => { strs[0] },
            LJ => { strs[1] }
        }
    }

    pub fn id(&self) -> usize {
        match self {
            Hertz => { 0 },
            LJ => { 1 }
        }
    }

    pub fn valid_strs() -> &'static [&'static str] {
        &["hertz", "lj"]
    }
}

#[derive(Serialize, Deserialize)]
pub struct System {
    x: Vec<[f64; 3]>,
    types: Vec<usize>,
    b: [f64; 3],
    bh: [f64; 3],
    sigmas: Box<[f64]>,
    dim: usize,
    vscale: f64,
    potential: Potential,
    numa: usize
}

fn calc_vol(len: f64, dim: usize) -> f64 {
    len.powi(dim as i32)
}

fn calc_l_from_phi(config: &Config, types: &Vec<usize>, sigmas: &Box<[f64]>, target_phi: f64) -> f64 {
    match config.potential {
        Potential::Hertz => {
            // // computes volume area fraction
            // let coeff: f64 = match config.dim {
            //     2 => std::f64::consts::PI,
            //     3 => 4.0/3.0*std::f64::consts::PI,
            //     _ => panic!("Dimensions other than 2 and 3 should have already been ruled out!")
            // };

            
            // let part_vol = types.iter().fold(0.0, |acc, idx| acc + coeff*sigmas[*idx].powi(config.dim as i32));
            // (part_vol/target_phi).powf(1.0/(config.dim as f64))
            ((config.num as f64)/target_phi).powf(1.0/(config.dim as f64))
        }, 
        Potential::LJ => {
            // computes simple N/[D], where [D] is the unit of distance
            ((config.num as f64)/target_phi).powf(1.0/(config.dim as f64))
        }
    }
}

impl System {

    fn calc_phi(&self, vol: f64) -> f64 {
        match self.potential {
            Potential::Hertz => {
                // computes volume area fraction
                // let coeff: f64 = match self.dim {
                //     2 => std::f64::consts::PI,
                //     3 => 4.0/3.0*std::f64::consts::PI,
                //     _ => panic!("Dimensions other than 2 and 3 should have already been ruled out!")
                // };

                
                // let part_vol = self.types.iter().fold(0.0, |acc, idx| acc + coeff*self.sigmas[*idx].powi(self.dim as i32));
                // part_vol/vol
                (self.x.len() as f64)/vol
            }, 
            Potential::LJ => {
                // computes simple N/[D], where [D] is the unit of distance
                (self.x.len() as f64)/vol
            }
        }
    }
}

pub enum OutputWriter {
    XYZBuffer(std::io::BufWriter<std::fs::File>),
    HDF5File(hdf5::File),
    JSONFile(std::io::BufWriter<std::fs::File>)
}

pub struct Simulation {
    pub sys: System,
    rng: rand_pcg::Lcg128Xsl64,
    normal: rand_distr::Normal<f64>,
    a_term: f64,  // dt/visc
    b_term: f64,  // (2.0/(visc*beta)).sqrt()
    pub file: OutputWriter,
    unwrap: Option<Vec<[f64; 3]>>,
    pub beta: f64
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
                old_system.sigmas = match config.potential {
                    Potential::Hertz => {
                        Box::new([0.5*config.rscale, 0.7*config.rscale])
                    },
                    Potential::LJ => {
                        let rscale = config.rscale;
                        Box::new([0.618*rscale, rscale, 1.176*rscale, 0.5, 1.0, 0.5])
                    }
                };

                old_system.vscale = config.vscale;
                old_system.potential = config.potential;
                old_system
            },
            None => {
                let num = config.num;
                let sigmas: Box<[f64]> = match config.potential {
                    Potential::Hertz => {
                        Box::new([0.5*config.rscale, 0.7*config.rscale])
                    },
                    Potential::LJ => {
                        let rscale = config.rscale;
                        Box::new([0.618*rscale, rscale, 1.176*rscale, 0.5, 1.0, 0.5])
                    }
                };
                let dim = config.dim;

                let mut b: [f64; 3] = [1., 1., 1.];
                let mut bh: [f64; 3] = [0.5, 0.5, 0.5];


                let mut types = Vec::<usize>::with_capacity(num);
                for i in 0..num {
                    if i < config.numa {
                        types.push(0);
                    }
                    else {
                        types.push(1);
                    }
                }

                // compute l from volume respecting dimension of the box
                l = if let Some(phi) = config.phi {
                    calc_l_from_phi(config, &types, &sigmas, phi)
                }
                else {
                    config.len
                };
                
                let l2 = l/2.0;
                for i in 0..dim {
                    b[i] = l;
                    bh[i] = l2;
                }

                let mut x = Vec::<[f64; 3]>::with_capacity(num);

                if dim == 3 {
                    for _ in 0..num {
                        x.push([rng.gen::<f64>()*l - l2, 
                            rng.gen::<f64>()*l - l2, 
                            rng.gen::<f64>()*l - l2]);
                    }
                }
                else if dim == 2 {
                    for _ in 0..(config.num) {
                        x.push([(rng.gen::<f64>()*l - l2), 
                        (rng.gen::<f64>()*l - l2), 0.0]);
                    }
                }
                else {
                    panic!("Incorrect dimension! Must be 2 or 3!");
                }
                System{x, types, b, bh, sigmas, dim, vscale: config.vscale, potential: config.potential, numa: config.numa}
            }
        };

        let vol = calc_vol(l, config.dim);

        let phi = sys.calc_phi(vol);

        let path = match config.mode {
            ProgramMode::Standard => {
                format!("{}/traj_n-{}_na-{}_l-{}_t-{}_time-{}_dt-{:e}_visc-{}_seed-{}_phi-{:.4}_pot-{}_rs-{}_vs-{}.xyz",
                    config.dir, sys.x.len(), sys.numa, l, config.temp, config.time, dt, visc, seed, phi, config.potential.to_str(), config.rscale, sys.vscale)
            },
            ProgramMode::Variant(_, _) => {
                format!("{}/variants_n-{}_na-{}_l-{}_t-{}_time-{}_dt-{:e}_visc-{}_seed-{}_phi-{:.4}_pot-{}_rs-{}_vs-{}.h5",
                    config.dir, sys.x.len(), sys.numa, l, config.temp, config.time, dt, visc, seed, phi, config.potential.to_str(), config.rscale, sys.vscale)
            },
            ProgramMode::GenVariant(_, _) => {
                format!("{}/genvariant_n-{}_na-{}_l-{}_t-{}_time-{}_dt-{:e}_visc-{}_seed-{}_phi-{:.4}_pot-{}_rs-{}_vs-{}.h5",
                    config.dir, sys.x.len(), sys.numa, l, config.temp, config.time, dt, visc, seed, phi, config.potential.to_str(), config.rscale, sys.vscale)
            },
            ProgramMode::Equilibrate(_, _) => {
                format!("{}/equil_n-{}_na-{}_l-{}_dt-{:e}_visc-{}_seed-{}_phi-{:.4}_pot-{}_rs-{}_vs-{}.json",
                    config.dir, sys.x.len(), sys.numa, l, dt, visc, seed, phi, config.potential.to_str(), config.rscale, sys.vscale)
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
                    .truncate(true)
                    .create(true)
                    .open(path)
                    .unwrap();
                OutputWriter::XYZBuffer(BufWriter::new(file))
            },
            ProgramMode::Variant(_, _) => 
                OutputWriter::HDF5File(hdf5::File::create(path).unwrap()),
            ProgramMode::GenVariant(_, _) => 
                OutputWriter::HDF5File(hdf5::File::create(path).unwrap()),
            ProgramMode::Equilibrate(_, _) => {
                let file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
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

        let sim = Simulation{sys, rng, normal, a_term: dt/visc, b_term: (2.0/(visc*beta)).sqrt(), file, unwrap, beta};
        sim
    }

    fn force(&self, norm: f64, i: usize, j: usize) -> Option<f64> {
        match self.sys.potential {
            Potential::Hertz => {
                let sigma = self.sys.sigmas[self.sys.types[i]] + self.sys.sigmas[self.sys.types[j]];
                if norm > sigma {
                    None
                }
                else {
                    let vscale = self.sys.vscale;
                    let mag = (vscale/sigma)*(1.0-norm/sigma).powf(1.5);
                    Some(mag)
                }
            }
            Potential::LJ => {
                // sys.sigmas has data for both the sigmas, and the relative potential strengths
                let pair =  self.sys.types[i] + self.sys.types[j];
                let sigma = self.sys.sigmas[pair];
                if norm > sigma*2.5 {
                    None
                }
                else {
                    let epsilon = self.sys.sigmas[pair + 3];
                    let vscale = self.sys.vscale;
                    let norm_inv = 1.0/norm;
                    let x = sigma*norm_inv;
                    let x2 = x*x;
                    let x4 = x2*x2;
                    let x6 = x2*x4;
                    let mag = vscale*epsilon*(12.0*x6*x6*norm_inv - 6.0*x6*norm_inv);
                    Some(mag)
                }
            }
        }
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
                dr = self.pbc_vdr_vec(i, j);
                unsafe {
                    norm = (dr.get_unchecked(0)*dr.get_unchecked(0)
                        + dr.get_unchecked(1)*dr.get_unchecked(1)
                        + dr.get_unchecked(2)*dr.get_unchecked(2)
                    ).sqrt();
                }
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

    pub fn f_system(&mut self) -> Vec<[f64; 3]> {
        let num = self.sys.x.len();
        let mut comp: f64;
        let mut f_hertz_all = Vec::<[f64; 3]>::with_capacity(num);
        for _ in 0..num {
            f_hertz_all.push([0.0, 0.0, 0.0]);
        }
        let mut dr: [f64; 3];
        let mut norm: f64;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec(i, j);
                unsafe {
                    norm = (dr.get_unchecked(0)*dr.get_unchecked(0)
                        + dr.get_unchecked(1)*dr.get_unchecked(1)
                        + dr.get_unchecked(2)*dr.get_unchecked(2)
                    ).sqrt();
                }
                if let Some(mag) = self.force(norm, i, j) {
                    for k in 0..(self.sys.dim) {
                        comp = mag*dr[k]/norm;
                        f_hertz_all[i][k] += comp;
                        f_hertz_all[j][k] -= comp;
                    }
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
                dr = self.pbc_vdr_vec(i, j);
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
    pub fn f_system_hertz_rescale(&mut self, rscale: f64, vscale: f64) -> Vec<[f64; 3]> {
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
                dr = self.pbc_vdr_vec(i, j);
                unsafe {
                    norm = (dr.get_unchecked(0)*dr.get_unchecked(0)
                        + dr.get_unchecked(1)*dr.get_unchecked(1)
                        + dr.get_unchecked(2)*dr.get_unchecked(2)
                    ).sqrt();
                }
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

    pub fn integration_factor_gen(
        &self, 
        force: &Vec<[f64; 3]>, 
        w: &Vec<f64>) -> [f64; 2] {
    let mut factors = [0.0; 2];
    let mut index = 0;
    for i in 0..self.sys.x.len() {
        for j in 0..self.sys.dim {
            factors[0] += 0.25*self.a_term*force[i][j]*force[i][j];
            factors[1] += 0.5*self.b_term*w[index]*force[i][j];
            index += 1;
        }
    }
    factors
}

    fn pbc_vdr_vec(&self, i: usize, j: usize) -> [f64; 3] {
        let mut mdr: [f64; 3] = [0.0, 0.0, 0.0];
        let mut dr: [f64; 3] = [0.0, 0.0, 0.0];
        let x1 = self.sys.x[i];
        let x2 = self.sys.x[j];
    
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
        let forces = self.f_system();
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
        let forces = self.f_system();
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

    fn store_scalar_in_group<T: hdf5::H5Type>(group: &hdf5::Group, data: T, name: &str) {
        let data_arr = arr0(data);
        group.new_dataset::<T>()
            .create(name, data_arr.shape()).unwrap()
            .write(&data_arr).unwrap();
    }

    pub fn dump_hdf5_meta_gen(&mut self, config: &Config, init_x: &Vec<[f64; 3]>) {
        let file = match &self.file {
            OutputWriter::HDF5File(file) => file,
            _ => panic!()
        };
        let group = file.create_group("meta").unwrap();

        Self::store_scalar_in_group(&group, config.temp, "temp");
        Self::store_scalar_in_group(&group, self.beta, "beta");
        Self::store_scalar_in_group(&group, config.time, "time");
        Self::store_scalar_in_group(&group, config.dt, "dt");
        Self::store_scalar_in_group(&group, config.dim, "dim");
        Self::store_scalar_in_group(&group, config.len, "len");
        Self::store_scalar_in_group(&group, config.visc, "visc");
        Self::store_scalar_in_group(&group, config.seed, "seed");
        Self::store_scalar_in_group(&group, self.sys.potential.id(), "pot_id");
        Self::store_scalar_in_group(&group, config.rscale, "rscale");
        Self::store_scalar_in_group(&group, config.vscale, "vscale");

        let x_arr = arr2(&(init_x.to_vec())[..]);
        let init_x_dataset = group.new_dataset::<f64>().create("init_x", x_arr.shape()).unwrap();
        init_x_dataset.write(&x_arr).unwrap();
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

    pub fn create_hdf5_dataset_collection(
        &mut self,
        group_name: &str,
        time_shape: &[usize],
        int_shape: &[usize],
        pos_shape: &[usize]
    ) -> HDF5VarDatasetCol {
        let file = match &self.file {
            OutputWriter::HDF5File(file) => file,
            _ => panic!()
        };
        let group = file.create_group(group_name).unwrap();
        let time = group.new_dataset::<f64>().create("time", time_shape).unwrap();
        let integration = group.new_dataset::<f64>().create("Ib", int_shape).unwrap();
        let position = group.new_dataset::<f64>().create("pos", pos_shape).unwrap();
        HDF5VarDatasetCol{time, position, integration}
    }

    pub fn create_gen_hdf5_dataset_collection(
        &mut self,
        group_name: &str,
        data_names: &[&str],
        data_shapes: &[Vec<usize>]
    ) -> (hdf5::Group, Vec<hdf5::Dataset>) {
        let file = match &self.file {
            OutputWriter::HDF5File(file) => file,
            _ => panic!()
        };
        let group = file.create_group(group_name).unwrap();
        let mut datasets = Vec::with_capacity(data_names.len());
        for (name, shape) in data_names.iter().zip(data_shapes) {
            datasets.push(group.new_dataset::<f64>().create(name, shape).unwrap());
        }
        (group, datasets)
    }

    pub fn dump_gen_hdf5_dataset_collection(
        &mut self,
        data_col: &Vec<hdf5::Dataset>,
        time_data: ArrayView1<f64>,
        norm_data: ArrayView2<f64>,
        msd_data: &Option<Array2<f64>>,
        pos_data: &Option<Array4<f64>>,
        q_data: &Option<Array3<f64>>
    ) {
        data_col[0].write(time_data).unwrap();
        data_col[1].write(norm_data).unwrap();
        let mut idx = 2;
        if let Some(data) = msd_data {
            data_col[idx].write(data).unwrap();
            idx += 1;
        }
        if let Some(data) = pos_data {
            data_col[idx].write(data).unwrap();
            idx += 1;
        }
        if let Some(data) = q_data {
            data_col[idx].write(data).unwrap();
        }
    }

    pub fn dump_hdf5_slices_to_dataset(
        &mut self,
        realization: &usize,
        time: ArrayView1<f64>,
        integration_factors: ArrayView2<f64>,
        positions: ArrayView3<f64>,
        data: &HDF5VarDatasetCol
    ) {
        let slice2d = s![*realization,..,..];
        let slice3d = s![*realization,..,..,..];
        if *realization == 0 { data.time.write(time).unwrap(); }
        data.integration.write_slice(integration_factors, slice2d).unwrap();
        data.position.write_slice(positions, slice3d).unwrap();
    }

    pub fn dump_hdf5_large_slices_to_dataset(
        &mut self,
        start: &usize,
        end: &usize,
        time: ArrayView1<f64>,
        integration_factors: ArrayView3<f64>,
        positions: ArrayView4<f64>,
        data: &HDF5VarDatasetCol
    ) {
        let slice3d = s![*start..*end,..,..];
        let slice4d = s![*start..*end,..,..,..];
        if *start == 0 { data.time.write(time).unwrap(); }
        data.integration.write_slice(integration_factors, slice3d).unwrap();
        data.position.write_slice(positions, slice4d).unwrap();
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