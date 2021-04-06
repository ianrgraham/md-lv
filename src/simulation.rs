use rand::prelude::*;
use rand_pcg::Pcg64;
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use crate::config::Config;
use crate::config::StdConfig;


pub struct Simulation {
    pub x: Vec<[f64; 3]>,
    pub b: [f64; 3],
    pub bh: [f64; 3],
    pub sigma: f64,
    pub rng: rand_pcg::Lcg128Xsl64,
    pub normal: rand_distr::Normal<f64>,
    pub dim: usize,
    pub a_term: f64,
    pub b_term: f64,
    pub file: std::io::BufWriter<std::fs::File>,
}

impl Simulation {

    // initialize system from Config struct
    pub fn new_from_config<T: Config>(gen_config: T) -> Simulation {

        let config: StdConfig = gen_config.into();

        let seed = config.seed;

        let sigma: f64 = 1.0;
        let beta = 1./config.temp;

        let dt = config.dt;
        let visc = config.visc;
        let dim = config.dim;
        
        let normal = Normal::new(0.0f64, dt.sqrt()).unwrap();

        let mut b: [f64; 3] = [1., 1., 1.];
        let mut bh: [f64; 3] = [0.5, 0.5, 0.5];

        // compute l from volume respecting dimension of the box
        let l = config.vol.powf(1./(dim as f64));
        let l2 = l/2.0;
        for i in 0..dim {
            b[i] = l;
            bh[i] = l2;
        }

        let mut x = Vec::<[f64; 3]>::with_capacity(config.num);

        let mut rng = Pcg64::seed_from_u64(seed);

        if dim == 3 {
            for _ in 0..(config.num) {
                x.push([rng.gen::<f64>()*l - l2, rng.gen::<f64>()*l - l2, rng.gen::<f64>()*l - l2])
            }
        }
        else if dim == 2 {
            for _ in 0..(config.num) {
                x.push([rng.gen::<f64>()*l - l2, rng.gen::<f64>()*l - l2, 0.0])
            }
        }
        else {
            panic!("Incorrect dimension! Must be 2 or 3!");
        }

        let file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(format!("traj_{}.xyz", gen_config.file_suffix()))
        .unwrap();

        let file = BufWriter::new(file);

        let sim = Simulation{x: x, b: b, bh: bh, rng: rng, normal: normal, sigma: sigma, 
                dim: dim, a_term: dt/visc, b_term: (2.0/(visc*beta)).sqrt(), file: file};
        sim
    }


    pub fn length_from_vol(&self, vol: &f64) -> f64 {
        vol.powf(1./(self.dim as f64))
    }

    pub fn new_from_config_and_rand<T: Config>(gen_config: T, rng: &mut rand_pcg::Lcg128Xsl64) -> Simulation {

        let config = gen_config.into();

        let seed = config.seed;

        let sigma: f64 = 1.0;
        let beta = 1./config.temp;

        let dt = config.dt;
        let visc = config.visc;
        let dim = config.dim;
        
        let normal = Normal::new(0.0f64, dt.sqrt()).unwrap();

        let mut b: [f64; 3] = [1., 1., 1.];
        let mut bh: [f64; 3] = [0.5, 0.5, 0.5];

        // compute l from volume respecting dimension of the box
        let l = config.vol.powf(1./(dim as f64));
        let l2 = l/2.0;
        for i in 0..dim {
            b[i] = l;
            bh[i] = 0.5*l;
        }

        let mut x = Vec::<[f64; 3]>::with_capacity(config.num);

        let mut _useless_rng = Pcg64::seed_from_u64(seed);

        if dim == 3 {
            for _ in 0..(config.num) {
                x.push([rng.gen::<f64>()*l - l2, rng.gen::<f64>()*l - l2, rng.gen::<f64>()*l - l2])
            }
        }
        else if dim == 2 {
            for _ in 0..(config.num) {
                x.push([rng.gen::<f64>()*l - l2, rng.gen::<f64>()*l - l2, 0.0])
            }
        }
        else {
            panic!("Incorrect dimension! Must be 2 or 3!");
        }

        let file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(format!("traj_{}.xyz", gen_config.file_suffix()))
        .unwrap();

        let file = BufWriter::new(file);

        let sim = Simulation{x: x, b: b, bh: bh, rng: _useless_rng, normal: normal, sigma: sigma, 
                dim: dim, a_term: dt/visc, b_term: (2.0/(visc*beta)).sqrt(), file: file};
        sim
    }

    #[allow(dead_code)]
    pub fn f_single_hertz(&self, idx: &usize) -> [f64; 3] {
        let i = *idx;
        let mut f_hertz: [f64; 3] = [0.0, 0.0, 0.0];
        let mut dr: [f64; 3];
        let mut norm: f64;
        let mut mag:f64;
        let num = self.x.len();
        for j in 0..num {
            if i==j {
                continue;
            }
            dr = self.pbc_vdr_vec(&i, &j);
            norm = (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]).sqrt();
            if norm > self.sigma {
                continue;
            }
            mag = (1.0/self.sigma)*(1.0-norm/self.sigma).powf(1.5);
            for k in 0..(self.dim) {
                f_hertz[k] += mag*dr[k]/norm;
            }
        }
        f_hertz
    }

    pub fn f_system_hertz(&mut self) -> Vec<[f64; 3]> {
        let num = self.x.len();
        let mut comp: f64;
        let mut f_hertz_all = Vec::<[f64; 3]>::with_capacity(num);
        for _ in 0..num {
            f_hertz_all.push([0.0, 0.0, 0.0]);
        }
        let mut dr: [f64; 3];
        let mut norm: f64;
        let mut mag: f64;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec(&i, &j);
                norm = (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]).sqrt();
                if norm > self.sigma {
                    continue;
                }
                mag = (1.0/self.sigma)*(1.0-norm/self.sigma).powf(1.5);
                for k in 0..(self.dim) {
                    comp = mag*dr[k]/norm;
                    f_hertz_all[i][k] += comp;
                    f_hertz_all[j][k] -= comp;
                }
            }

        }
        f_hertz_all
    }

    // calculate forces with different sigma
    pub fn f_system_hertz_sigma(&mut self, sigma: &f64) -> Vec<[f64; 3]> {
        let num = self.x.len();
        let mut comp: f64;
        let mut f_hertz_all = Vec::<[f64; 3]>::with_capacity(num);
        for _ in 0..num {
            f_hertz_all.push([0.0, 0.0, 0.0]);
        }
        let mut dr: [f64; 3];
        let mut norm: f64;
        let mut mag: f64;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec(&i, &j);
                norm = (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]).sqrt();
                if norm > self.sigma {
                    continue;
                }
                mag = (1.0/sigma)*(1.0-norm/sigma).powf(1.5);
                for k in 0..(self.dim) {
                    comp = mag*dr[k]/norm;
                    f_hertz_all[i][k] += comp;
                    f_hertz_all[j][k] -= comp;
                }
            }

        }
        f_hertz_all
    }

    pub fn f_system_hertz_box(&mut self, b: &[f64; 3]) -> Vec<[f64; 3]> {
        let num = self.x.len();
        let mut comp: f64;
        let mut f_hertz_all = Vec::<[f64; 3]>::with_capacity(num);
        for _ in 0..num {
            f_hertz_all.push([0.0, 0.0, 0.0]);
        }
        let mut dr: [f64; 3];
        let mut norm: f64;
        let mut mag: f64;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec_box(&i, &j, &b);
                norm = (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]).sqrt();
                if norm > self.sigma {
                    continue;
                }
                mag = (1.0/self.sigma)*(1.0-norm/self.sigma).powf(1.5);
                for k in 0..(self.dim) {
                    comp = mag*dr[k]/norm;
                    f_hertz_all[i][k] += comp;
                    f_hertz_all[j][k] -= comp;
                }
            }

        }
        f_hertz_all
    }

    pub fn integration_factor(&self, force_bias: &Vec<[f64; 3]>, w: &Vec<f64>) -> f64 {
        let mut factor = 0.0f64;
        let mut index = 0;
        for i in 0..self.x.len() {
            for j in 0..self.dim {
                factor += force_bias[i][j]*(0.25*self.a_term*force_bias[i][j] + 0.5*self.b_term*w[index]);
                index += 1;
            }
        }
        factor
    }

    // #[allow(dead_code)]
    // pub fn rescale_box() {
        
    // }

    fn pbc_vdr_vec(&self, i: &usize, j: &usize) -> [f64; 3] {
        let mut mdr: [f64; 3] = [0.0, 0.0, 0.0];
        let mut dr: [f64; 3] = [0.0, 0.0, 0.0];
        let x1 = self.x[*i];
        let x2 = self.x[*j];
    
        for i in 0..(self.dim) {
            dr[i] = x1[i] - x2[i];
    
            if dr[i] >= self.bh[i] {
                dr[i] -= self.b[i];
            }
            else if dr[i] < -self.bh[i] {
                dr[i] += self.b[i];
            }
            mdr[i] += dr[i]
        }
        mdr
    }

    fn pbc_vdr_vec_box(&self, i: &usize, j: &usize, b: &[f64; 3]) -> [f64; 3] {
        let mut mdr: [f64; 3] = [0.0, 0.0, 0.0];
        let mut dr: [f64; 3] = [0.0, 0.0, 0.0];
        let bh: [f64; 3] = [b[0]*0.5, b[1]*0.5, b[2]*0.5];
        let x1 = self.x[*i];
        let x2 = self.x[*j];
    
        for i in 0..(self.dim) {
            dr[i] = x1[i] - x2[i];
    
            if dr[i] >= bh[i] {
                dr[i] -= b[i];
                while dr[i] >= bh[i] {
                    dr[i] -= b[i];
                }
            }
            else if dr[i] < -bh[i] {
                dr[i] += b[i];
                while dr[i] < -bh[i] {
                    dr[i] += b[i];
                }
            }
            mdr[i] += dr[i]
        }
        mdr
    }


    // use internal random number generator to fetch an index
    #[allow(dead_code)]
    fn rand_index(&mut self) -> usize {
        let i = self.rng.gen_range(0..(self.x.len()));
        i
    }

    pub fn langevin_step(&mut self) {
        // calculate forces
        let forces = self.f_system_hertz();
        let dim = self.dim;

        // sample normal distribution 
        let w: Vec<f64> = self.normal.sample_iter(&mut self.rng).take(self.dim*self.x.len()).collect();
        
        // apply Euler–Maruyama method to update postions
        let mut index = 0;
        for i in 0..(self.x.len()) {
            for k in 0..dim {
                self.x[i][k] += self.a_term*forces[i][k] + self.b_term*w[index];
                if self.x[i][k] >= self.bh[k] {
                    self.x[i][k] -= self.b[k]
                }
                else if self.x[i][k] < -self.bh[k] {
                    self.x[i][k] += self.b[k]
                }
                index += 1;
            }
        }
    }

    pub fn langevin_step_with_forces_w(&mut self, forces: &Vec<[f64; 3]>, w: &Vec<f64>) {

        let dim = self.dim;
        
        // apply Euler–Maruyama method to update postions
        let mut index = 0;
        for i in 0..(self.x.len()) {
            for k in 0..dim {
                self.x[i][k] += self.a_term*forces[i][k] + self.b_term*w[index];
                if self.x[i][k] >= self.bh[k] {
                    self.x[i][k] -= self.b[k]
                }
                else if self.x[i][k] < -self.bh[k] {
                    self.x[i][k] += self.b[k]
                }
                index += 1;
            }
        }
    }

    // dump simulation state to xyz file
    pub fn dump_xyz(&mut self) {

        writeln!(self.file, "{}\n", self.x.len()).expect("FILE IO ERROR!");
    
        for i in 0..(self.x.len()) {
            writeln!(self.file, "H {} {} {}", self.x[i][0], self.x[i][1], self.x[i][2]).expect("FILE IO ERROR!");
        }
    }

    // return a random vector with the the dimensions of configuration space
    pub fn rand_force_vector(&mut self) -> Vec<f64> {
        let w: Vec<f64> = self.normal.sample_iter(&mut self.rng).take(self.dim*self.x.len()).collect();
        w
    }
}