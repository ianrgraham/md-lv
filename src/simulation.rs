use rand::prelude::*;
use rand_pcg::Pcg64;
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use crate::config::Config;


pub struct Simulation {
    pub x: Vec<[f64; 3]>,
    pub types: Vec<usize>,
    pub b: [f64; 3],
    pub bh: [f64; 3],
    pub sigmas: [f64; 2],
    pub rng: rand_pcg::Lcg128Xsl64,
    pub normal: rand_distr::Normal<f64>,
    pub dim: usize,
    pub a_term: f64,  // dt/visc
    pub b_term: f64,  // (2.0/(visc*beta)).sqrt()
    pub file: std::io::BufWriter<std::fs::File>,
    pub config: Config
}


impl Simulation {

    // initialize system from Config struct
    pub fn new_from_config(config: Config) -> Simulation {

        let seed = config.seed;
        let num = config.num;

        let sigmas: [f64; 2] = [0.5, 0.7];
        let beta = 1./config.temp;

        let dt = config.dt;
        let visc = config.visc;
        let dim = config.dim;
        
        let normal = Normal::new(0.0f64, dt.sqrt()).unwrap();

        let mut b: [f64; 3] = [1., 1., 1.];
        let mut bh: [f64; 3] = [0.5, 0.5, 0.5];

        // compute l from volume respecting dimension of the box
        let l = config.len;
        let l2 = l/2.0;
        for i in 0..dim {
            b[i] = l;
            bh[i] = l2;
        }

        let mut x = Vec::<[f64; 3]>::with_capacity(num);
        let mut types = Vec::<usize>::with_capacity(num);

        let mut rng = Pcg64::seed_from_u64(seed);

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
                x.push([(rng.gen::<f64>()*l - l2)*0.1, 
                (rng.gen::<f64>()*l - l2)*0.1, 0.0]);
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

        let coeff: f64 = match dim {
            2 => std::f64::consts::PI,
            3 => 4.0/3.0*std::f64::consts::PI,
            _ => panic!("Dimensions other than 2 and 3 should have already been ruled out!")
        };

        let vol = l.powi(dim as i32);
        let part_vol = types.iter().fold(0.0, |acc, idx| acc + coeff*sigmas[*idx].powi(dim as i32));
        let phi = part_vol/vol;

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(format!("{}/traj_{}_phi{:.3}.xyz", config.dir, config.file_suffix(), phi))
            .unwrap();

        let file = BufWriter::new(file);

        let sim = Simulation{x: x, types: types, b: b, bh: bh, rng: rng, normal: normal, sigmas: sigmas, 
                dim: dim, a_term: dt/visc, b_term: (2.0/(visc*beta)).sqrt(), file: file, config: config};
        sim
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
        let mut sigma: f64;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec(&i, &j);
                norm = (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]).sqrt();
                sigma = self.sigmas[self.types[i]] + self.sigmas[self.types[j]];
                if norm > sigma {
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

    // calculate forces with different sigma
    pub fn f_system_hertz_scale_sigma(&mut self, scale: &f64) -> Vec<[f64; 3]> {
        let num = self.x.len();
        let mut comp: f64;
        let mut f_hertz_all = Vec::<[f64; 3]>::with_capacity(num);
        for _ in 0..num {
            f_hertz_all.push([0.0, 0.0, 0.0]);
        }
        let mut dr: [f64; 3];
        let mut norm: f64;
        let mut mag: f64;
        let mut sigma: f64;
        for i in 0..(num-1) {
            for j in (i+1)..num {
                dr = self.pbc_vdr_vec(&i, &j);
                norm = (dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]).sqrt();
                sigma = (self.sigmas[self.types[i]] + self.sigmas[self.types[j]])*scale;
                if norm > sigma {
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

    pub fn integration_factor(
            &self, 
            force_bias: &Vec<[f64; 3]>, 
            w: &Vec<f64>) -> f64 {
        let mut factor = 0.0f64;
        let mut index = 0;
        for i in 0..self.x.len() {
            for j in 0..self.dim {
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
        let w: Vec<f64> = self.normal
            .sample_iter(&mut self.rng)
            .take(self.dim*self.x.len())
            .collect();
        
        // apply Euler–Maruyama method to update postions
        let mut index = 0;
        for i in 0..(self.x.len()) {
            for k in 0..dim {
                self.x[i][k] += 
                    self.a_term*forces[i][k] + self.b_term*w[index];
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

    pub fn langevin_step_with_forces_w(
            &mut self, 
            forces: &Vec<[f64; 3]>, 
            w: &Vec<f64>) {

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
    
        for (x, typeid) in self.x.iter().zip(&self.types) {
            writeln!(
                self.file, 
                "{} {} {} {} {}", 
                if *typeid == 0 { "A" } else { "B" },
                x[0], 
                x[1], 
                x[2],
                self.sigmas[*typeid])
                .expect("FILE IO ERROR!");
        }
    }

    // return a random vector with the the dimensions of configuration space
    pub fn rand_force_vector(&mut self) -> Vec<f64> {
        let w: Vec<f64> = self.normal.sample_iter(&mut self.rng)
            .take(self.dim*self.x.len())
            .collect();
        w
    }
}