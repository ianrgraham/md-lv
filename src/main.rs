use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::fs::OpenOptions;
use std::io::prelude::*;
use clap::{Arg, App};
use std::str::FromStr;

struct Simulation {
    x: Vec<[f64; 3]>,
    b: [f64; 3],
    bh: [f64; 3],
    rcut: f64,
    rng: rand::prelude::ThreadRng,
    normal: rand_distr::Normal<f64>,
    sigma: f64,
    dim: usize,
    a_term: f64,
    b_term: f64,
}

struct Config {
    num: usize,
    vol: f64,
    temp: f64,
    step_max: usize,
}

#[derive(PartialEq)]
enum WriteMode {
    Append,
    New,
}

fn main() {

    // parse command line options
    let config = Config::new();

    // format the output data depending upon inputs
    let output_suffix = config.format_file_suffix();
    
    // initialize simulation box
    let mut sim = Simulation::new_from_config(&config);

    let mut write_mode = WriteMode::New;

    for step in 0..(config.step_max) {

        // run MC step, if move is successful add 1 to acceptance count
        sim.langevin_step();

        // print output and write data to file
        if step % (2500) == 0 {
            println!("{}", step);
            sim.write_xyz(format!("traj_{}.xyz", output_suffix).to_string(), &write_mode);
            if WriteMode::New == write_mode {
                write_mode = WriteMode::Append;
            }
        }
    }
}

#[allow(dead_code)]
impl Simulation {

    // initialize system from Config struct
    fn new_from_config(config: &Config) -> Simulation {
        // TODO some of these variables need to be moved to the configurations side, but this should work for now
        let rcut: f64 = 1.0;
        let dim = 2;
        let beta = 1./config.temp;
        let dt = 0.01f64;
        let visc = 1.0f64;
        let normal = Normal::new(0.0f64, dt.sqrt()).unwrap();

        let mut b: [f64; 3] = [0., 0., 0.];
        let mut bh: [f64; 3] = [0., 0., 0.];
        let l = config.vol.powf(1./3.);
        let l2 = l/2.0;
        for i in 0..3 {
            b[i] = l;
            bh[i] = 0.5*l;
        }

        let mut x = Vec::<[f64; 3]>::with_capacity(config.num);

        let mut rng = rand::thread_rng();
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

        let sim = Simulation{x: x, b: b, bh: bh, rcut: rcut, rng: rng, normal: normal, sigma: rcut, dim: dim, a_term: dt/visc, b_term: (2.0/(visc*beta)).sqrt()};
        sim
    }

    fn f_single_hertz(&self, idx: &usize) -> [f64; 3] {
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
            if norm > self.rcut {
                continue;
            }
            mag = (1.0/self.sigma)*(1.0-norm/self.sigma).powf(1.5);
            for k in 0..(self.dim) {
                f_hertz[k] += mag*dr[k]/norm;
            }
        }
        f_hertz
    }

    fn f_system_hertz(&mut self) -> Vec<[f64; 3]> {
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
                if norm > self.rcut {
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
    fn rand_index(&mut self) -> usize {
        let i = self.rng.gen_range(0..(self.x.len()));
        i
    }

    fn langevin_step(&mut self) {
        // calculate forces
        let forces = self.f_system_hertz();
        let dim = self.dim;
        let rng = rand::thread_rng();

        // sample normal distribution 
        let w: Vec<f64> = self.normal.sample_iter(rng).take(self.dim*self.x.len()).collect();
        
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
    fn write_xyz(&self, file: String, mode: &WriteMode) -> () {

        let mut file_handle = match mode {
            WriteMode::Append => OpenOptions::new()
            .write(true)
            .append(true)
            .open(file)
            .unwrap(),
            WriteMode::New => OpenOptions::new()
            .write(true)
            .create(true)
            .open(file)
            .unwrap(),
        };
    
        writeln!(file_handle, "{}\n", self.x.len()).expect("FILE IO ERROR!");
    
        for i in 0..(self.x.len()) {
            writeln!(file_handle, "H {} {} {}", self.x[i][0], self.x[i][1], self.x[i][2]).expect("FILE IO ERROR!");
        }
    }

}

impl Config {

    // initialize configuration from command line arguments
    fn new() -> Config {
        let matches = App::new("Langevin dynamics simulation")
            .version("0.2.0")
            .author("Ian Graham <irgraham1@gmail.com>")
            .about("Runs a simulation of a collection of Hertzian particles in the NVT ensemble. Applies overdamped langevin dynamics to update the system")
            .arg(Arg::with_name("NUM")
                .short("n")
                .long("num")
                .help("Number of particles in the box")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("VOL")
                .short("v")
                .long("vol")
                .help("Volume of the box (assuming 3 dimensions)")
                .takes_value(true)
                .default_value("17.0"))
            .arg(Arg::with_name("TEMP")
                .short("t")
                .long("temp")
                .help("Temperature of the system")
                .takes_value(true)
                .default_value("1.5"))
            .arg(Arg::with_name("DT")
                .long("dt")
                .help("Size of the system timestep")
                .takes_value(true)
                .default_value("1e-2"))
            .arg(Arg::with_name("VISC")
                .long("visc")
                .help("Viscous drag coefficient on the particles of the system")
                .takes_value(true)
                .default_value("1.5"))
            .arg(Arg::with_name("STEP")
                .short("s")
                .long("steps")
                .help("Maximum number of simulation steps")
                .takes_value(true)
                .default_value("25000000"))
            .arg(Arg::with_name("dim")
                .short("d")
                .long("dim")
                .help("Dimensions of the simulation box (2 or 3)")
                .takes_value(true)
                .default_value("2"))
            .get_matches();

        let num: usize;
        if let Some(in_num) = matches.value_of("NUM") {
            num = FromStr::from_str(in_num).unwrap();
        }
        else {
            println!("Falling back to secondary default");
            num = 200;
        }

        let vol: f64;
        if let Some(in_vol) = matches.value_of("VOL") {
            vol = FromStr::from_str(in_vol).unwrap();
        }
        else {
            vol = 500.0;
        }

        let temp: f64;
        if let Some(in_temp) = matches.value_of("TEMP") {
            temp = FromStr::from_str(in_temp).unwrap();
        }
        else {
            temp = 1.5;
        }

        let step_max: usize;
        if let Some(in_steps) = matches.value_of("STEP") {
            step_max = FromStr::from_str(in_steps).unwrap();
        }
        else {
            step_max = 25000000;
        }

        Config{num: num, vol: vol, temp: temp, step_max: step_max}
    }

    // format output file suffix with configuration data
    fn format_file_suffix(&self) -> String {
        format!("n{}_v{}_t{}_s{}", self.num, self.vol, self.temp, self.step_max)
    }
}

#[allow(dead_code)]
fn hertz(r: f64, sigma: f64) -> (f64, f64) {
    let v = 0.4*(1.0-r/sigma).powf(2.5);
    let f = (1.0/sigma)*(1.0-r/sigma).powf(1.5);
    (v,f)
}