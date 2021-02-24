use clap::{Arg, App};
use std::str::FromStr;

pub trait Config: Copy + Into<StdConfig> {
    fn file_suffix(&self) -> String;
}

#[derive(Copy, Clone)]
pub struct StdConfig {
    pub num: usize,
    pub vol: f64,
    pub temp: f64,
    pub step_max: usize,
    pub dim: usize,
    pub dt: f64,
    pub visc: f64,
    pub write_step: usize,
    pub stdout_step: usize,
    pub seed: u64,
}

impl StdConfig {

    // initialize configuration from command line arguments
    pub fn new() -> StdConfig {
        let matches = App::new("Langevin dynamics simulation")
            .version("0.2.1")
            .author("Ian Graham <irgraham1@gmail.com>")
            .about("Runs a simulation of a collection of Hertzian particles in the NVT ensemble. Applies overdamped langevin dynamics to update the system.")
            .arg(Arg::with_name("NUM")
                .short("n")
                .long("num")
                .help("Number of particles in the box")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("VOL")
                .short("v")
                .long("vol")
                .help("Volume (area) of the box")
                .takes_value(true)
                .default_value("6.5"))
            .arg(Arg::with_name("TEMP")
                .short("t")
                .long("temp")
                .help("Temperature of the system")
                .takes_value(true)
                .default_value("0.5"))
            .arg(Arg::with_name("DT")
                .long("dt")
                .help("Size of the system timestep")
                .takes_value(true)
                .default_value("1e-3"))
            .arg(Arg::with_name("VISC")
                .long("visc")
                .help("Viscous drag coefficient on the particles of the system")
                .takes_value(true)
                .default_value("5.0"))
            .arg(Arg::with_name("STEP")
                .short("s")
                .long("steps")
                .help("Maximum number of simulation steps")
                .takes_value(true)
                .default_value("100000"))
            .arg(Arg::with_name("DIM")
                .short("d")
                .long("dim")
                .help("Dimensions of the simulation box (2 or 3)")
                .takes_value(true)
                .default_value("2"))
            .arg(Arg::with_name("OUT")
                .short("o")
                .long("outstep")
                .help("Number of steps between dump to output")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("IO")
                .short("i")
                .long("iostep")
                .help("Number of steps between messages to stdout")
                .takes_value(true)
                .default_value("10000"))
            .arg(Arg::with_name("SEED")
                .long("seed")
                .help("Random seed to initialize the system state")
                .takes_value(true)
                .default_value("0"))
            .get_matches();

        let num = StdConfig::conv_match::<usize>(&matches, "NUM");
        let vol = StdConfig::conv_match::<f64>(&matches, "VOL");
        let temp = StdConfig::conv_match::<f64>(&matches, "TEMP");
        let step_max = StdConfig::conv_match::<usize>(&matches, "STEP");
        let dt = StdConfig::conv_match::<f64>(&matches, "DT");
        let visc = StdConfig::conv_match::<f64>(&matches, "VISC");
        let dim = StdConfig::conv_match::<usize>(&matches, "DIM");
        let write_step = StdConfig::conv_match::<usize>(&matches, "OUT");
        let stdout_step = StdConfig::conv_match::<usize>(&matches, "IO");
        let seed = StdConfig::conv_match::<u64>(&matches, "SEED");

        StdConfig{num: num, vol: vol, temp: temp, step_max: step_max, dt: dt, visc: visc, 
                dim: dim, write_step: write_step, stdout_step: stdout_step, seed: seed}
    }

    // format output file suffix with configuration data
    // pub fn format_file_suffix(&self) -> String {
    //     self.file_suffix()
    // }

    // convert matches to corresponding generic types
    fn conv_match<T>(matches: &clap::ArgMatches, tag: &str) -> T
        where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
        // println!("{}", tag);
        FromStr::from_str(matches.value_of(tag).unwrap()).unwrap()
    }
}

impl Config for StdConfig {
    fn file_suffix(&self) -> String {
        format!("replica_n{}_v{}_t{}_step{}_dt{}_visc{}_seed{}", 
                self.num, self.vol, self.temp, self.step_max, 
                self.dt, self.visc, self.seed)
    }
}

impl From<&StdConfig> for StdConfig {
    fn from(config: &StdConfig) -> StdConfig {
        StdConfig{num: config.num, vol: config.vol, temp: config.temp, step_max: config.step_max, dt: config.dt, visc: config.visc, 
            dim: config.dim, write_step: config.write_step, stdout_step: config.stdout_step, seed: config.seed}
    }
}

#[derive(Copy, Clone)]
pub struct ReplicaConfig {
    pub num: usize,
    pub vol: f64,
    pub fvol2: f64,
    pub temp: f64,
    pub step_max: usize,
    pub dim: usize,
    pub dt: f64,
    pub visc: f64,
    pub write_step: usize,
    pub stdout_step: usize,
    pub seed: u64,
}

impl ReplicaConfig {

    // initialize configuration from command line arguments
    pub fn new() -> ReplicaConfig {
        let matches = App::new("Langevin dynamics simulation of Replicas")
            .version("0.2.1")
            .author("Ian Graham <irgraham1@gmail.com>")
            .about("Runs two parallel simulations of collections of Hertzian particles in the NVT ensemble. Applies overdamped langevin dynamics to update the system, where both systems share the same noise. \
                    The second realization is compressed while the first is held at a fixed box size.")
            .arg(Arg::with_name("NUM")
                .short("n")
                .long("num")
                .help("Number of particles in each box")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("VOL")
                .short("v")
                .long("vol")
                .help("Initial volume (area) of both boxes")
                .takes_value(true)
                .default_value("8.0"))
            .arg(Arg::with_name("FVOL")
                .long("fvol")
                .help("Final volume (area) of the replica box")
                .takes_value(true)
                .default_value("5.0"))
            .arg(Arg::with_name("TEMP")
                .short("t")
                .long("temp")
                .help("Temperature of both systems")
                .takes_value(true)
                .default_value("0.5"))
            .arg(Arg::with_name("DT")
                .long("dt")
                .help("Size of both systems' timestep")
                .takes_value(true)
                .default_value("1e-3"))
            .arg(Arg::with_name("VISC")
                .long("visc")
                .help("Viscous drag coefficient on the particles of each system")
                .takes_value(true)
                .default_value("5.0"))
            .arg(Arg::with_name("STEP")
                .short("s")
                .long("steps")
                .help("Maximum number of simulation steps")
                .takes_value(true)
                .default_value("100000"))
            .arg(Arg::with_name("DIM")
                .short("d")
                .long("dim")
                .help("Dimensions of the simulation boxes (2 or 3)")
                .takes_value(true)
                .default_value("2"))
            .arg(Arg::with_name("OUT")
                .short("o")
                .long("outstep")
                .help("Number of steps between dump to output")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("IO")
                .short("i")
                .long("iostep")
                .help("Number of steps between messages to stdout")
                .takes_value(true)
                .default_value("10000"))
            .arg(Arg::with_name("SEED")
                .long("seed")
                .help("Random seed to initialize the system state")
                .takes_value(true)
                .default_value("0"))
            .get_matches();

        let num = ReplicaConfig::conv_match::<usize>(&matches, "NUM");
        let vol = ReplicaConfig::conv_match::<f64>(&matches, "VOL");
        let fvol = ReplicaConfig::conv_match::<f64>(&matches, "FVOL");
        let temp = ReplicaConfig::conv_match::<f64>(&matches, "TEMP");
        let step_max = ReplicaConfig::conv_match::<usize>(&matches, "STEP");
        let dt = ReplicaConfig::conv_match::<f64>(&matches, "DT");
        let visc = ReplicaConfig::conv_match::<f64>(&matches, "VISC");
        let dim = ReplicaConfig::conv_match::<usize>(&matches, "DIM");
        let write_step = ReplicaConfig::conv_match::<usize>(&matches, "OUT");
        let stdout_step = ReplicaConfig::conv_match::<usize>(&matches, "IO");
        let seed = ReplicaConfig::conv_match::<u64>(&matches, "SEED");

        ReplicaConfig{num: num, vol: vol, fvol2: fvol, temp: temp, step_max: step_max, dt: dt, visc: visc, 
                dim: dim, write_step: write_step, stdout_step: stdout_step, seed: seed}
    }

    // format output file suffix with configuration data
    pub fn format_file_suffix(&self) -> String {
        self.file_suffix()
    }

    // convert matches to corresponding generic types
    fn conv_match<T>(matches: &clap::ArgMatches, tag: &str) -> T
        where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
        // println!("{}", tag);
        FromStr::from_str(matches.value_of(tag).unwrap()).unwrap()
    }
}

impl Config for ReplicaConfig {
    fn file_suffix(&self) -> String {
        format!("replica_n{}_v{}_vf{}_t{}_step{}_dt{}_visc{}_seed{}", 
                self.num, self.vol, self.fvol2, self.temp, self.step_max, 
                self.dt, self.visc, self.seed)
    }
}

impl From<ReplicaConfig> for StdConfig {
    fn from(config: ReplicaConfig) -> StdConfig {
        StdConfig{num: config.num, vol: config.vol, temp: config.temp, step_max: config.step_max, dt: config.dt, visc: config.visc, 
            dim: config.dim, write_step: config.write_step, stdout_step: config.stdout_step, seed: config.seed}
    }
}

#[derive(Copy, Clone)]
pub struct CompressConfig {
    pub num: usize,
    pub vol: f64,
    pub fvol2: f64,
    pub temp: f64,
    pub step_max: usize,
    pub dim: usize,
    pub dt: f64,
    pub visc: f64,
    pub write_step: usize,
    pub stdout_step: usize,
    pub seed: u64,
}

impl CompressConfig {

    // initialize configuration from command line arguments
    pub fn new() -> CompressConfig {
        let matches = App::new("Langevin dynamics simulation under compression")
            .version("0.2.1")
            .author("Ian Graham <irgraham1@gmail.com>")
            .about("Runs a simulation of a collection of Hertzian particles in the NVT ensemble. Applies overdamped langevin dynamics to update the system. The system is compressed to a final pressure at the end of the run")
            .arg(Arg::with_name("NUM")
                .short("n")
                .long("num")
                .help("Number of particles in the box")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("VOL")
                .short("v")
                .long("vol")
                .help("Volume (area) of the box")
                .takes_value(true)
                .default_value("6.5"))
            .arg(Arg::with_name("FVOL")
                .long("fvol")
                .help("Final volume (area) of the box")
                .takes_value(true)
                .default_value("5.0"))
            .arg(Arg::with_name("TEMP")
                .short("t")
                .long("temp")
                .help("Temperature of the system")
                .takes_value(true)
                .default_value("0.5"))
            .arg(Arg::with_name("DT")
                .long("dt")
                .help("Size of the system timestep")
                .takes_value(true)
                .default_value("1e-3"))
            .arg(Arg::with_name("VISC")
                .long("visc")
                .help("Viscous drag coefficient on the particles of the system")
                .takes_value(true)
                .default_value("5.0"))
            .arg(Arg::with_name("STEP")
                .short("s")
                .long("steps")
                .help("Maximum number of simulation steps")
                .takes_value(true)
                .default_value("100000"))
            .arg(Arg::with_name("DIM")
                .short("d")
                .long("dim")
                .help("Dimensions of the simulation box (2 or 3)")
                .takes_value(true)
                .default_value("2"))
            .arg(Arg::with_name("OUT")
                .short("o")
                .long("outstep")
                .help("Number of steps between dump to output")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("IO")
                .short("i")
                .long("iostep")
                .help("Number of steps between messages to stdout")
                .takes_value(true)
                .default_value("10000"))
            .arg(Arg::with_name("SEED")
                .long("seed")
                .help("Random seed to initialize the system state")
                .takes_value(true)
                .default_value("0"))
            .get_matches();

        let num = CompressConfig::conv_match::<usize>(&matches, "NUM");
        let vol = CompressConfig::conv_match::<f64>(&matches, "VOL");
        let fvol = CompressConfig::conv_match::<f64>(&matches, "FVOL");
        let temp = CompressConfig::conv_match::<f64>(&matches, "TEMP");
        let step_max = CompressConfig::conv_match::<usize>(&matches, "STEP");
        let dt = CompressConfig::conv_match::<f64>(&matches, "DT");
        let visc = CompressConfig::conv_match::<f64>(&matches, "VISC");
        let dim = CompressConfig::conv_match::<usize>(&matches, "DIM");
        let write_step = CompressConfig::conv_match::<usize>(&matches, "OUT");
        let stdout_step = CompressConfig::conv_match::<usize>(&matches, "IO");
        let seed = CompressConfig::conv_match::<u64>(&matches, "SEED");

        CompressConfig{num: num, vol: vol, fvol2: fvol, temp: temp, step_max: step_max, dt: dt, visc: visc, 
                dim: dim, write_step: write_step, stdout_step: stdout_step, seed: seed}
    }

    // format output file suffix with configuration data
    pub fn format_file_suffix(&self) -> String {
        self.file_suffix()
    }

    // convert matches to corresponding generic types
    fn conv_match<T>(matches: &clap::ArgMatches, tag: &str) -> T
        where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
        // println!("{}", tag);
        FromStr::from_str(matches.value_of(tag).unwrap()).unwrap()
    }
}

impl Config for CompressConfig {
    fn file_suffix(&self) -> String {
        format!("compress_n{}_v{}_vf{}_t{}_step{}_dt{}_visc{}_seed{}", 
                self.num, self.vol, self.fvol2, self.temp, self.step_max, 
                self.dt, self.visc, self.seed)
    }
}

impl From<CompressConfig> for StdConfig {
    fn from(config: CompressConfig) -> StdConfig {
        StdConfig{num: config.num, vol: config.vol, temp: config.temp, step_max: config.step_max, dt: config.dt, visc: config.visc, 
            dim: config.dim, write_step: config.write_step, stdout_step: config.stdout_step, seed: config.seed}
    }
}