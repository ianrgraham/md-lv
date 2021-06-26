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

impl Config for StdConfig {
    fn file_suffix(&self) -> String {
        format!("n{}_v{}_t{}_step{}_dt{}_visc{}_seed{}_out{}", 
                self.num, self.vol, self.temp, self.step_max, 
                self.dt, self.visc, self.seed, self.write_step)
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
            .about("Runs a simulation of a collection of Hertzian particles in the NVT ensemble. Applies overdamped langevin dynamics to update the system. Tracks potential bias of the system assuming compression of the box \
                    (dialation of the particle radii in practice).")
            .arg(Arg::with_name("NUM")
                .short("n")
                .long("num")
                .help("Number of particles in the box")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("VOL")
                .short("v")
                .long("vol")
                .help("Initial volume (area) of the box")
                .takes_value(true)
                .default_value("8.0"))
            .arg(Arg::with_name("FVOL")
                .long("fvol")
                .help("Final virtual volume (area) of the replica box")
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
                .help("Dimensions of the simulation boxe (2 or 3)")
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
        format!("replica_n{}_v{}_vf{}_t{}_step{}_dt{}_visc{}_seed{}_out{}", 
                self.num, self.vol, self.fvol2, self.temp, self.step_max, 
                self.dt, self.visc, self.seed, self.write_step)
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
                .help("Final virutal volume (area) of the box")
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
        format!("compress_n{}_v{}_vf{}_t{}_step{}_dt{}_visc{}_seed{}_out{}", 
                self.num, self.vol, self.fvol2, self.temp, self.step_max, 
                self.dt, self.visc, self.seed, self.write_step)
    }
}

impl From<CompressConfig> for StdConfig {
    fn from(config: CompressConfig) -> StdConfig {
        StdConfig{num: config.num, vol: config.vol, temp: config.temp, step_max: config.step_max, dt: config.dt, visc: config.visc, 
            dim: config.dim, write_step: config.write_step, stdout_step: config.stdout_step, seed: config.seed}
    }
}

#[derive(Copy, Clone)]
pub struct InitCompressConfig {
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

impl InitCompressConfig {

    // initialize configuration from command line arguments
    pub fn new() -> InitCompressConfig {
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
                .help("Virtual volume (area) of the box")
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

        let num = InitCompressConfig::conv_match::<usize>(&matches, "NUM");
        let vol = InitCompressConfig::conv_match::<f64>(&matches, "VOL");
        let fvol = InitCompressConfig::conv_match::<f64>(&matches, "FVOL");
        let temp = InitCompressConfig::conv_match::<f64>(&matches, "TEMP");
        let step_max = InitCompressConfig::conv_match::<usize>(&matches, "STEP");
        let dt = InitCompressConfig::conv_match::<f64>(&matches, "DT");
        let visc = InitCompressConfig::conv_match::<f64>(&matches, "VISC");
        let dim = InitCompressConfig::conv_match::<usize>(&matches, "DIM");
        let write_step = InitCompressConfig::conv_match::<usize>(&matches, "OUT");
        let stdout_step = InitCompressConfig::conv_match::<usize>(&matches, "IO");
        let seed = InitCompressConfig::conv_match::<u64>(&matches, "SEED");

        InitCompressConfig{num: num, vol: vol, fvol2: fvol, temp: temp, step_max: step_max, dt: dt, visc: visc, 
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

impl Config for InitCompressConfig {
    fn file_suffix(&self) -> String {
        format!("init_compress_n{}_v{}_vf{}_t{}_step{}_dt{}_visc{}_seed{}_out{}", 
                self.num, self.vol, self.fvol2, self.temp, self.step_max, 
                self.dt, self.visc, self.seed, self.write_step)
    }
}

impl From<InitCompressConfig> for StdConfig {
    fn from(config: InitCompressConfig) -> StdConfig {
        StdConfig{num: config.num, vol: config.vol, temp: config.temp, step_max: config.step_max, dt: config.dt, visc: config.visc, 
            dim: config.dim, write_step: config.write_step, stdout_step: config.stdout_step, seed: config.seed}
    }
}

#[derive(Copy, Clone)]
pub struct InitCompressReplicaConfig {
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

impl InitCompressReplicaConfig {

    // initialize configuration from command line arguments
    pub fn new() -> InitCompressReplicaConfig {
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
                .help("Virtual volume (area) of the replica box")
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

        let num = InitCompressReplicaConfig::conv_match::<usize>(&matches, "NUM");
        let vol = InitCompressReplicaConfig::conv_match::<f64>(&matches, "VOL");
        let fvol = InitCompressReplicaConfig::conv_match::<f64>(&matches, "FVOL");
        let temp = InitCompressReplicaConfig::conv_match::<f64>(&matches, "TEMP");
        let step_max = InitCompressReplicaConfig::conv_match::<usize>(&matches, "STEP");
        let dt = InitCompressReplicaConfig::conv_match::<f64>(&matches, "DT");
        let visc = InitCompressReplicaConfig::conv_match::<f64>(&matches, "VISC");
        let dim = InitCompressReplicaConfig::conv_match::<usize>(&matches, "DIM");
        let write_step = InitCompressReplicaConfig::conv_match::<usize>(&matches, "OUT");
        let stdout_step = InitCompressReplicaConfig::conv_match::<usize>(&matches, "IO");
        let seed = InitCompressReplicaConfig::conv_match::<u64>(&matches, "SEED");

        InitCompressReplicaConfig{num: num, vol: vol, fvol2: fvol, temp: temp, step_max: step_max, dt: dt, visc: visc, 
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

impl Config for InitCompressReplicaConfig {
    fn file_suffix(&self) -> String {
        format!("init_compress_replica_n{}_v{}_vf{}_t{}_step{}_dt{}_visc{}_seed{}_out{}", 
                self.num, self.vol, self.fvol2, self.temp, self.step_max, 
                self.dt, self.visc, self.seed, self.write_step)
    }
}

impl From<InitCompressReplicaConfig> for StdConfig {
    fn from(config: InitCompressReplicaConfig) -> StdConfig {
        StdConfig{num: config.num, vol: config.vol, temp: config.temp, step_max: config.step_max, dt: config.dt, visc: config.visc, 
            dim: config.dim, write_step: config.write_step, stdout_step: config.stdout_step, seed: config.seed}
    }
}

pub struct VarBoxSigmaConfig {
    pub num: usize,
    pub len: f64,
    pub temp: f64,
    pub step_max: usize,
    pub dim: usize,
    pub dt: f64,
    pub visc: f64,
    pub write_step: usize,
    pub stdout_step: usize,
    pub seed: u64,
    pub sigma: f64,
    pub fsigma: f64,
    pub escale: f64,
}

impl VarBoxSigmaConfig {

    // initialize configuration from command line arguments
    pub fn new() -> VarBoxSigmaConfig {
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
            .arg(Arg::with_name("LENGTH")
                .short("l")
                .long("length")
                .help("Side length of the simulation box")
                .takes_value(true)
                .default_value("3.0"))
            .arg(Arg::with_name("TEMP")
                .short("t")
                .long("temp")
                .help("Temperature of the system")
                .takes_value(true)
                .default_value("1e-4"))
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
                .default_value("2000"))
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
                .default_value("100"))
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
            .arg(Arg::with_name("SIGMA")
                .long("sigma")
                .help("Maximum distance of interparticle interaction")
                .takes_value(true)
                .default_value("1.5"))
            .arg(Arg::with_name("FSIGMA")
                .long("fsigma")
                .help("Maximum distance of interparticle interaction in the replica")
                .takes_value(true)
                .default_value("1.0"))
            .arg(Arg::with_name("ESCALE")
                .long("energy")
                .help("Energy scale of the system")
                .takes_value(true)
                .default_value("1.0"))
            .arg(Arg::with_name("FESCALE")
                .long("fenergy")
                .help("Energy scale in the replica")
                .takes_value(true)
                .default_value("1.0"))
            .get_matches();

        let num = conv_match::<usize>(&matches, "NUM");
        let len = conv_match::<f64>(&matches, "LENGTH");
        let temp = conv_match::<f64>(&matches, "TEMP");
        let step_max = conv_match::<usize>(&matches, "STEP");
        let dt = conv_match::<f64>(&matches, "DT");
        let visc = conv_match::<f64>(&matches, "VISC");
        let dim = conv_match::<usize>(&matches, "DIM");
        let write_step = conv_match::<usize>(&matches, "OUT");
        let stdout_step = conv_match::<usize>(&matches, "IO");
        let seed = conv_match::<u64>(&matches, "SEED");
        let sigma = conv_match::<f64>(&matches, "SIGMA");
        let fsigma = conv_match::<f64>(&matches, "FSIGMA");
        let escale = conv_match::<f64>(&matches, "ESCALE");

        VarBoxSigmaConfig{num: num, len: len, temp: temp, step_max: step_max, dt: dt, visc: visc, 
                dim: dim, write_step: write_step, stdout_step: stdout_step, seed: seed,
                sigma: sigma, fsigma: fsigma, escale: escale}
    }

    // format output file suffix with configuration data
    pub fn file_suffix(&self) -> String {
        format!("init_compress_replica_n{}_l{}_t{}_si{}_fsi{}_es{}_step{}_dt{}_visc{}_seed{}_out{}", 
                self.num, self.len, self.temp, self.sigma, self.fsigma,
                self.escale, self.step_max, 
                self.dt, self.visc, self.seed, self.write_step)
    }

    // convert matches to corresponding generic types
}

fn conv_match<T>(matches: &clap::ArgMatches, tag: &str) -> T
    where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
    FromStr::from_str(matches.value_of(tag).unwrap()).unwrap()
}