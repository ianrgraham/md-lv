use clap::{Arg, App};
use std::str::FromStr;

pub struct Config {
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

impl Config {

    // initialize configuration from command line arguments
    pub fn new() -> Config {
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
                .default_value("100_000"))
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
                .default_value("100_000"))
            .arg(Arg::with_name("SEED")
                .long("seed")
                .help("Random seed to initialize the system state")
                .takes_value(true)
                .default_value("0"))
            .get_matches();

        let num = Config::conv_match::<usize>(&matches, "NUM");
        let vol = Config::conv_match::<f64>(&matches, "VOL");
        let temp = Config::conv_match::<f64>(&matches, "TEMP");
        let step_max = Config::conv_match::<usize>(&matches, "STEP");
        let dt = Config::conv_match::<f64>(&matches, "DT");
        let visc = Config::conv_match::<f64>(&matches, "VISC");
        let dim = Config::conv_match::<usize>(&matches, "DIM");
        let write_step = Config::conv_match::<usize>(&matches, "OUT");
        let stdout_step = Config::conv_match::<usize>(&matches, "IO");
        let seed = Config::conv_match::<u64>(&matches, "SEED");

        Config{num: num, vol: vol, temp: temp, step_max: step_max, dt: dt, visc: visc, 
                dim: dim, write_step: write_step, stdout_step: stdout_step, seed: seed}
    }

    // format output file suffix with configuration data
    pub fn format_file_suffix(&self) -> String {
        format!("n{}_v{}_t{}_step{}_dt{}_visc{}_seed{}", 
                self.num, self.vol, self.temp, self.step_max, 
                self.dt, self.visc, self.seed)
    }

    // convert matches to corresponding generic types
    fn conv_match<T>(matches: &clap::ArgMatches, tag: &str) -> T
        where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
        FromStr::from_str(matches.value_of(tag).unwrap()).unwrap()
    }
}