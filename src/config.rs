use clap::{Arg, App, SubCommand};
use std::str::FromStr;
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
struct MyConfig {
    version: u8,
    api_key: String,
}

/// `MyConfig` implements `Default`
impl ::std::default::Default for MyConfig {
    fn default() -> Self { Self { version: 0, api_key: "".into() } }
}


pub struct Config {
    pub num: usize,
    pub len: f64,
    pub temp: f64,
    pub time: f64,
    pub step_max: usize,
    pub dim: usize,
    pub dt: f64,
    pub visc: f64,
    pub write_step: usize,
    pub stdout_step: usize,
    pub seed: u64,
    pub dir: String,
    pub dryprint: bool,
    pub rscale: f64,
    pub vscale: f64
}

// convert matches to corresponding generic types, panic if there is an issue
fn conv_match<T>(matches: &clap::ArgMatches, tag: &str) -> T
    where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
    let value = matches.value_of(tag).unwrap();
    FromStr::from_str(value).expect("Failed to convert &str to type T")
}

impl Config {

    // todo need to add addition config parameters for modifying potential and 

    // initialize configuration from command line arguments
    pub fn new() -> Config {
        let matches = App::new("Langevin dynamics simulation")
            .version("0.3.0")
            .author("Ian Graham <irgraham1@gmail.com>")
            .about("Runs a simulation of a collection of Hertzian particles in the NVT ensemble. \
                Applies overdamped langevin dynamics to update the system.")
            .arg(Arg::with_name("NUM")
                .short("n")
                .long("num")
                .help("Number of particles in the box")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("LEN")
                .short("l")
                .long("len")
                .help("Side length of the simulation box")
                .takes_value(true)
                .default_value("4.0"))
            .arg(Arg::with_name("TEMP")
                .short("t")
                .long("temp")
                .help("Temperature of the system")
                .takes_value(true)
                .default_value("0.63"))
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
            .arg(Arg::with_name("RSCALE")
                .long("rscale")
                .help("Rescaling distance cutoff for the inter-atomic potentials")
                .takes_value(true)
                .default_value("1.0"))
            .arg(Arg::with_name("VSCALE")
                .long("vscale")
                .help("Rescaling coefficient factor for the inter-atomic potentials")
                .takes_value(true)
                .default_value("1.0"))
            .arg(Arg::with_name("TIME")
                .long("time")
                .help("Run time of the simulation")
                .takes_value(true)
                .default_value("50.0"))
            .arg(Arg::with_name("DIM")
                .short("d")
                .long("dim")
                .help("Dimensions of the simulation box (2 or 3)")
                .takes_value(true)
                .default_value("2"))
            .arg(Arg::with_name("DIR")
                .long("dir")
                .help("Output directory of data dumps")
                .takes_value(true)
                .default_value("."))
            .arg(Arg::with_name("OUT")
                .short("o")
                .long("out-time")
                .help("Time between output dumps")
                .takes_value(true)
                .default_value("0.1"))
            .arg(Arg::with_name("STDOUT")
                .short("i")
                .long("stdout-time")
                .help("Time between terminal writes")
                .takes_value(true)
                .default_value("100.0"))
            .arg(Arg::with_name("SEED")
                .long("seed")
                .help("Random seed to initialize the internal random number generator")
                .takes_value(true)
                .default_value("0"))
            .arg(Arg::with_name("OUTFORMAT")
                .long("out-format")
                .help("Format of the simulation output")
                .possible_values(&["xyz", "hdf"])
                .takes_value(true)
                .default_value("xyz"))
            .subcommand(SubCommand::with_name("dryprint")
                .about("Used to print out simulation config without running md"))
            .subcommand(SubCommand::with_name("bias")
                .about("Used to print out simulation config without running md"))
            .get_matches();

        let num = conv_match(&matches, "NUM");
        let len = conv_match(&matches, "LEN");
        let temp = conv_match(&matches, "TEMP");
        let time = conv_match::<f64>(&matches, "TIME");
        let dt = conv_match::<f64>(&matches, "DT");
        let visc = conv_match(&matches, "VISC");
        let dim = conv_match(&matches, "DIM");
        let write_time = conv_match::<f64>(&matches, "OUT");
        let stdout_time = conv_match::<f64>(&matches, "STDOUT");
        let seed = conv_match(&matches, "SEED");
        let dir = conv_match(&matches, "DIR");
        let rscale = conv_match(&matches, "RSCALE");
        let vscale = conv_match(&matches, "VSCALE");
        let dryprint = matches.subcommand_matches("dryprint").is_some();

        let step_max = (time/dt).round() as usize;
        let write_step = (write_time/dt).round() as usize;
        let stdout_step = (stdout_time/dt).round() as usize;

        Config{num: num, len: len, temp: temp, time: time, step_max: step_max, dt: dt, visc: visc, 
                dim: dim, write_step: write_step, stdout_step: stdout_step, seed: seed, dir: dir,
                dryprint: dryprint, rscale: rscale, vscale: vscale}
    }

    // format output file suffix with configuration data
    pub fn file_suffix(&self) -> String {
        format!("n-{}_l-{}_t-{}_time-{}_dt-{:e}_visc-{}_seed-{}", 
                self.num, self.len, self.temp, self.time, 
                self.dt, self.visc, self.seed)
    }
}