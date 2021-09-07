use clap::{Arg, App, SubCommand};
use std::str::FromStr;
use std::fs::File;
use std::io::BufReader;
use serde::*;


pub enum ProgramMode {
    Standard,
    Variant(VariantConfigs, usize),
    GenVariant(usize),
    Equilibrate(f64, f64)
}

#[derive(hdf5::H5Type, Clone, PartialEq, Serialize, Deserialize, Debug)]
#[repr(C)]
pub struct VariantConfig {
    pub rscale: f64,
    pub vscale: f64
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VariantConfigs {
    pub configs: Vec<VariantConfig>
}

impl VariantConfigs {
    pub fn len(&self) -> usize {
        self.configs.len()
    }
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
    pub stdout_step: Option<usize>,
    pub seed: u64,
    pub dir: String,
    pub dryprint: bool,
    pub rscale: f64,
    pub vscale: f64,
    pub mode: ProgramMode,
    pub init_config: Option<String>,
    pub unwrap: bool
}

// convert matches to corresponding generic types, panic if there is an issue
fn conv_match<T>(matches: &clap::ArgMatches, tag: &str) -> T
    where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
    let value = matches.value_of(tag).unwrap();
    FromStr::from_str(value).expect("Failed to convert &str to type T")
}

// convert matches to corresponding generic types, panic if there is an issue
fn conv_optional_match<T>(matches: &clap::ArgMatches, tag: &str) -> Option<T>
    where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
    let value = match matches.value_of(tag) {
        Some(value) => value,
        None => return None
    };
    Some(FromStr::from_str(value).expect("Failed to convert &str to type T"))
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
                .default_value("3.0"))
            // .arg(Arg::with_name("PHI")
            //     .long("phi")
            //     .takes_value(true)
            //     .help("Specify instead the packing fraction")
            //     .conflicts_with("LEN"))
            .arg(Arg::with_name("TEMP")
                .short("t")
                .long("temp")
                .help("Temperature of the system")
                .takes_value(true)
                .default_value("0.1"))
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
                .default_value("10.0"))
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
                .default_value("1.0"))
            // .arg(Arg::with_name("OUTLOG")
            //     .short("o")
            //     .long("out-time-log")
            //     .help("Time between log-separated output dumps")
            //     .takes_value(true)
            //     .conflicts_with("OUT"))
            .arg(Arg::with_name("STDOUT")
                .short("i")
                .long("stdout-time")
                .help("Time between terminal writes")
                .takes_value(true))
            .arg(Arg::with_name("INIT_CONFIG")
                .long("init-config")
                .help("JSON config file initializing the system state. \
                    Will assert that config is valid")
                .takes_value(true)
                .conflicts_with("NUM")
                .conflicts_with("LEN")
                .conflicts_with("DIM"))
            .arg(Arg::with_name("SEED")
                .long("seed")
                .help("Random seed to initialize the internal random number generator")
                .takes_value(true)
                .default_value("0"))
            .arg(Arg::with_name("DRYPRINT")
                .long("dryprint")
                .help("Print out simulation config without running md"))
            .arg(Arg::with_name("UNWRAP")
                .long("unwrap")
                .help("Output unwrapped particle trajectories"))
            .arg(Arg::with_name("KAHAN")
                .long("kahan")
                .help("Utilize Kahan Summation in the Euler-Mayurama method"))
            .subcommand(SubCommand::with_name("variant")
                .about("Used to run variant trajectories with differing parameters")
                .arg(Arg::with_name("CONFIG")
                    .required(true)
                    .index(1)
                    .help("JSON config file containing variant params"))
                .arg(Arg::with_name("REALIZATIONS")
                    .long("realizations")
                    .takes_value(true)
                    .default_value("1000000")
                    .help("Number of realizations to run")))
            .subcommand(SubCommand::with_name("gen-variant")
                .about("Used to run generic variant method")
                .arg(Arg::with_name("REALIZATIONS")
                    .long("realizations")
                    .takes_value(true)
                    .default_value("1000000")
                    .help("Number of realizations to run")))
            .subcommand(SubCommand::with_name("equil-gd")
                .about("Generate loadable simulation config quenched to its inherent structure")
                .arg(Arg::with_name("MAX_DR")
                    .required(true)
                    .default_value("1e-10"))
                .arg(Arg::with_name("MAX_F")
                    .required(true)
                    .default_value("1e-10")))
            .get_matches();

        let num = conv_match(&matches, "NUM");
        let len = conv_match(&matches, "LEN");
        let temp = conv_match(&matches, "TEMP");
        let time = conv_match::<f64>(&matches, "TIME");
        let dt = conv_match::<f64>(&matches, "DT");
        let visc = conv_match(&matches, "VISC");
        let dim = conv_match(&matches, "DIM");
        let write_time = conv_match::<f64>(&matches, "OUT");
        let stdout_time = conv_optional_match::<f64>(&matches, "STDOUT");
        let seed = conv_match(&matches, "SEED");
        let dir = conv_match(&matches, "DIR");
        let rscale = conv_match(&matches, "RSCALE");
        let vscale = conv_match(&matches, "VSCALE");

        let dryprint = matches.is_present("DRYPRINT");
        let unwrap = matches.is_present("UNWRAP");
        let init_config = matches.value_of("INIT_CONFIG").map(|path| path.to_string());

        let mode: ProgramMode = {
            if let Some(variant_match) = matches.subcommand_matches("variant") {
                let path = variant_match.value_of("CONFIG").unwrap();
                let realizations = conv_match(&variant_match, "REALIZATIONS");
                //let variants: VariantConfigs = confy::load_path(path)
                let reader = BufReader::new(File::open(path).unwrap());
                let variants = serde_json::from_reader(reader)
                    .expect("Failed to open variant config!");
                dbg!(&variants);
                ProgramMode::Variant(variants, realizations)
            }
            else if let Some(equil_match) = matches.subcommand_matches("equil-gd") {
                let max_dr: f64 = conv_match(&equil_match, "MAX_DR");
                let max_f: f64 = conv_match(&equil_match, "MAX_F");
                ProgramMode::Equilibrate(max_dr, max_f)
            }
            else if let Some(gen_variant_match) = matches.subcommand_matches("gen-variant") {
                let realizations = conv_match(&gen_variant_match, "REALIZATIONS");
                ProgramMode::GenVariant(realizations)
            }
            else {
                ProgramMode::Standard
            }
        };

        let step_max = (time/dt).round() as usize;
        let write_step = (write_time/dt).round() as usize;
        let stdout_step = match stdout_time {
            Some(some_stdout_time) => Some((some_stdout_time/dt).round() as usize),
            None => None
        };

        Config{num: num, len: len, temp: temp, time: time, step_max: step_max, dt: dt, visc: visc, 
                dim: dim, write_step: write_step, stdout_step: stdout_step, seed: seed, dir: dir,
                dryprint: dryprint, rscale: rscale, vscale: vscale, mode: mode, init_config: init_config,
                unwrap: unwrap}
    }

    // format output file suffix with configuration data
    pub fn file_suffix(&self) -> String {
        format!("n-{}_l-{}_t-{}_time-{}_dt-{:e}_visc-{}_seed-{}", 
                self.num, self.len, self.temp, self.time, 
                self.dt, self.visc, self.seed)
    }
}