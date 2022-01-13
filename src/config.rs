use clap::{Arg, App, SubCommand};
use std::str::FromStr;

use crate::simulation::Potential;

pub enum ProgramMode {
    Standard,
    GenVariant(usize, Option<(Vec<f64>, bool, bool, Option<Vec<f64>>)>),
    Equilibrate(f64, f64, bool)
}

// data structure to ferry parameters from the CLI to the simulation
pub struct Config {
    pub num: usize,
    pub numa: usize,
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
    pub unwrap: bool,
    pub phi: Option<f64>,
    pub potential: Potential,
    pub images: bool
}

// convert matches to corresponding generic types, panic if there is an issue
fn conv_match<T>(matches: &clap::ArgMatches, tag: &str) -> T
    where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
    let value = matches.value_of(tag).unwrap();
    FromStr::from_str(value).expect(&format!("Failed to convert &str to type {}", std::any::type_name::<T>()))
}

// convert matches to corresponding generic types, panic if there is an issue
fn conv_optional_match<T>(matches: &clap::ArgMatches, tag: &str) -> Option<T>
    where T: FromStr, <T as std::str::FromStr>::Err : std::fmt::Debug  {
    let value = match matches.value_of(tag) {
        Some(value) => value,
        None => return None
    };
    Some(FromStr::from_str(value).expect(&format!("Failed to convert &str to type {}", std::any::type_name::<T>())))
}

impl Config {

    // initialize configuration from command line arguments
    pub fn new() -> Config {

        let matches = App::new("Langevin dynamics simulation")
            .version(env!("CARGO_PKG_VERSION"))
            .author("Ian Graham <irgraham1@gmail.com>")
            .about("Runs a simulation of a collection of particles in the NVT ensemble. \
                Applies overdamped langevin dynamics to update the system.")
            .arg(Arg::with_name("NUM")
                .short("n")
                .long("num")
                .help("Total number of particles in the box")
                .takes_value(true)
                .default_value("10"))
            .arg(Arg::with_name("NUMA")
                .short("a")
                .long("num-a")
                .help("Max number of A (small) particles in the box. For the LJ and WCA potentials this corresponds to the particles that are traditionally labeled B in the literature")
                .takes_value(true)
                .default_value("5"))
            .arg(Arg::with_name("LEN")
                .short("l")
                .long("len")
                .help("Side length of the (square or cubic) simulation box")
                .takes_value(true)
                .default_value("3.0"))
            .arg(Arg::with_name("PHI")
                .long("phi")
                .takes_value(true)
                .help("Specify the packing fraction instead of box length [conflicts with: LEN]")
                .conflicts_with("LEN"))
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
                .help("Rescaling distance of the inter-atomic potentials")
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
                .help("Dimensions of the simulation box")
                .takes_value(true)
                .default_value("2")
                .possible_values(&["2", "3"]))
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
            .arg(Arg::with_name("STDOUT")
                .short("i")
                .long("stdout-time")
                .help("Time between terminal writes")
                .takes_value(true))
            .arg(Arg::with_name("POT")
                .long("potential")
                .help("Bidisperse potential used in the simulation")
                .takes_value(true)
                .default_value("hertz")
                .possible_values(Potential::valid_strs()))
            .arg(Arg::with_name("INIT_CONFIG")
                .long("init-config")
                .help("JSON config file initializing the system state. \
                    Will assert that the config is valid [conflicts with: NUM, NUMA, LEN, PHI, and DIM]")
                .takes_value(true)
                .conflicts_with("NUM")
                .conflicts_with("NUMA")
                .conflicts_with("LEN")
                .conflicts_with("PHI")
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
            .arg(Arg::with_name("IMAGES")
                .long("images")
                .help("Force computation of all periodic images"))
            .subcommand(SubCommand::with_name("gen-variant")
                .about("Run simulation alongside Chen and Horing method to compute bias potentials. We denote each ")
                .arg(Arg::with_name("REALIZATIONS")
                    .long("realizations")
                    .takes_value(true)
                    .default_value("1000000")
                    .help("Number of realizations to run"))
                .arg(Arg::with_name("DELVAR")
                    .long("del-var")
                    .use_delimiter(true)
                    .takes_value(true)
                    .help("Optional variants to compute. This is a list of \\chi values that describe the relative difference in the potential prefactors between the reference and target simulations"))
                .arg(Arg::with_name("CALCMSD")
                    .requires("DELVAR")
                    .long("calc-msd")
                    .help("Calculate biased MSD"))
                .arg(Arg::with_name("CALCPOS")
                    .requires("DELVAR")
                    .long("calc-pos")
                    .help("Calculate biased particle trajectory for each system"))
                .arg(Arg::with_name("CALCQ")
                    .requires("DELVAR")
                    .long("calc-q")
                    .use_delimiter(true)
                    .takes_value(true)
                    .help("Calculate biased Q(a) (simple overlap function)")))
            .subcommand(SubCommand::with_name("equil-gd")
                .about("Generate loadable simulation config quenched to its inherent structure. By default ignores TIME unless the --melt flag is provided")
                .arg(Arg::with_name("MAX_DR")
                    .required(true)
                    .default_value("1e-10")
                    .help("Maximum allowed step displacement size before stopping"))
                .arg(Arg::with_name("MAX_F")
                    .required(true)
                    .default_value("1e-10")
                    .help("Maximum allowed inter-particle force before stopping"))
                .arg(Arg::with_name("MELT")
                    .long("melt")
                    .help("Run Langevin dynamics before quenching")))
            .get_matches();

        let num = conv_match(&matches, "NUM");
        let numa = conv_match(&matches, "NUMA");
        let len = conv_match(&matches, "LEN");
        let temp = conv_match(&matches, "TEMP");
        let time = conv_match::<f64>(&matches, "TIME");
        let dt = conv_match::<f64>(&matches, "DT");
        let visc = conv_match(&matches, "VISC");
        let dim = conv_match(&matches, "DIM");
        let write_time = conv_match::<f64>(&matches, "OUT");
        let stdout_time = conv_optional_match::<f64>(&matches, "STDOUT");
        let phi = conv_optional_match::<f64>(&matches, "PHI");
        let seed = conv_match(&matches, "SEED");
        let dir = conv_match(&matches, "DIR");
        let rscale = conv_match(&matches, "RSCALE");
        let vscale = conv_match(&matches, "VSCALE");
        let pot_str = matches.value_of("POT").unwrap();

        let potential = Potential::from_str(pot_str).unwrap();

        // We don't want to fiddle with this any longer
        assert_eq!(rscale, 1.0, "The 'rscale' parameter is deprecated in this analysis.");

        let dryprint = matches.is_present("DRYPRINT");
        let unwrap = matches.is_present("UNWRAP");
        let images = matches.is_present("IMAGES");
        let init_config = matches.value_of("INIT_CONFIG").map(|path| path.to_string());

        let mode: ProgramMode = {
            if let Some(equil_match) = matches.subcommand_matches("equil-gd") {
                let max_dr: f64 = conv_match(&equil_match, "MAX_DR");
                let max_f: f64 = conv_match(&equil_match, "MAX_F");
                let melt = equil_match.is_present("MELT");
                ProgramMode::Equilibrate(max_dr, max_f, melt)
            }
            else if let Some(gen_variant_match) = matches.subcommand_matches("gen-variant") {
                let realizations = conv_match(&gen_variant_match, "REALIZATIONS");
                let delvar = gen_variant_match.values_of("DELVAR")
                    .and_then(|vals| Some(vals.map(|e| FromStr::from_str(e).unwrap()).collect::<Vec<f64>>()));
                if delvar.is_some() {
                    let msd = gen_variant_match.is_present("CALCMSD");
                    let pos = gen_variant_match.is_present("CALCPOS");
                    let calc_q = gen_variant_match.values_of("CALCQ")
                        .and_then(|vals| Some(vals.map(|e| FromStr::from_str(e).unwrap()).collect::<Vec<f64>>()));
                    if !(msd || pos || calc_q.is_some()) {
                        panic!("No computable quantities specified!");
                    }
                    ProgramMode::GenVariant(realizations, Some((delvar.unwrap(), msd, pos, calc_q)))
                }
                else { ProgramMode::GenVariant(realizations, None) }
                
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

        Config{num, numa, len, temp, time, step_max, dt, visc, 
                dim, write_step, stdout_step, seed, dir,
                dryprint, rscale, vscale, mode, init_config,
                unwrap, phi, potential, images}
    }

    // format output file suffix with configuration data
    pub fn file_suffix(&self) -> String {
        format!("n-{}_l-{}_t-{}_time-{}_dt-{:e}_visc-{}_seed-{}", 
                self.num, self.len, self.temp, self.time, 
                self.dt, self.visc, self.seed)
    }
}