# md-lv
Overdamped Langevin dynamics with periodic boundary conditions in Rust. This code is meant to be run on very small systems (N < 100) due to the lack of internal neighbor data.

![Ovito output](movies/liquid1.gif)

# Usage Description
```
Langevin dynamics simulation 0.2.1
Ian Graham <irgraham1@gmail.com>
Runs a simulation of a collection of Hertzian particles in the NVT ensemble. Applies overdamped langevin dynamics to update the system.

USAGE:
    md-lv [OPTIONS]

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -d, --dim <DIM>        Dimensions of the simulation box (2 or 3) [default: 2]
        --dt <DT>          Size of the system timestep [default: 1e-3]
    -i, --iostep <IO>      Number of steps between messages to stdout [default: 100_000]
    -n, --num <NUM>        Number of particles in the box [default: 10]
    -o, --outstep <OUT>    Number of steps between dump to output [default: 10]
        --seed <SEED>      Random seed to initialize the system state [default: 0]
    -s, --steps <STEP>     Maximum number of simulation steps [default: 100_000]
    -t, --temp <TEMP>      Temperature of the system [default: 0.5]
        --visc <VISC>      Viscous drag coefficient on the particles of the system [default: 5.0]
    -v, --vol <VOL>        Volume (area) of the box [default: 6.5]
```