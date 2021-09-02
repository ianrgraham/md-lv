use ndarray::*;
use serde::*;

pub struct LangevinSimulation<F, T>
    where 
        for<'a> T: Interaction<'a> {
    sys: System,
    rng: rand_pcg::Lcg128Xsl64,
    normal: rand_distr::Normal<f64>,
    file: F,
    interaction: T
}

pub struct System {
    pos: Array2<f64>,
    forces: Array2<f64>,
    types: Array1<u8>,
    b: [f64; 3],
    bh: [f64; 3],
    dim: usize
}

pub trait Interaction<'t>: Serialize + Deserialize<'t> {
    fn sys_potential(&self, sys: &System) -> f64;
    fn sys_forces<'a>(&self, sys: &'a mut System) -> ArrayView2<'a, f64>;
    fn len(&self) -> usize;

}

#[derive(Serialize, Deserialize)]
struct BidisperseHertzian {
    sigmas: [f64; 2],
    coeff: f64
}

impl<'t> Interaction<'t> for BidisperseHertzian {

    fn sys_potential(&self, sys: &System) -> f64 {
        1.0
    }

    fn sys_forces<'a>(&self, sys: &'a mut System) -> ArrayView2<'a, f64> {
        sys.forces.view()
    }

    fn len(&self) -> usize {
        2
    }
}