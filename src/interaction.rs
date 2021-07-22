pub struct LangevinSimulation {
    pub sys: System,
    pub rng: rand_pcg::Lcg128Xsl64,
    pub normal: rand_distr::Normal<f64>,
    pub integration_terms: [f64; 2],
    pub file: std::io::BufWriter<std::fs::File>,
    pub interaction: Box<dyn Interaction<Params = dyn InteractionParameters>>
}

pub struct System {
    pub pos: Vec<[f64; 3]>,
    pub types: Vec<usize>,
    pub b: [f64; 3],
    pub bh: [f64; 3],
    pub species: Vec<u8>,
    pub dim: usize
}

pub trait InteractionParameters {}

pub trait Interaction {
    type Params: InteractionParameters;
    // fn sys_potential(&self, sys: &System) -> f64;
    // fn sys_force(&self, sys: &System) -> Vec<f64>;
    fn potential(&self, dr: f64) -> f64;
    fn force(&self, dr: f64, vdr: [f64; 3]) -> [f64; 3];
}

pub struct HertzianParameters {

}

impl InteractionParameters for HertzianParameters {}

pub struct HertzianInteraction {
    pub params: HertzianParameters
}

impl Interaction for HertzianInteraction {

    type Params = HertzianParameters;

    fn potential(&self, dr: f64) -> f64 {
        todo!();
        1.0
    }

    fn force(&self, dr: f64, vdr: [f64; 3]) -> [f64; 3] {
        todo!();
        [1.0, 1.0, 1.0]
    }
}