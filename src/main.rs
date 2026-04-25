
mod basis;
mod integrals;
mod scf;

use serde::Deserialize;
use basis::build_basis;
#[derive(Deserialize)]
pub struct Atom {
    pub element: String,
    pub position: (f64, f64, f64),
}

#[derive(Deserialize)]
pub struct Molecule {
    pub charge: i32,
    pub atoms: Vec<Atom>,
}

#[derive(Deserialize)]
pub struct SCFConfig {
    pub max_iter: u32,
    pub tol: f64,
}

#[derive(Deserialize)]
pub struct InputConfig {
    pub molecule: Molecule,
    pub basis: String,
    pub scf: SCFConfig,
}

fn main() {
       
    let json = std::fs::read_to_string("input.json").unwrap();
    let config: InputConfig = serde_json::from_str(&json).unwrap();

    let basis = build_basis(&config.molecule.atoms, &config.basis);

    let overlap_matrix = integrals::compute_overlap_matrix(&basis);
    let kinetic_matrix = integrals::compute_kinetic_matrix(&basis);
    let nuclear_attraction_matrix = integrals::compute_nuclear_attraction_matrix(&basis, &config.molecule.atoms);
    let eri_tensor = integrals::compute_eri_tensor(&basis);
    let hcore_matrix = integrals::hcore_matrix(basis.len(), &kinetic_matrix, &nuclear_attraction_matrix);
    let n_elec: usize = config.molecule.atoms.iter()
    .map(|a| integrals::nuclear_charge(&a.element) as usize)
    .sum::<usize>() - config.molecule.charge as usize;
    
    scf::run_scf(&overlap_matrix, &hcore_matrix, &eri_tensor, &config.molecule.atoms, n_elec, config.scf.max_iter, config.scf.tol);
}