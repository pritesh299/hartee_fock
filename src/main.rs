use serde::Deserialize;

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
    println!("Molecule charge: {}", config.molecule.charge);

    
}