use crate::Atom;
pub struct Gaussian {
    pub center: [f64; 3],
    pub exponent: f64,
    pub coefficient: f64,
    pub norm: f64,
}

pub struct ContractedGaussian {
    pub center: [f64; 3],
    pub primitives: Vec<Gaussian>
}

// N(α) = (2α/π)^(3/4)
fn normalization(alpha: f64) -> f64 {
    (2.0 * alpha / std::f64::consts::PI).powf(0.75)
}

pub fn build_basis(atoms: Vec<Atom>, basis_name: String) -> Vec<ContractedGaussian> {
    let mut  basis = Vec::new();

    for atom in atoms {
        let center = [atom.position.0 * 1.8897259886, atom.position.1 * 1.8897259886, atom.position.2 * 1.8897259886];

        let primiives_data =  match basis_name.as_str() {
            "gto" => gto(&atom.element),
            _ => panic!("Basis set not implemented: {}", basis_name),
        };
        
        let primitives: Vec<Gaussian> = primiives_data.into_iter()
            .map(|(alpha, coefficient)| Gaussian {
                center,
                exponent: alpha,
                coefficient,
                norm: normalization(alpha),
            })
            .collect();
        let mut contracted_gaussian = ContractedGaussian { center, primitives };
        
        basis.push(contracted_gaussian);
        }

    basis
}

fn gto(element: &str) -> Vec<(f64, f64)> {
    match element {
        "H"  => vec![(0.4166, 1.0)],
        "He" => vec![(0.7739, 1.0)],
        _ => panic!("GTO not implemented for element: {}", element),
    }
}

