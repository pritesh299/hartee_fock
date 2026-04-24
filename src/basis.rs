use crate::Atom;
pub struct Gaussian {
    pub center: [f64; 3],
    pub exponent: f64,
    pub coefficient: f64,
    pub norm: f64,
}

pub struct ContractedGaussian {
    pub primitives: Vec<Gaussian>
}

// N(α) = (2α/π)^(3/4)
fn normalization(alpha: f64) -> f64 {
    (2.0 * alpha / std::f64::consts::PI).powf(0.75)
}

pub fn build_basis(atoms: &[Atom], basis_name: &str) -> Vec<ContractedGaussian> {
    let mut  basis = Vec::new();

    for atom in atoms {
        let center = [atom.position.0 * 1.8897259886, atom.position.1 * 1.8897259886, atom.position.2 * 1.8897259886];

        let primiives_data =  match basis_name {
            "sto-1g" => sto1g(&atom.element),
            "sto-2g" => sto2g(&atom.element),
            "sto-3g" => sto3g(&atom.element),
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
        let  contracted_gaussian = ContractedGaussian { primitives };
        
        basis.push(contracted_gaussian);
        }

    basis
}

fn sto1g(element: &str) -> Vec<(f64, f64)> {
    match element {
        "H"  => vec![(0.4166, 1.0)],
        "He" => vec![(0.7739, 1.0)],
        _ => panic!("STO-1G not implemented for element: {}", element),
    }
}


fn sto2g(element: &str) -> Vec<(f64, f64)> {
    match element {
        "H" => vec![
            (0.151623, 0.678914),
            (0.851819, 0.430129),
        ],
        "He" => vec![
            (2.4328790, 0.4301285),
            (0.4330510, 0.6789135),
        ],
        
        _ => panic!("STO-2G not implemented for element: {}", element),
    }
}

// ── STO-3G ─────────────────────────────────────────

fn sto3g(element: &str) -> Vec<(f64, f64)> {
    match element {
        "H" => vec![
            (3.4252509, 0.1543290),
            (0.6239137, 0.5353281),
            (0.1688554, 0.4446345),
        ],
        "He" => vec![
            (6.3624214, 0.1543290),
            (1.1589230, 0.5353281),
            (0.3136498, 0.4446345),
        ],
        _ => panic!("STO-3G not implemented for element: {}", element),
    }
}
