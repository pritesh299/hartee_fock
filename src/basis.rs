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
            "sto-4g" => sto4g(&atom.element),
            "sto-5g" => sto5g(&atom.element),
            "sto-6g" => sto6g(&atom.element),
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
        "H"  => vec![(0.2829421200, 1.0)],
        "He" => vec![(0.7739, 1.0)],
        _ => panic!("STO-1G not implemented for element: {}", element),
    }
}


fn sto2g(element: &str) -> Vec<(f64, f64)> {
    match element {
        "H" => vec![
            (0.2331359749, 0.678914),
            (1.309756377, 0.430129),
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
            (3.425250914, 0.1543289673),
            (0.6239137298, 0.5353281423),
            (0.1688554040, 0.4446345422),
        ],
        "He" => vec![
            (6.362421394, 0.1543289673),
            (1.158922999, 0.5353281423),
            (0.3136497915, 0.4446345422),
        ],
        _ => panic!("STO-3G not implemented for element: {}", element),
    }
}

fn sto4g(element: &str) -> Vec<(f64, f64)> {
    match element {
        "H" => vec![
            (8.021420155, 0.05675242080),
            (1.467821061, 0.2601413550),
            (0.4077767635, 0.5328461143),
            (0.1353374420, 0.2916254405),
        ],
        "He" => vec![
            (14.89982967, 0.05675242080),
            (2.726485258, 0.2601413550),
            (0.7574474599, 0.5328461143),
            (0.2513900027, 0.2916254405),
        ],
        _ => panic!("STO-4G not implemented for element: {}", element),
    }
}

fn sto5g(element: &str) -> Vec<(f64, f64)> {
    match element {
        "H" => vec![
            (17.38354739, 0.02214055312),
            (3.185489246, 0.1135411520),
            (0.8897299079, 0.3318161484),
            (0.3037874103, 0.4825700713),
            (0.1144784984, 0.1935721966),
        ],
        "He" => vec![
            (32.29002972, 0.02214055312),
            (5.917062849, 0.1135411520),
            (1.652677933, 0.3318161484),
            (0.5642866953, 0.4825700713),
            (0.2126444063, 0.1935721966),
        ],
        _ => panic!("STO-5G not implemented for element: {}", element),
    }
}

fn sto6g(element: &str) -> Vec<(f64, f64)> {
    match element {
        "H" => vec![
            (35.52322122, 0.009163596281),
            (6.513143725, 0.04936149294),
            (1.822142904, 0.1685383049),
            (0.6259552659, 0.3705627997),
            (0.2430767471, 0.4164915298),
            (0.1001124280, 0.1303340841),
        ],
        "He" => vec![
            (65.98456824, 0.009163596281),
            (12.09819836, 0.04936149294),
            (3.384639924, 0.1685383049),
            (1.162715163, 0.3705627997),
            (0.4515163224, 0.4164915298),
            (0.1859593559, 0.1303340841),
        ],
        _ => panic!("STO-6G not implemented for element: {}", element),
    }
}