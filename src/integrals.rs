use crate::basis::{ContractedGaussian, Gaussian};

fn compute_primitive_overlap(phi_1: &Gaussian, phi_2: &Gaussian) -> f64 {
    let r1 = phi_1.center;
    let r2 = phi_2.center;
    let alpha1 = phi_1.exponent;
    let alpha2 = phi_2.exponent;

    let prefactor = (std::f64::consts::PI / (alpha1 + alpha2)).powf(1.5);
    let exponent = (-alpha1 * alpha2 / (alpha1 + alpha2) * ((r1[0] - r2[0]).powi(2) + (r1[1] - r2[1]).powi(2) + (r1[2] - r2[2]).powi(2))).exp();

    phi_1.norm* phi_2.norm * prefactor * exponent
}

fn compute_overlap(psi_1: &ContractedGaussian, psi_2: &ContractedGaussian) -> f64 {
    let mut overlap = 0.0;

    for phi_1 in &psi_1.primitives {
        for phi_2 in &psi_2.primitives {
            let s = compute_primitive_overlap(phi_1, phi_2);
            overlap += phi_1.coefficient * phi_2.coefficient * s;
        }
    }

    overlap
}


pub fn compute_overlap_matrix(basis: &Vec<ContractedGaussian>) -> Vec<Vec<f64>> {
    let n = basis.len();
    let mut overlap_matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            overlap_matrix[i][j] = compute_overlap(&basis[i], &basis[j]);
        }
    }

    overlap_matrix
}