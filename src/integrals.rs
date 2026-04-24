use std::f64::consts::PI;
use crate::{Atom, basis::{ContractedGaussian, Gaussian}};



/// Squared distance between two 3D centres
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

/// Boys function F0(x) = 0.5 * sqrt(π/x) * erf(sqrt(x))
/// Used in nuclear-attraction and ERI integrals.
/// For x ≈ 0 we use the limiting value 1.
fn boys0(x: f64) -> f64 {
    if x < 1e-8 {
        1.0 - x / 3.0 
    } else {
        0.5 * (std::f64::consts::PI / x).sqrt() * erf(x.sqrt())
    }
}

// Error function via Horner approximation (Abramowitz & Stegun 7.1.26)
fn erf(x: f64) -> f64 {
    if x < 0.0 { return -erf(-x); }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    1.0 - poly * (-x * x).exp()
}

pub fn nuclear_charge(element: &str) -> f64 {
    match element {
        "H"  => 1.0,
        "He" => 2.0,
        _    => panic!("unknown element: {}", element),
    }

}

/// Gaussian product centre and exponent
fn product(ai: f64, ri: [f64; 3], aj: f64, rj: [f64; 3]) -> (f64, [f64; 3]) {
    let ap = ai + aj;
    let rp: [f64; 3] = std::array::from_fn(|k| (ai * ri[k] + aj * rj[k]) / ap);
    (ap, rp)
}

// overlap between two primitive gaussians
fn compute_primitive_overlap(phi_1: &Gaussian, phi_2: &Gaussian) -> f64 {
    let alpha1 = phi_1.exponent;
    let alpha2 = phi_2.exponent;
    let ap = alpha1 + alpha2;
    let rij2 = dist2(phi_1.center, phi_2.center);

    let prefactor = (PI / ap).powf(1.5);
    let exponent = (-alpha1 * alpha2 / ap * (rij2)).exp();

    phi_1.norm* phi_2.norm * prefactor * exponent
}

// overlap between two contracted gaussians
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

// overlap matrix S_ij = <psi_i | psi_j>
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

// primitive kinetic
fn compute_kinetic_primitive(phi_1: &Gaussian, phi_2: &Gaussian) -> f64 {
    let ai = phi_1.exponent;
    let aj = phi_2.exponent;

    let ap = ai + aj;
    let rab2 = dist2(phi_1.center, phi_2.center);

    let reduced = ai * aj / ap;

    let prefac = (PI / ap).powf(1.5);
    let decay  = (-reduced* rab2).exp();

    phi_1.norm * phi_2.norm * phi_1.coefficient * phi_2.coefficient *
        prefac *
        reduced * (3.0 - 2.0 * reduced * rab2) *
        decay 
}

// contracted kinetic energy 
pub fn compute_kinetic(psi_1: &ContractedGaussian, psi_2: &ContractedGaussian) -> f64 {
    let mut T_ij = 0.0;

    for phi_i in &psi_1.primitives {
        for phi_j in &psi_2.primitives {
            T_ij += compute_kinetic_primitive(phi_i, phi_j);
        }
    }
    T_ij
}

// kinetic matrix T_ij = <psi_i | -1/2 ∇^2 | psi_j>
pub fn compute_kinetic_matrix(basis: &[ContractedGaussian]) -> Vec<Vec<f64>> {
    let n = basis.len();
    let mut T = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let val = compute_kinetic(&basis[i], &basis[j]);
            T[i][j] = val;
            T[j][i] = val;//symmetry
        }
    }
    T
}

// primitive nuclear attraction
fn compute_nuclear_attraction_primitive(
    phi_1: &Gaussian,
    phi_2: &Gaussian,
    R_k: [f64; 3],
    Z_k: f64,
) -> f64 {
    let ai = phi_1.exponent;
    let aj = phi_2.exponent;

    let (ap, rp) = product(ai, phi_1.center, aj, phi_2.center);

    let rij2 = dist2(phi_1.center, phi_2.center);
    let rpc2 = dist2(rp, R_k);

    let prefac = -Z_k * 2.0 * PI / ap;
    let decay  = (-ai * aj / ap * rij2).exp();

    phi_1.norm * phi_2.norm * phi_1.coefficient * phi_2.coefficient
        * prefac * decay * boys0(ap * rpc2)
}


// contracted nuclear attraction
pub fn compute_nuclear_attraction(
    psi_i: &ContractedGaussian,
    psi_j: &ContractedGaussian,
    nuclei: &[([f64; 3], f64)],
) -> f64 {
    let mut v = 0.0;

    for &(R_k, Z_k) in nuclei {
        let mut v_ij = 0.0;

        for phi_i in &psi_i.primitives {
            for phi_j in &psi_j.primitives {
                v_ij += compute_nuclear_attraction_primitive(phi_i, phi_j, R_k, Z_k);
            }
        }
        v += v_ij;
    }

    v
}

//nuclear matrix V_ij = <psi_i | V_nuc | psi_j>
pub fn compute_nuclear_attraction_matrix(
    basis: &[ContractedGaussian],
    nuclei_atoms: &[Atom],
) -> Vec<Vec<f64>> {
    let n = basis.len();
    let mut v = vec![vec![0.0; n]; n];

    let nuclei: Vec<([f64; 3], f64)> = nuclei_atoms.iter()
    .map(|a| (
        [
            a.position.0 * 1.8897259886,
            a.position.1 * 1.8897259886,
            a.position.2 * 1.8897259886,
        ],
        nuclear_charge(&a.element)))
    .collect();

    for i in 0..n {
        for j in 0..=i {
            let val = compute_nuclear_attraction(&basis[i], &basis[j],  &nuclei);
            v[i][j] = val;
            v[j][i] = val; //symmetry
        }
    }

    v
}


// ── ERI: two-electron repulsion (ijkl) ───────────────────────────────────────

/// Primitive ERI: (ij|kl) = ∫∫ φ_i(r₁) φ_j(r₁) 1/|r₁-r₂| φ_k(r₂) φ_l(r₂) dr₁ dr₂
fn compute_eri_primitive(
    gi: &Gaussian, gj: &Gaussian,
    gk: &Gaussian, gl: &Gaussian,
) -> f64 {
    let (ai, ri) = (gi.exponent, gi.center);
    let (aj, rj) = (gj.exponent, gj.center);
    let (ak, rk) = (gk.exponent, gk.center);
    let (al, rl) = (gl.exponent, gl.center);

    let (ap, rp) = product(ai, ri, aj, rj);
    let (aq, rq) = product(ak, rk, al, rl);

    let rij2 = dist2(ri, rj);
    let rkl2 = dist2(rk, rl);
    let rpq2 = dist2(rp, rq);

    let prefac = 2.0 * PI.powi(2) * (PI / (ap + aq)).sqrt() / (ap * aq);
    let decay  = (-ai * aj / ap * rij2 - ak * al / aq * rkl2).exp();
    let theta  = ap * aq / (ap + aq);

    gi.norm * gj.norm * gk.norm * gl.norm
        * gi.coefficient * gj.coefficient * gk.coefficient * gl.coefficient
        * prefac * decay * boys0(theta * rpq2)
}

/// Contracted ERI: sums over primitives
pub fn compute_eri(
    gi: &ContractedGaussian,
    gj: &ContractedGaussian,
    gk: &ContractedGaussian,
    gl: &ContractedGaussian,
) -> f64 {
    let mut val = 0.0;

    for pi in &gi.primitives {
        for pj in &gj.primitives {
            for pk in &gk.primitives {
                for pl in &gl.primitives {
                    val += compute_eri_primitive(pi, pj, pk, pl);
                }
            }
        }
    }

    val
}

/// Build ERI tensor: flattened [i*n³ + j*n² + k*n + l]
pub fn compute_eri_tensor(basis: &[ContractedGaussian]) -> Vec<f64> {
    let n = basis.len();
    let mut tensor = vec![0.0f64; n * n * n * n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let idx = i * n * n * n + j * n * n + k * n + l;
                    tensor[idx] = compute_eri(&basis[i], &basis[j], &basis[k], &basis[l]);
                }
            }
        }
    }

    tensor
}

// H_core = T + V
pub fn Hcore_matrix(
    basis_length: usize,
    kinetic: &[Vec<f64>],
    nuclear: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let n = basis_length;
    let mut H = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            H[i][j] = kinetic[i][j] + nuclear[i][j];
        }
    }

    H   
}
