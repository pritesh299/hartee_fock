use ndarray::{Array2};
use ndarray_linalg::{Eigh, UPLO};

use crate::Atom;
use crate::integrals;



fn vec_to_array(v: &[Vec<f64>]) -> Array2<f64> {
    let n = v.len();
    Array2::from_shape_vec((n, n), v.iter().flatten().cloned().collect()).unwrap()
}


// S^{-1/2}

fn s_invsqrt(s: &[Vec<f64>]) -> Array2<f64> {
    let s_arr = vec_to_array(s);

    let (evals, evecs) = s_arr.eigh(UPLO::Upper).unwrap();

    let n = evals.len();
    let mut d = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        d[[i, i]] = 1.0 / evals[i].sqrt();
    }

    evecs.dot(&d).dot(&evecs.t())
}

// Density matrix (supports odd electrons)

fn density_matrix(c: &Array2<f64>, n_elec: usize) -> Array2<f64> {
    let (n, _) = c.dim();
    let n_occ = n_elec / 2;
    let is_odd = n_elec % 2 == 1;

    let mut d = Array2::<f64>::zeros((n, n));

    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;

            for i in 0..n_occ {
                sum += 2.0 * c[[mu, i]] * c[[nu, i]];
            }

            if is_odd {
                let i = n_occ;
                sum += c[[mu, i]] * c[[nu, i]];
            }

            d[[mu, nu]] = sum;
        }
    }

    d
}

// Fock matrix

fn fock_matrix(h: &Array2<f64>, d: &Array2<f64>, eri: &[f64]) -> Array2<f64> {
    let n = h.dim().0;
    let mut f = h.clone();

    for i in 0..n {
        for j in 0..n {
            let mut g = 0.0;

            for k in 0..n {
                for l in 0..n {
                    let coulomb = eri[i*n*n*n + j*n*n + k*n + l];
                    let exchange = eri[i*n*n*n + k*n*n + j*n + l];

                    g += d[[k, l]] * (coulomb - 0.5 * exchange);
                }
            }

            f[[i, j]] += g;
        }
    }

    f
}

// Energy

fn electronic_energy(d: &Array2<f64>, h: &Array2<f64>, f: &Array2<f64>) -> f64 {
    let n = d.dim().0;

    let mut e = 0.0;
    for i in 0..n {
        for j in 0..n {
            e += d[[i, j]] * (h[[i, j]] + f[[i, j]]);
        }
    }

    0.5 * e
}

// Nuclear repulsion

pub fn nuclear_repulsion(nuclei: &[([f64; 3], f64)]) -> f64 {
    let mut e = 0.0;

    for a in 0..nuclei.len() {
        for b in (a + 1)..nuclei.len() {
            let (ra, za) = nuclei[a];
            let (rb, zb) = nuclei[b];

            let r = ((0..3).map(|k| (ra[k] - rb[k]).powi(2)).sum::<f64>()).sqrt();

            e += za * zb / r;
        }
    }

    e
}

// SCF

pub fn run_scf(
    s: &[Vec<f64>],
    h: &[Vec<f64>],
    eri: &[f64],
    nuclei_atoms: &[Atom],
    n_elec: usize,
    max_iter: u32,
    tol: f64,
) {

    let nuclei: Vec<([f64; 3], f64)> = nuclei_atoms.iter()
    .map(|a| (
        [
            a.position.0 * 1.8897259886,
            a.position.1 * 1.8897259886,
            a.position.2 * 1.8897259886,
        ],
        integrals::nuclear_charge(&a.element)))
    .collect();
    let n = s.len();

    let s_inv = s_invsqrt(s);
    let h_arr = vec_to_array(h);

    let mut d = Array2::<f64>::zeros((n, n));
    let mut e_prev = 0.0;

    println!("iter     E_elec          ΔE");

    for iter in 0..max_iter {
        let f = fock_matrix(&h_arr, &d, eri);

        let f_prime = s_inv.t().dot(&f).dot(&s_inv);

        let (_eps, c_prime) = f_prime.eigh(UPLO::Upper).unwrap();

        let c = s_inv.dot(&c_prime);

        d = density_matrix(&c, n_elec);

        let e = electronic_energy(&d, &h_arr, &f);
        let delta = (e - e_prev).abs();

        println!("{:4}  {:14.8}  {:10.2e}", iter+1, e, delta);

        if iter > 0 && delta < tol {
            println!("SCF converged\n");
            break;
        }

        e_prev = e;
    }

    let e_elec = electronic_energy(&d, &h_arr, &fock_matrix(&h_arr, &d, eri));
    let e_nuc = nuclear_repulsion(&nuclei);

    println!("E_total = {}", e_elec + e_nuc);
}
