# Hartree–Fock in Rust

> 📄 For a detailed description of the implementation, see [Hartree–Fock-rust-implementation.pdf](Hartree–Fock-rust-implemntation.pdf).

A Rust implementation of the Restricted Hartree–Fock (RHF) method for small molecules.

- Reads molecular geometry and basis set from `input.json`
- Supports STO-nG basis sets (STO-1G to STO-6G)
- Computes overlap, core Hamiltonian, and two-electron integrals
- Runs SCF iterations until convergence
- Outputs total energy to the console and `run.log`

## Requirements

Requires the Rust `cargo` package manager:

```bash
sudo apt install rustc cargo
```

## Build & Run

```bash
git clone https://github.com/pritesh299/hartee_fock.git
cd hartee_fock
cargo build
```

## Input File

Edit `input.json` to configure the molecule, basis set, and SCF parameters:
Important parameters
basis:['sto-1g','sto-2g','sto-3g','sto-4g','sto-5g','sto-6g']
element: ['H','He']
Position: [x,y,z]
Example
```json
{
  "molecule": {
    "charge": 0,
    "atoms": [
      { "element": "H", "position": [0.0, 0.0, 0.0] },
      { "element": "H", "position": [0.74, 0.0, 0.0] }
    ]
  },
  "basis": "sto-3g",
  "scf": { "max_iter": 100, "tol": 1e-8 }
}
```

Then run:

```bash
cargo run
```

Alternatively, run the batch script:

```bash
bash run_HF.sh
```

This runs a series of test cases and prints the resulting HF energies.

## Expected Output

```
Running: H_atom | sto-1g
E_total = -0.42441318157838764
Running: He_atom | sto-1g
E_total = -2.3009405882292153
Running: H2_0.74A | sto-1g
E_total = -0.9390654918031686
Running: H2_1.5A | sto-1g
E_total = -0.8776612394188507
Running: H2_3.0A | sto-1g
E_total = -0.6581200067100337
.
.
.
Running: H_atom | sto-6g
E_total = -0.47103905417809255
Running: He_atom | sto-6g
E_total = -2.8462920947813606
Running: H2_0.74A | sto-6g
E_total = -1.1253721375239847
Running: H2_1.5A | sto-6g
E_total = -0.9189359504793427
Running: H2_3.0A | sto-6g
E_total = -0.6656565216167651
Running: HeH_plus | sto-6g
E_total = -2.8645967713698774
```
