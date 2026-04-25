# Hartree–Fock in Rust

A Rust implementation of the Restricted Hartree–Fock (RHF) method for small molecules.

- Reads molecular geometry and basis set from `input.json`
- Supports STO‑nG basis sets (STO‑1G to STO‑6G)
- Computes overlap, core Hamiltonian, and two‑electron integrals
- Runs SCF iterations until convergence
- Outputs total energy to console and `run.log`

requires rust cargo package manneger 
``` bash
sudo apt install rustc cargo
```

##  run Build & Run

```bash
git clone https://github.com/pritesh299/hartee_fock.git
cd hartee_fock
cargo build 
```

## Input file
Can make changes in input.json to run
```JSON
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
Then following

```bash
cargo run 
```