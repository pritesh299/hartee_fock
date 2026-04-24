#!/bin/bash
set -e

OUTPUT_FILE="results.csv"
LOG_FILE="run.log"

# reset files
echo "case,E_total" > $OUTPUT_FILE
echo "Hartree-Fock Test Run" > $LOG_FILE
run_case () {
    name=$1
    json=$2

    echo "Running: $name"

    echo "$json" > input.json

    # run and capture full output
    output=$(cargo run --quiet -- input.json)

    # print full output to log
    echo "$name " >> $LOG_FILE
    echo "$output" >> $LOG_FILE
    echo "" >> $LOG_FILE

    # extract energy
    energy=$(echo "$output" | grep "E_total" | awk '{print $3}')

    echo "E_total = $energy"
    echo ""

    # write to CSV
    echo "$name,$energy" >> $OUTPUT_FILE
}

# Hydrogen atom
run_case "H atom" '{
  "molecule": {
    "charge": 0,
    "atoms": [
      { "element": "H", "position": [0.0, 0.0, 0.0] }
    ]
  },
  "basis": "sto-3g",
  "scf": { "max_iter": 50, "tol": 1e-6 }
}'

# Helium atom
run_case "He atom" '{
  "molecule": {
    "charge": 0,
    "atoms": [
      { "element": "He", "position": [0.0, 0.0, 0.0] }
    ]
  },
  "basis": "sto-3g",
  "scf": { "max_iter": 50, "tol": 1e-6 }
}'

# H2 molecule (0.74 Å)
run_case "H2 (0.74 Å)" '{
  "molecule": {
    "charge": 0,
    "atoms": [
      { "element": "H", "position": [0.0, 0.0, 0.0] },
      { "element": "H", "position": [0.0, 0.0, 0.74] }
    ]
  },
  "basis": "sto-3g",
  "scf": { "max_iter": 50, "tol": 1e-6 }
}'

# H2 stretched (1.5 Å)
run_case "H2 (1.5 Å)" '{
  "molecule": {
    "charge": 0,
    "atoms": [
      { "element": "H", "position": [0.0, 0.0, 0.0] },
      { "element": "H", "position": [0.0, 0.0, 1.5] }
    ]
  },
  "basis": "sto-3g",
  "scf": { "max_iter": 50, "tol": 1e-6 }
}'

# H2 very stretched (3.0 Å)
run_case "H2 (3.0 Å)" '{
  "molecule": {
    "charge": 0,
    "atoms": [
      { "element": "H", "position": [0.0, 0.0, 0.0] },
      { "element": "H", "position": [0.0, 0.0, 3.0] }
    ]
  },
  "basis": "sto-3g",
  "scf": { "max_iter": 50, "tol": 1e-6 }
}'