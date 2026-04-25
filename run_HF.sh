#!/bin/bash
set -e

OUTPUT_FILE="results.csv"
LOG_FILE="run.log"

echo "case,basis,E_total" > $OUTPUT_FILE
echo "Hartree-Fock Test Run" > $LOG_FILE


BASIS_LIST=("sto-1g" "sto-2g" "sto-3g" "sto-4g" "sto-5g" "sto-6g")

run_case () {
    name=$1
    basis=$2
    json=$3

    echo "Running: $name | $basis"

    echo "$json" > input.json

    output=$(cargo run --quiet -- input.json)

    echo "$name | $basis" >> $LOG_FILE
    echo "$output" >> $LOG_FILE
    echo "" >> $LOG_FILE

    energy=$(echo "$output" | grep "E_total" | awk '{print $3}')

    echo "E_total = $energy"
    echo ""

    echo "$name,$basis,$energy" >> $OUTPUT_FILE
}



for basis in "${BASIS_LIST[@]}"; do

    # H atom
    run_case "H_atom" "$basis" "{
      \"molecule\": {
        \"charge\": 0,
        \"atoms\": [
          { \"element\": \"H\", \"position\": [0.0, 0.0, 0.0] }
        ]
      },
      \"basis\": \"$basis\",
      \"scf\": { \"max_iter\": 50, \"tol\": 1e-6 }
    }"

    # He atom
    run_case "He_atom" "$basis" "{
      \"molecule\": {
        \"charge\": 0,
        \"atoms\": [
          { \"element\": \"He\", \"position\": [0.0, 0.0, 0.0] }
        ]
      },
      \"basis\": \"$basis\",
      \"scf\": { \"max_iter\": 50, \"tol\": 1e-6 }
    }"

    # H2 equilibrium
    run_case "H2_0.74A" "$basis" "{
      \"molecule\": {
        \"charge\": 0,
        \"atoms\": [
          { \"element\": \"H\", \"position\": [0.0, 0.0, 0.0] },
          { \"element\": \"H\", \"position\": [0.0, 0.0, 0.74] }
        ]
      },
      \"basis\": \"$basis\",
      \"scf\": { \"max_iter\": 50, \"tol\": 1e-6 }
    }"

    # H2 stretched
    run_case "H2_1.5A" "$basis" "{
      \"molecule\": {
        \"charge\": 0,
        \"atoms\": [
          { \"element\": \"H\", \"position\": [0.0, 0.0, 0.0] },
          { \"element\": \"H\", \"position\": [0.0, 0.0, 1.5] }
        ]
      },
      \"basis\": \"$basis\",
      \"scf\": { \"max_iter\": 50, \"tol\": 1e-6 }
    }"

    # H2 dissociated
    run_case "H2_3.0A" "$basis" "{
      \"molecule\": {
        \"charge\": 0,
        \"atoms\": [
          { \"element\": \"H\", \"position\": [0.0, 0.0, 0.0] },
          { \"element\": \"H\", \"position\": [0.0, 0.0, 3.0] }
        ]
      },
      \"basis\": \"$basis\",
      \"scf\": { \"max_iter\": 50, \"tol\": 1e-6 }
    }"

    # HeH+
    run_case "HeH_plus" "$basis" "{
      \"molecule\": {
        \"charge\": 1,
        \"atoms\": [
          { \"element\": \"He\", \"position\": [0.0, 0.0, 0.0] },
          { \"element\": \"H\", \"position\": [0.0, 0.0, 1.4] }
        ]
      },
      \"basis\": \"$basis\",
      \"scf\": { \"max_iter\": 50, \"tol\": 1e-6 }
    }"

done

echo "All runs completed. Results saved to $OUTPUT_FILE"