#!/bin/bash

# If there aren't any command line arguments or if the argument count is not equal to 1, display an error message and exit
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <activation-key>"
    exit 1
fi

ACTIVATION_KEY=$1

# Run the script for a specific configuration (number of nodes and embedding dimensions)
run_script() {
  NUM_NODES=$1
  EMBEDDING_DIMENSION=$2
  MODEL_SIZE=$3

  # Display the current configuration
  echo "Running with ${NUM_NODES} nodes, embedding dimension ${EMBEDDING_DIMENSION}, and ${MODEL_SIZE} model size" 

  # Execute the script with the current configuration
  python3 train.py --num_nodes "${NUM_NODES}" --embedding_dimension "${EMBEDDING_DIMENSION}" --activation_key "${ACTIVATION_KEY}" 

  # Clearing previous train/test files to free disk space for next iteration
  rm -rf ~/ray_results train_file* test_file*
}

# Run the script for specific configurations
run_script 48 256 "25M" 2>&1 | tee -a log_file.txt
run_script 48 384 "37.5M" 2>&1 | tee -a log_file.txt
run_script 48 512 "50M" 2>&1 | tee -a log_file.txt

run_script 24 256 "25M" 2>&1 | tee -a log_file.txt
run_script 24 384 "37.5M" 2>&1 | tee -a log_file.txt
run_script 24 512 "50M" 2>&1 | tee -a log_file.txt

run_script 12 256 "25M" 2>&1 | tee -a log_file.txt
run_script 12 384 "37.5M" 2>&1 | tee -a log_file.txt
run_script 12 512 "50M" 2>&1 | tee -a log_file.txt
