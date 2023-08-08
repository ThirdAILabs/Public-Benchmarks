#!/bin/bash

# Run the script for a specific configuration (number of nodes and embedding dimensions)
run_script() {
  NUM_NODES=$1
  EMBEDDING_DIMENSION=$2
  MODEL_SIZE=$3

  # Display the current configuration
  echo "Running with ${NUM_NODES} nodes, embedding dimension ${EMBEDDING_DIMENSION}, and ${MODEL_SIZE} model size" 

  # Execute the script with the current configuration
  python3 train.py --num_nodes "${NUM_NODES}" --embedding_dimension "${EMBEDDING_DIMENSION}" 2>&1 | tee -a log_file.txt

  # Clearing previous training results to free disk space for next iteration
  # If you want to use the trained model, you can save it somewhere else or find it on s3 storage specified in Ray Runtime.
  rm -rf ~/ray_results train* test*
}

# Run the script for specific configurations
run_script 48 256 "25M"
run_script 48 384 "37.5M"
run_script 48 512 "50M"

run_script 24 256 "25M"
run_script 24 384 "37.5M"
run_script 24 512 "50M"

run_script 12 256 "25M"
run_script 12 384 "37.5M"
run_script 12 512 "50M"
