#!/bin/bash
# gpu-discovery.sh - identifies GPUs on each worker node

# Get list of available GPUs using nvidia-smi
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | sort)

# Format output for Spark: {"name": "gpu", "addresses": ["0", "1", "2", "3"]}
if [ -n "$GPUS" ]; then
  # Convert newline-separated list to comma-separated
  GPU_ADDRESSES=$(echo $GPUS | tr '\n' ',' | sed 's/,$//')
  echo "{\"name\": \"gpu\", \"addresses\": [\"$GPU_ADDRESSES\"]}"
else
  # No GPUs found
  echo "{\"name\": \"gpu\", \"addresses\": []}"
fi
