#!/bin/bash

# This script generates the dataset for training and testing the model.
# It creates a directory structure and populates it with the necessary files.
echo "Generating dataset..."

# Run the Python script to generate the dataset
python src/data/generate_dataset.py \
    --output_dir data/dataset \
    --config configs/dataset_config.yaml

echo "Dataset generation complete!"