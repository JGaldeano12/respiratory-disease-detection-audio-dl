#!/bin/bash

# This script generates the dataset for training and testing the model.
# It creates a directory structure and populates it with the necessary files.
echo "Generating dataset..."

# Run the Python script to generate the dataset
python -m src.data.generate_dataset --seed 20260119

echo "Dataset generation complete!"