#!/bin/bash

# Check if a directory path is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_yaml_files_directory>"
    exit 1
fi

# Assign the directory path to a variable
yaml_dir=$1

# Check if the directory exists
if [ ! -d "$yaml_dir" ]; then
    echo "The directory '$yaml_dir' does not exist."
    exit 1
fi

# Loop over all YAML configuration files in the specified directory
for config_file in "$yaml_dir"/experiment_*.yaml; do
    # Check if the file exists
    if [[ -f "$config_file" ]]; then
        # Call the experiment with the configuration file
        echo "Running experiment with config: $config_file"
        python3 train.py --config="$config_file"
    else
        echo "Configuration file '$config_file' does not exist."
    fi
done

echo "All experiments have been run."