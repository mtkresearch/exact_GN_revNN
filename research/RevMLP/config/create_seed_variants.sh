#!/bin/bash

# Check if a directory path is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_yaml_files_directory>"
    exit 1
fi

# Loop over all YAML configuration files in the specified directory
for config_file in "$yaml_dir"*.yaml; do
    # Check if the file exists
    if [[ -f "$config_file" ]]; then
        # Call the experiment with the configuration file
        echo "Generating seed variants for config: $config_file"
        for i in 43 44; do
            new_yaml="${config_file%.yaml}-seed$i.yaml"
            cp "$config_file" "$new_yaml"
            sed -i "s/seed: 42/seed: $i/" $new_yaml
        done
    else
        echo "Configuration file '$config_file' does not exist."
    fi
done
