#!/bin/bash

# Read from stdin
input=$(cat)

# Define output file
output_file="/tmp/hook.out"

# Append input followed by two newlines
echo -e "${input}\n\n" >> "$output_file"

# Exit with status 0
exit 0
