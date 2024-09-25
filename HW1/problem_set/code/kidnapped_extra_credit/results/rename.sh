#!/bin/bash

# Set the starting number for renaming
counter=0

# Loop through all the PNG files in numerical order and rename them sequentially
for file in *.png; do
    # Format the counter to 4 digits (e.g., 0000, 0001, etc.)
    new_filename=$(printf "%04d.png" $counter)
    
    # Rename the file
    mv "$file" "$new_filename"
    
    # Increment the counter
    counter=$((counter + 1))
done

