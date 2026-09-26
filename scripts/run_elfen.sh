#!/bin/bash
# Make sure the script runs in a virtual environment with the required dependencies installed.
# Make excutable: chmod +x run_elfen.sh
# Usage: ./run_elfen.sh

# # Update these variables as needed!!!
# INPUT_FILE=""
# OUTPUT_DIR=""
INPUT_FILE="${1:?usage: ./run_elfen.sh INPUT.csv OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: ./run_elfen.sh INPUT.csv OUTPUT_DIR}"
CREATE_COMBINED=true     # Create combined features file???? (true/false)
NORMALIZE_METHOD="token" # Normalization method: none, token, standard, ratio, rescale

# # Python script namee
# PYTHON_SCRIPT="elfen_extractor.py"
PYTHON_SCRIPT="src/features/elfen_extractor.py"

echo "=== Elfen Feature Extraction ==="
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Create combined file: $CREATE_COMBINED"
echo "Normalization method: $NORMALIZE_METHOD"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    echo "Please check the INPUT_FILE variable in this script."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# the python command
PYTHON_CMD="python $PYTHON_SCRIPT --input \"$INPUT_FILE\" --output \"$OUTPUT_DIR\" --normalize $NORMALIZE_METHOD"

# incured if combined features to be created or not
if [ "$CREATE_COMBINED" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --combined"
fi

echo "Executing: $PYTHON_CMD"
echo ""

# Run the Python script
eval $PYTHON_CMD

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Feature extraction completed successfully!"
else
    echo ""
    echo "Error: Feature extraction failed."
    exit 1
fi