#!/bin/bash
# Quick start script for running the experiment

echo "=========================================="
echo "Linguistically-Aware EWC Experiment"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run experiment
echo ""
echo "Starting experiment..."
echo "This will train 4 different methods:"
echo "  1. Naive fine-tuning"
echo "  2. Standard EWC"
echo "  3. Linguistic EWC (our proposal)"
echo "  4. Random EWC (sanity check)"
echo ""

python train.py

# Generate visualizations
echo ""
echo "Generating visualizations..."
python visualize.py

# Show results
echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to ./results/"
echo "Check the following files:"
echo "  - comparison.json (numerical results)"
echo "  - forgetting_comparison.png"
echo "  - final_accuracy.png"
echo "  - learning_curves.png"
echo ""
