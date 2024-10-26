#!/bin/bash

# Exit on error
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages if not already installed
pip install --upgrade pip
pip install open-interpreter uvicorn streamlit

# Run the interpreter
interpreter

# Deactivate virtual environment when done
deactivate
