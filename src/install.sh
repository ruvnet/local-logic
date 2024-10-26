#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package list and install required packages
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install flask aider-chat

# Set up environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Run the Flask application
flask run --host=0.0.0.0 --port=5000
