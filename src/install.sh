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
pip install streamlit open-interpreter aider-chat

# Create a streamlit config directory if it doesn't exist
mkdir -p ~/.streamlit

# Create or update the Streamlit config file
cat > ~/.streamlit/config.toml << EOL
[server]
port = 8501
address = "0.0.0.0"
headless = true
EOL

# Set execute permissions for the script
chmod +x app.py

# Start the Streamlit application
streamlit run app.py
