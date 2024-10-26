#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the Streamlit application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
