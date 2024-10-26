#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start VNC server
export DISPLAY=:0
Xvfb $DISPLAY -screen 0 1024x768x16 &
fluxbox &
x11vnc -display $DISPLAY -forever -shared &

# Start the Streamlit application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
