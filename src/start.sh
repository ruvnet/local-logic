#!/bin/bash

# Set display
export DISPLAY=:0

# Start Xvfb and wait for it to be ready
Xvfb :0 -screen 0 $RESOLUTION &
sleep 2

# Start window manager
fluxbox &
sleep 1

# Start VNC server
x11vnc -display :0 -forever -usepw -create &
sleep 1

# Start the Streamlit application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
