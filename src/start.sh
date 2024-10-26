#!/bin/bash

# Start VNC server
Xvfb :0 -screen 0 $RESOLUTION &
fluxbox &
x11vnc -display :0 -forever -usepw -create &

# Start Streamlit application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
#!/bin/bash

# Start Xvfb
Xvfb :0 -screen 0 $RESOLUTION &
sleep 1

# Start window manager
fluxbox &

# Start VNC server
x11vnc -display :0 -forever -usepw -create &

# Start the Streamlit application
streamlit run app.py
