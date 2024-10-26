#!/bin/bash

# Setup display for X11
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! xhost >& /dev/null; then
        export DISPLAY=:0
        sudo Xvfb :0 -screen 0 1024x768x24 > /dev/null 2>&1 &
        sleep 2
    fi
fi

# Activate virtual environment
source interpreter-env/bin/activate

# Run interpreter in OS mode
interpreter --os
