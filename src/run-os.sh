#!/bin/bash

# Installation script for Open Interpreter OS mode
# Save as: setup-interpreter.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Setting up Open Interpreter with OS mode...${NC}"

# Install Python 3.11 from source
install_python_from_source() {
    echo -e "${BLUE}Installing Python 3.11 from source...${NC}"
    
    # Install build dependencies
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
        libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev \
        wget libbz2-dev

    # Download and extract Python 3.11
    cd /tmp
    wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
    tar -xf Python-3.11.7.tgz
    cd Python-3.11.7

    # Configure and install
    ./configure --enable-optimizations --prefix=/usr/local
    make -j $(nproc)
    sudo make altinstall

    # Install pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.11

    # Create symbolic links
    sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1
}

# Install system dependencies
install_system_deps() {
    echo -e "${BLUE}Installing system dependencies...${NC}"
    sudo apt-get update
    sudo apt-get install -y \
        python3-tk \
        python3-dev \
        scrot \
        x11-apps \
        xauth \
        xvfb \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxtst6 \
        libxi6 \
        python3-venv \
        python3-pip
}

# Install Python if needed
if ! command -v python3.11 &> /dev/null; then
    install_python_from_source
fi

# Install system dependencies
install_system_deps

# Create and activate virtual environment
echo -e "${BLUE}Setting up virtual environment...${NC}"
python3.11 -m venv interpreter-env
source interpreter-env/bin/activate

# Upgrade pip and install dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
python -m pip install --upgrade pip
python -m pip install wheel setuptools
python -m pip install uvicorn
python -m pip install "open-interpreter[os]"

# Create launcher script
cat > run-interpreter.sh << 'EOL'
#!/bin/bash

# Setup display for X11
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! xhost >& /dev/null; then
        export DISPLAY=:0
        sudo Xvfb :0 -screen 0 1024x768x24 > /dev/null 2>&1 &
        sleep 2
    fi
fi

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "${SCRIPT_DIR}/interpreter-env/bin/activate"

# Install any missing dependencies
python -m pip install --quiet uvicorn "open-interpreter[os]"

# Run interpreter in OS mode
interpreter --os
EOL

# Make launcher executable
chmod +x run-interpreter.sh

# Create uninstall script
cat > uninstall-interpreter.sh << 'EOL'
#!/bin/bash
deactivate 2>/dev/null
rm -rf interpreter-env
rm run-interpreter.sh
rm uninstall-interpreter.sh
EOL

chmod +x uninstall-interpreter.sh

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}To run Open Interpreter in OS mode:${NC}"
echo -e "${BLUE}./run-interpreter.sh${NC}"
echo -e "${GREEN}To uninstall:${NC}"
echo -e "${BLUE}./uninstall-interpreter.sh${NC}"

# Activate the environment for immediate use
source interpreter-env/bin/activate