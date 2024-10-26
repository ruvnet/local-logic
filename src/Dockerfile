FROM ubuntu:20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    nodejs \
    npm \
    x11vnc \
    xvfb \
    fluxbox \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Install Aider
RUN pip3 install aider

# Set up VNC environment
ENV DISPLAY=:0
ENV RESOLUTION=1920x1080x24

# Create workspace directory
WORKDIR /app

# Copy application files
COPY . /app/

# Expose ports for Streamlit and VNC
EXPOSE 8501 5900

# Start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
