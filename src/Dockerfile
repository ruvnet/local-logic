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
    xauth \
    x11-xserver-utils \
    xdg-utils \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip

# Create workspace directory
WORKDIR /app

# Copy requirements before installing
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install aider

# Set up VNC environment
ENV DISPLAY=:0
ENV RESOLUTION=1920x1080x24

# Copy application files
COPY . /app/

# Expose ports for Streamlit and VNC
EXPOSE 8501 5900

# Start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && \
    ln -s /usr/bin/python3 /usr/bin/python

# Ensure executables are in PATH
ENV PATH="/usr/bin:/usr/local/bin:${PATH}"

# Create VNC password
RUN mkdir ~/.vnc && x11vnc -storepasswd somepassword ~/.vnc/passwd

ENTRYPOINT ["/app/start.sh"]
