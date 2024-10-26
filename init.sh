#!/bin/bash

# install.sh
# This script sets up the Open Interpreter environment with VNC, Streamlit, and Aider.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting installation..."

# Create project directory
mkdir -p open_interpreter_env
cd open_interpreter_env

echo "Creating Dockerfile..."
# Create Dockerfile
cat > Dockerfile <<EOF
FROM ubuntu:20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    git \\
    nodejs \\
    npm \\
    x11vnc \\
    xvfb \\
    fluxbox \\
    wget \\
    curl \\
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
EOF

echo "Creating requirements.txt..."
# Create requirements.txt
cat > requirements.txt <<EOF
streamlit>=1.24.0
open-interpreter>=0.1.4
python-dotenv>=0.19.0
websockets>=10.0
numpy>=1.21.0
pandas>=1.3.0
EOF

echo "Creating start.sh..."
# Create start.sh
cat > start.sh <<'EOF'
#!/bin/bash

# Start VNC server
Xvfb :0 -screen 0 $RESOLUTION &
fluxbox &
x11vnc -display :0 -forever -usepw -create &

# Start Streamlit application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
EOF

# Make start.sh executable
chmod +x start.sh

echo "Creating app.py..."
# Create app.py
cat > app.py <<'EOF'
import streamlit as st
from interpreter import interpreter
import subprocess
import os

class InterpreterUI:
    def __init__(self):
        self.interpreter = interpreter
        self.interpreter.auto_run = True
        self.setup_streamlit()

    def setup_streamlit(self):
        st.set_page_config(
            page_title="Open Interpreter Enhanced Environment",
            layout="wide"
        )
        st.title("Open Interpreter Enhanced Environment")

        # VNC Display
        st.components.v1.iframe(
            src="http://localhost:5900",
            height=600,
            scrolling=True
        )

        # Command Input
        self.command = st.text_area(
            "Enter your command or code request:",
            height=100,
            key="command_input"
        )

        # Execute Button
        if st.button("Execute"):
            self.execute_command()

        # History Display
        if 'history' not in st.session_state:
            st.session_state.history = []

        self.display_history()

    def execute_command(self):
        if self.command:
            with st.spinner('Processing...'):
                # Add command to history
                st.session_state.history.append({
                    'command': self.command,
                    'status': 'Running'
                })

                try:
                    # Check if the command is a code request
                    if self.command.lower().startswith("code:"):
                        code_request = self.command[5:].strip()
                        response = self.generate_code(code_request)
                    else:
                        # Execute command using Open Interpreter
                        response = self.interpreter.chat(self.command)

                    # Update history with response
                    st.session_state.history[-1]['status'] = 'Complete'
                    st.session_state.history[-1]['response'] = response

                except Exception as e:
                    st.session_state.history[-1]['status'] = 'Failed'
                    st.session_state.history[-1]['response'] = str(e)

    def generate_code(self, code_request):
        """Generate code using Aider based on natural language description."""
        # Use Aider to generate code
        from aider import Aider
        aider = Aider()
        code = aider.generate_code(code_request)
        # Save code to a file (optional)
        with open('generated_code.py', 'w') as f:
            f.write(code)
        return code

    def display_history(self):
        st.subheader("Command History")
        for item in st.session_state.history[::-1]:
            with st.expander(f"{item['command']} ({item['status']})", expanded=False):
                if 'response' in item:
                    st.code(item['response'], language='python')

if __name__ == "__main__":
    app = InterpreterUI()
EOF

echo "Creating docker-compose.yml..."
# Create docker-compose.yml
cat > docker-compose.yml <<EOF
version: '3'
services:
  interpreter:
    build: .
    ports:
      - "8501:8501"  # Streamlit
      - "5900:5900"  # VNC
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
    volumes:
      - ./:/app
    restart: unless-stopped
EOF

echo "Creating .env file..."
# Create .env file with placeholder for OpenAI API Key
cat > .env <<EOF
# Replace 'your_api_key_here' with your actual OpenAI API key
OPENAI_API_KEY=your_api_key_here
EOF

echo "Installation complete."
echo "Please replace 'your_api_key_here' in the .env file with your actual OpenAI API key."
echo "To build and run the Docker container, execute: docker-compose up --build"
