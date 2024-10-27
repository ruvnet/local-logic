#!/bin/bash

display_ai_initialization() {
    echo -e "\n🤖 Initializing Poker AI System..."
    sleep 0.5
    for step in "Loading neural network..." "Calibrating decision matrices..." "Ready!"; do
        echo -ne "⚡ $step\r"
        sleep 0.3
        echo -e "✅ $step"
    done
}

check_requirements() {
    echo -e "${CYAN}Checking requirements...${RESET}"
    
    # Only create venv if it doesn't exist
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${RESET}"
        python -m venv venv
    fi
    
    source venv/bin/activate

    # Install pip and core dependencies
    echo -e "${YELLOW}Installing/upgrading pip and core dependencies...${RESET}"
    python -m pip install --upgrade pip wheel setuptools

    # Install phoenix and its dependencies first
    echo -e "${YELLOW}Installing Phoenix and dependencies...${RESET}"
    pip install \
        phoenix-ai==0.2.0 \
        opentelemetry-api \
        opentelemetry-sdk \
        opentelemetry-instrumentation \
        opentelemetry-instrumentation-requests \
        openinference-instrumentation-dspy \
        openinference-instrumentation-litellm

    # Install other required packages
    echo -e "${YELLOW}Installing other required packages...${RESET}"
    pip install \
        dspy-ai \
        numpy \
        pandas \
        treys \
        pytest \
        scikit-learn \
        colorama \
        matplotlib \
        seaborn \
        tqdm \
        python-dotenv \
        requests

    # Install the project in development mode
    echo -e "${YELLOW}Installing project in development mode...${RESET}"
    pip install -e poker_bot/src/

    # Set PYTHONPATH
    export PYTHONPATH="${PWD}/poker_bot/src:${PYTHONPATH}"
}

# Main execution
echo "🎲 Starting Poker Bot..."
check_requirements
display_ai_initialization

# Run the main application
python poker_bot/src/poker_bot/main.py
