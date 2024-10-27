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
    echo "Checking requirements..."
    
    # Only create venv if it doesn't exist
    if [ ! -d "venv" ]; then
        python -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install core dependencies first
    pip install --upgrade pip wheel setuptools poetry >/dev/null 2>&1
    
    # Install project using poetry
    cd poker_bot
    poetry install >/dev/null 2>&1
    cd ..
    
    # Set PYTHONPATH
    export PYTHONPATH="${PWD}/poker_bot/src:${PYTHONPATH}"
}

# Main execution
echo "🎲 Starting Poker Bot..."
check_requirements
display_ai_initialization

# Run the main application
python poker_bot/src/poker_bot/main.py
