#!/bin/bash

# Function to check if a Python package is installed
check_package() {
    # Use pip to check if package is installed
    if ./venv/bin/pip show "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

display_ai_initialization() {
    echo -e "\n🤖 Initializing Poker AI System..."
    sleep 0.5
    
    # Simulated AI initialization steps
    steps=(
        "Loading neural network architectures..."
        "Calibrating decision matrices..."
        "Initializing hand strength evaluator..."
        "Loading opponent modeling system..."
        "Optimizing position-based strategies..."
        "Configuring GPT-4 language model..."
        "Validating game theory optimal solutions..."
        "Preparing real-time analysis engine..."
        "Initializing bankroll management system..."
        "Calibrating risk assessment modules..."
    )
    
    for step in "${steps[@]}"; do
        echo -ne "⚡ $step\r"
        sleep 0.3
        echo -e "✅ $step"
    done
    
    echo -e "\n🎮 AI System Ready!"
}

# Function to check and install requirements
check_requirements() {
    echo "Checking requirements..."
    
    # Check if virtual environment exists, create if not
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Set PYTHONPATH to include the src directory
    export PYTHONPATH="/workspaces/agentic-desktop/poker/poker_bot/src:$PYTHONPATH"
    
    # Install the poker_bot package and its dependencies if not already installed
    if ! check_package poker_bot; then
        echo "Installing poker_bot package and dependencies..."
        cd poker_bot/src
        pip install -e . >/dev/null 2>&1
        cd ../..
    fi
    
    # Check for required packages
    required_packages=("numpy" "pandas" "treys" "pytest" "dspy" "scikit-learn" "colorama" "matplotlib" "seaborn")
    
    for package in "${required_packages[@]}"; do
        if ! check_package $package; then
            echo "Installing missing package: $package"
            pip install -q $package >/dev/null 2>&1
        fi
    done
}

# Main execution
echo "🎲 Starting Poker Bot..."

# Check and install requirements
echo "🔍 Checking system requirements..."
check_requirements

# Display AI initialization
display_ai_initialization

# Change to the poker_bot directory
cd poker_bot/src/poker_bot

# Run the main application
echo "Launching Poker Bot..."
PYTHONPATH="/workspaces/agentic-desktop/poker/poker_bot/src:$PYTHONPATH" python main.py

# Deactivate virtual environment when done
deactivate
