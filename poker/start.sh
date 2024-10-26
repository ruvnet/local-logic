#!/bin/bash

# Function to check requirements in background
check_requirements_background() {
    check_requirements > /dev/null 2>&1
    echo "REQUIREMENTS_DONE" > /tmp/requirements_status
}

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
    
    # Upgrade pip first
    pip install --upgrade pip >/dev/null 2>&1
    
    # Install dspy-ai with all dependencies
    pip install "dspy-ai[all]" >/dev/null 2>&1
    
    # Additional step to ensure dspy is available
    pip install --no-deps dspy >/dev/null 2>&1
    
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
    required_packages=(
        "numpy"
        "pandas"
        "treys"
        "pytest"
        "dspy-ai[all]"
        "dspy"
        "scikit-learn"
        "colorama"
        "matplotlib"
        "seaborn"
        "openai"
    )
    
    for package in "${required_packages[@]}"; do
        if ! check_package $package; then
            echo "Installing missing package: $package"
            pip install -q $package >/dev/null 2>&1
        fi
    done
}

# Main execution
echo "🎲 Starting Poker Bot..."

# Start requirements check in background
echo "🔍 Checking system requirements..."
check_requirements_background &

# Display AI initialization while requirements are checking
display_ai_initialization

# Wait for requirements check to complete
while [ ! -f /tmp/requirements_status ]; do
    sleep 0.1
done
rm /tmp/requirements_status

# Change to the poker_bot directory
cd poker_bot/src/poker_bot

# Run the main application
echo "Launching Poker Bot..."
PYTHONPATH="/workspaces/agentic-desktop/poker/poker_bot/src:$PYTHONPATH" python main.py

# Deactivate virtual environment when done
deactivate
