#!/bin/bash

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
    
    # Install wheel and setuptools first
    pip install --upgrade wheel setuptools >/dev/null 2>&1
    
    # Install DSPy and its dependencies explicitly
    pip install "dspy-ai[all]" >/dev/null 2>&1
    pip install dspy >/dev/null 2>&1
    
    # Set PYTHONPATH to include the src directory
    export PYTHONPATH="${PYTHONPATH}:/workspaces/agentic-desktop/poker/poker_bot/src"
    
    # Install the poker_bot package in editable mode
    cd /workspaces/agentic-desktop/poker/poker_bot/src
    pip install -e . >/dev/null 2>&1
    cd -
    
    # Install other required packages
    required_packages=(
        "numpy"
        "pandas"
        "treys"
        "pytest"
        "scikit-learn"
        "colorama"
        "matplotlib"
        "seaborn"
        "openai"
    )
    
    for package in "${required_packages[@]}"; do
        if ! pip show $package >/dev/null 2>&1; then
            echo "Installing missing package: $package"
            pip install $package >/dev/null 2>&1
        fi
    done
}

# Main execution
echo "🎲 Starting Poker Bot..."

# Clean up any existing virtual environment
rm -rf venv/

# Run requirements check
check_requirements

# Display AI initialization
display_ai_initialization

# Set PYTHONPATH again before running the main script
export PYTHONPATH="/workspaces/agentic-desktop/poker/poker_bot/src:$PYTHONPATH"

# Run the main application
echo "Launching Poker Bot..."
python /workspaces/agentic-desktop/poker/poker_bot/src/poker_bot/main.py

# Deactivate virtual environment when done
deactivate
