#!/bin/bash

# Function to check if a Python package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
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
        pip install -e .
        cd ../..
    fi
    
    # Check for required packages
    required_packages=("numpy" "pandas" "treys" "pytest" "dspy" "sklearn")
    
    for package in "${required_packages[@]}"; do
        if ! check_package $package; then
            echo "Installing missing package: $package"
            pip install $package
        fi
    done
}

# Main execution
echo "Starting Poker Bot..."

# Check and install requirements
check_requirements

# Change to the poker_bot directory
cd poker_bot/src/poker_bot

# Run the main application
echo "Launching Poker Bot..."
PYTHONPATH="/workspaces/agentic-desktop/poker/poker_bot/src:$PYTHONPATH" python main.py

# Deactivate virtual environment when done
deactivate
