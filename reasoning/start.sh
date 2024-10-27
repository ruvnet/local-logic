#!/bin/bash

display_ai_initialization() {
    echo -e "\n🤖 Initializing Reasoning System..."
    sleep 0.5
    for step in "Initializing reasoning modules..." "Setting up logic compilers..." "Ready!"; do
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
    pip install --upgrade pip wheel setuptools >/dev/null 2>&1
    
    # Install packages in one command to reduce overhead
    pip install numpy pandas pytest "dspy-ai[all]" scikit-learn colorama matplotlib seaborn openai >/dev/null 2>&1
    
    # Set PYTHONPATH to include the reasoning source directory
    export PYTHONPATH="${PYTHONPATH}:${PWD}/reasoning/reasoning/src"
    
    # Install the package in development mode
    if [ -f "reasoning/reasoning/src/setup.py" ]; then
        cd reasoning/reasoning/src
        pip install -e . >/dev/null 2>&1
        cd ../../..
    fi
}

# Create necessary directories if they don't exist
create_directories() {
    mkdir -p reasoning/reasoning/src/reasoning_bot
    mkdir -p reasoning/reasoning/src/reasoning_bot/models
    mkdir -p reasoning/reasoning/src/reasoning_bot/data
    mkdir -p reasoning/reasoning/src/reasoning_bot/config
}

# Main execution
echo "🧠 Starting Reasoning System..."
create_directories
check_requirements
display_ai_initialization

# Run the main application with proper error handling
if python reasoning/reasoning/src/reasoning_bot/main.py; then
    echo "✅ Reasoning System completed successfully"
else
    echo "❌ Reasoning System encountered an error"
    exit 1
fi
