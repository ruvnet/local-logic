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
    
    # Remove old venv if it exists
    if [ -d "venv" ]; then
        echo "Removing old virtual environment..."
        rm -rf venv
    fi
    
    # Create fresh venv
    echo "Creating new virtual environment..."
    python -m venv venv
    
    source venv/bin/activate
    
    # Install core dependencies first
    echo "Installing dependencies..."
    pip install --upgrade pip wheel setuptools >/dev/null 2>&1
    
    # Install packages in one command to reduce overhead
    pip install numpy pandas pytest "dspy-ai[all]" scikit-learn colorama matplotlib seaborn openai >/dev/null 2>&1
    
    # Set PYTHONPATH to include the reasoning source directory
    export PYTHONPATH="${PWD}/reasoning/src:${PYTHONPATH}"
    
    # Install the package in development mode
    cd reasoning/src
    pip install -e . >/dev/null 2>&1
    cd ../..
}

# Main execution
echo "🧠 Starting Reasoning System..."
check_requirements
display_ai_initialization

# Run the main application with proper error handling
if python reasoning/src/reasoning_bot/main.py; then
    echo "✅ Reasoning System completed successfully"
else
    echo "❌ Reasoning System encountered an error"
    exit 1
fi
