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
    pip install numpy pandas treys pytest "dspy-ai[all]" scikit-learn colorama matplotlib seaborn openai >/dev/null 2>&1
    
    # Set PYTHONPATH
    export PYTHONPATH="${PYTHONPATH}:/workspaces/agentic-desktop/reasoning/reasoning/src"
    
    # Install the package in development mode
    cd /workspaces/agentic-desktop/reasoning/reasoning/src
    pip install -e . >/dev/null 2>&1
}

# Main execution
echo "🧠 Starting Reasoning System..."
check_requirements
display_ai_initialization

# Run the main application
python /workspaces/agentic-desktop/reasoning/reasoning/src/reasoning_bot/main.py
