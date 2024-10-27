#!/bin/bash

# Add color variables
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

display_ai_initialization() {
    echo -e "\n🧠 Initializing Local Logic System..."
    sleep 0.5
    for step in "Loading reasoning patterns..." "Compiling logic templates..." "Initializing inference engine..." "Activating decision modules..." "System ready"; do
        echo -ne "⚡ $step\r"
        sleep 0.3
        echo -e "✅ $step"
    done
}

display_main_menu() {
    echo -e "\n${CYAN}============================================================${RESET}"
    echo -e "${YELLOW}🧠 LOCAL LOGIC REASONING SYSTEM${RESET}"
    echo -e "${CYAN}============================================================${RESET}\n"
    
    echo -e "${YELLOW}📚 MAIN MENU:${RESET}\n"
    
    echo -e "${YELLOW}1. Pattern Management${RESET}"
    echo -e "   🔄 compile  - Compile new reasoning patterns"
    echo -e "   📊 optimize - Tune logic templates"
    echo -e "   📈 analyze  - View pattern metrics"
    
    echo -e "\n${YELLOW}2. Reasoning Modes${RESET}"
    echo -e "   💭 reason   - Interactive reasoning session"
    echo -e "   🤖 simulate - Test with synthetic problems"
    echo -e "   🔍 review   - Analyze reasoning history"
    
    echo -e "\n${YELLOW}3. Knowledge Base${RESET}"
    echo -e "   💾 save     - Save reasoning templates"
    echo -e "   📂 load     - Load template library"
    echo -e "   📋 list     - Show available templates"
    
    echo -e "\n${YELLOW}4. System${RESET}"
    echo -e "   ⚙️  config   - Configure system parameters"
    echo -e "   ❓ help     - Show detailed help"
    echo -e "   🚪 exit     - Exit system"
    
    echo -e "\n${CYAN}============================================================${RESET}"
    echo -e "${YELLOW}Enter command:${RESET} "
}

display_help() {
    echo -e "\n${CYAN}============================================================${RESET}"
    echo -e "${YELLOW}📚 LOCAL LOGIC SYSTEM GUIDE${RESET}"
    echo -e "${CYAN}============================================================${RESET}\n"
    
    echo -e "${YELLOW}Pattern Compilation:${RESET}"
    echo -e "• Extract reasoning patterns from input data"
    echo -e "• Generate optimized logic templates"
    echo -e "• Create efficient decision rules"
    
    echo -e "\n${YELLOW}Template Management:${RESET}"
    echo -e "• Store and organize reasoning templates"
    echo -e "• Version control for logic patterns"
    echo -e "• Template library management"
    
    echo -e "\n${YELLOW}Inference Engine:${RESET}"
    echo -e "• Real-time reasoning processing"
    echo -e "• Pattern matching and application"
    echo -e "• Decision optimization"
    
    echo -e "\n${YELLOW}System Commands:${RESET}"
    echo -e "• compile  - Create new reasoning patterns"
    echo -e "• optimize - Tune existing templates"
    echo -e "• reason   - Start interactive session"
    echo -e "• simulate - Run test scenarios"
    echo -e "• exit     - Close the system"
    
    echo -e "\n${CYAN}============================================================${RESET}"
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
echo -e "${YELLOW}🧠 Starting Local Logic System...${RESET}"
check_requirements
display_ai_initialization
display_main_menu

# Run the main application with proper error handling
if python reasoning/src/reasoning_bot/main.py; then
    echo -e "${CYAN}✅ Local Logic System completed successfully${RESET}"
else
    echo -e "${YELLOW}❌ Local Logic System encountered an error${RESET}"
    exit 1
fi
