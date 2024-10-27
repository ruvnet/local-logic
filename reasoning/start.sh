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

display_reasoning_settings() {
    echo -e "\n${CYAN}============================================================${RESET}"
    echo -e "${YELLOW}⚙️ REASONING SYSTEM SETTINGS${RESET}"
    echo -e "${CYAN}============================================================${RESET}\n"
    
    echo -e "${YELLOW}1. Inference Settings${RESET}"
    echo -e "• Inference depth: 3 levels"
    echo -e "• Pattern matching threshold: 0.85"
    echo -e "• Logic chain length: Dynamic"
    
    echo -e "\n${YELLOW}2. Template Parameters${RESET}"
    echo -e "• Template complexity: Medium"
    echo -e "• Pattern recognition sensitivity: 0.75"
    echo -e "• Update frequency: Real-time"
    
    echo -e "\n${YELLOW}3. System Configuration${RESET}"
    echo -e "• Processing mode: Optimized"
    echo -e "• Memory allocation: Adaptive"
    echo -e "• Cache strategy: Dynamic"
    
    echo -e "\n${YELLOW}4. Runtime Options${RESET}"
    echo -e "• Parallel processing: Enabled"
    echo -e "• Debug mode: Disabled"
    echo -e "• Performance logging: Active"
    
    echo -e "\n${CYAN}============================================================${RESET}"
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
    
    # Install packages for reasoning system
    pip install numpy pandas pytest scikit-learn colorama matplotlib \
        seaborn openai networkx spacy nltk transformers torch \
        tensorflow sympy >/dev/null 2>&1
    
    # Install additional NLP and reasoning packages
    python -m spacy download en_core_web_sm >/dev/null 2>&1
    python -m nltk.downloader punkt averaged_perceptron_tagger wordnet >/dev/null 2>&1
    
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
display_reasoning_settings
display_main_menu

# Run the main application with proper error handling
if python reasoning/src/reasoning_bot/main.py; then
    echo -e "${CYAN}✅ Local Logic System completed successfully${RESET}"
else
    echo -e "${YELLOW}❌ Local Logic System encountered an error${RESET}"
    exit 1
fi
