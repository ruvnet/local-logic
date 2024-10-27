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

display_settings() {
    echo -e "\n${CYAN}============================================================${RESET}"
    echo -e "${YELLOW}⚙️  SYSTEM SETTINGS & CONFIGURATION${RESET}"
    echo -e "${CYAN}============================================================${RESET}\n"
    
    echo -e "${YELLOW}1. Inference Engine Configuration${RESET}"
    echo -e "${CYAN}Depth Control:${RESET}"
    echo -e "• Current depth: 3 levels"
    echo -e "• Adjustable range: 1-5 levels"
    echo -e "• Affects reasoning complexity"
    
    echo -e "\n${CYAN}Pattern Matching:${RESET}"
    echo -e "• Threshold: 0.85"
    echo -e "• Precision mode: High"
    echo -e "• Match validation: Enabled"
    
    echo -e "\n${CYAN}Logic Processing:${RESET}"
    echo -e "• Chain length: Dynamic"
    echo -e "• Auto-optimization: Active"
    echo -e "• Chain validation: Enabled"
    
    echo -e "\n${YELLOW}2. Template System${RESET}"
    echo -e "${CYAN}Complexity Settings:${RESET}"
    echo -e "• Current level: Medium"
    echo -e "• Available levels: Basic, Medium, Advanced"
    echo -e "• Auto-scaling: Enabled"
    
    echo -e "\n${CYAN}Pattern Recognition:${RESET}"
    echo -e "• Sensitivity: 0.75"
    echo -e "• Confidence threshold: 0.80"
    echo -e "• False positive prevention: Active"
    
    echo -e "\n${CYAN}Update Management:${RESET}"
    echo -e "• Frequency: Real-time"
    echo -e "• Pattern refresh: Automatic"
    echo -e "• Version control: Enabled"
    
    echo -e "\n${YELLOW}3. System Performance${RESET}"
    echo -e "${CYAN}Processing:${RESET}"
    echo -e "• Mode: Optimized"
    echo -e "• Thread allocation: Dynamic"
    echo -e "• Resource balancing: Active"
    
    echo -e "\n${CYAN}Memory Management:${RESET}"
    echo -e "• Allocation: Adaptive"
    echo -e "• Garbage collection: Automatic"
    echo -e "• Memory limits: Dynamic"
    
    echo -e "\n${CYAN}Cache System:${RESET}"
    echo -e "• Strategy: Dynamic"
    echo -e "• Invalidation: Smart"
    echo -e "• Prefetching: Enabled"
    
    echo -e "\n${YELLOW}4. Runtime Configuration${RESET}"
    echo -e "${CYAN}Processing Options:${RESET}"
    echo -e "• Parallel processing: Enabled"
    echo -e "• Thread count: Auto (1-8)"
    echo -e "• Load balancing: Active"
    
    echo -e "\n${CYAN}Debug Settings:${RESET}"
    echo -e "• Debug mode: Disabled"
    echo -e "• Trace level: 0"
    echo -e "• Error reporting: Standard"
    
    echo -e "\n${CYAN}Performance Monitoring:${RESET}"
    echo -e "• Logging: Active"
    echo -e "• Metrics collection: Enabled"
    echo -e "• Analysis tools: Available"
    
    echo -e "\n${YELLOW}Configuration Commands:${RESET}"
    echo -e "• set-depth <1-5>     - Adjust inference depth"
    echo -e "• set-threshold <val>  - Modify pattern matching threshold"
    echo -e "• toggle-debug        - Enable/disable debug mode"
    echo -e "• set-mode <mode>     - Change processing mode"
    
    echo -e "\n${CYAN}============================================================${RESET}"
    echo -e "${YELLOW}Press Enter to return to main menu${RESET}"
    read
}

display_main_menu() {
    echo -e "\n${CYAN}============================================================${RESET}"
    echo -e "${YELLOW}🧠 LOCAL LOGIC REASONING SYSTEM${RESET}"
    echo -e "${CYAN}============================================================${RESET}\n"
    
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
    echo -e "   ⚙️  settings - View/modify system settings"
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
    
    # Only create venv if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating new virtual environment..."
        python -m venv venv
    fi
    
    source venv/bin/activate

    # Function to check if a package is installed
    package_installed() {
        python -c "import $1" 2>/dev/null
        return $?
    }

    # Function to check pip package version
    pip_package_installed() {
        pip show $1 >/dev/null 2>&1
        return $?
    }

    # Only upgrade pip if version is old
    if ! pip --version | grep -q "pip 2[3-9]"; then
        echo "Upgrading pip..."
        pip install --upgrade pip >/dev/null 2>&1
    fi

    # List of required packages with their import names
    declare -A packages=(
        ["numpy"]="numpy"
        ["pandas"]="pandas"
        ["pytest"]="pytest"
        ["scikit-learn"]="sklearn"
        ["colorama"]="colorama"
        ["matplotlib"]="matplotlib"
        ["seaborn"]="seaborn"
        ["openai"]="openai"
    )

    # Check and install missing packages
    missing_packages=()
    for pkg in "${!packages[@]}"; do
        if ! package_installed "${packages[$pkg]}"; then
            missing_packages+=("$pkg")
        fi
    done

    # Install missing packages if any
    if [ ${#missing_packages[@]} -ne 0 ]; then
        echo "Installing missing packages: ${missing_packages[*]}"
        pip install "${missing_packages[@]}" >/dev/null 2>&1
    fi

    # Set PYTHONPATH
    export PYTHONPATH="${PWD}/reasoning/src:${PYTHONPATH}"

    # Only install in development mode if not already installed
    if ! pip_package_installed "reasoning-bot"; then
        cd reasoning/src
        pip install -e . >/dev/null 2>&1
        cd ../..
    fi
}

# Main execution
echo -e "${YELLOW}🧠 Starting Local Logic System...${RESET}"
check_requirements
display_ai_initialization
display_main_menu

# Handle commands
while true; do
    read -r command
    case $command in
        "settings")
            display_settings
            display_main_menu
            ;;
        "exit")
            echo -e "${YELLOW}Exiting system...${RESET}"
            exit 0
            ;;
        *)
            # Handle other commands...
            ;;
    esac
done

# Run the main application with proper error handling
if python reasoning/src/reasoning_bot/main.py; then
    echo -e "${CYAN}✅ Local Logic System completed successfully${RESET}"
else
    echo -e "${YELLOW}❌ Local Logic System encountered an error${RESET}"
    exit 1
fi
