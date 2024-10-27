#!/bin/bash

# Add color variables
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

# Track container IDs
PHOENIX_CONTAINER=""
POKER_CONTAINER=""

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${RESET}"
    
    # Stop Phoenix container if running
    if [ ! -z "$PHOENIX_CONTAINER" ]; then
        echo "Stopping Phoenix container..."
        docker stop "$PHOENIX_CONTAINER" >/dev/null 2>&1
        docker rm "$PHOENIX_CONTAINER" >/dev/null 2>&1
    fi
    
    # Stop Poker container if running
    if [ ! -z "$POKER_CONTAINER" ]; then
        echo "Stopping Poker container..."
        docker stop "$POKER_CONTAINER" >/dev/null 2>&1
        docker rm "$POKER_CONTAINER" >/dev/null 2>&1
    fi
    
    echo -e "${GREEN}Cleanup complete${RESET}"
    exit 0
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

display_ai_initialization() {
    echo -e "\n🤖 Initializing Poker AI System..."
    sleep 0.5
    for step in "Loading neural network..." "Calibrating decision matrices..." "Ready!"; do
        echo -ne "⚡ $step\r"
        sleep 0.3
        echo -e "✅ $step"
    done
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${RESET}"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Docker daemon is not running. Please start Docker first.${RESET}"
        exit 1
    }
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
    echo -e "${CYAN}Checking requirements...${RESET}"
    
    # Only create venv if it doesn't exist
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${RESET}"
        python -m venv venv
    fi
    
    source venv/bin/activate
    
    echo -e "${YELLOW}Installing/upgrading pip and core dependencies...${RESET}"
    pip install --upgrade pip wheel setuptools poetry >/dev/null 2>&1
    
    echo -e "${YELLOW}Installing required packages...${RESET}"
    pip install --upgrade \
        phoenix-ai \
        opentelemetry-api \
        opentelemetry-sdk \
        opentelemetry-instrumentation \
        openinference-instrumentation-dspy \
        openinference-instrumentation-litellm \
        dspy-ai \
        numpy \
        pandas \
        treys \
        pytest \
        scikit-learn \
        colorama \
        matplotlib \
        seaborn \
        tqdm \
        python-dotenv >/dev/null 2>&1
    
    # Install project in development mode
    cd poker_bot/src
    pip install -e . >/dev/null 2>&1
    cd ../..
    
    # Set PYTHONPATH
    export PYTHONPATH="${PWD}/poker_bot/src:${PYTHONPATH}"
}

start_phoenix() {
    echo -e "${YELLOW}Starting Phoenix container...${RESET}"
    
    # Stop any existing Phoenix container
    docker ps -q --filter "name=phoenix" | xargs -r docker stop >/dev/null 2>&1
    docker ps -aq --filter "name=phoenix" | xargs -r docker rm >/dev/null 2>&1
    
    # Start new Phoenix container
    PHOENIX_CONTAINER=$(docker run -d \
        --name phoenix \
        -p 6006:6006 \
        -p 4317:4317 \
        arizephoenix/phoenix:latest)
    
    echo -e "${YELLOW}Waiting for Phoenix to start...${RESET}"
    until curl -s http://localhost:6006 >/dev/null; do
        echo -n "."
        sleep 1
    done
    echo -e "\n${GREEN}Phoenix is ready!${RESET}"
}

build_poker() {
    echo -e "${YELLOW}Building Poker Bot container...${RESET}"
    docker-compose build --no-cache poker_bot
}

start_poker() {
    echo -e "${YELLOW}Starting Poker Bot container...${RESET}"
    
    # Stop any existing Poker container
    docker ps -q --filter "name=poker_bot" | xargs -r docker stop >/dev/null 2>&1
    docker ps -aq --filter "name=poker_bot" | xargs -r docker rm >/dev/null 2>&1
    
    # Start new Poker container
    POKER_CONTAINER=$(docker-compose up -d poker_bot)
    
    # Follow logs
    docker-compose logs -f poker_bot
}

# Main execution
echo -e "${GREEN}🎲 Starting Poker Bot System...${RESET}"

# Check Docker installation and daemon
check_docker

# Build and start containers
build_poker
start_phoenix
start_poker

# Wait for user input (the containers will be cleaned up when script exits)
echo -e "\n${YELLOW}Press Ctrl+C to stop the application${RESET}"
wait
