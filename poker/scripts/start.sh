#!/bin/bash

# Add color variables
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

# Enhanced debugging function
debug_info() {
    echo -e "${CYAN}[DEBUG] $1${RESET}"
}

# Enhanced error function
error_info() {
    echo -e "${RED}[ERROR] $1${RESET}"
}

# Check if Phoenix container is running
check_phoenix_container() {
    debug_info "Checking Phoenix container status..."
    if ! docker ps | grep -q "phoenix"; then
        error_info "Phoenix container is not running"
        debug_info "Starting Phoenix container..."
        docker run -d \
            --name phoenix \
            --network poker-net \
            -p 6006:6006 \
            -p 4317:4317 \
            arizephoenix/phoenix:latest
        
        # Wait for container to start
        sleep 5
    else
        debug_info "Phoenix container is running"
    fi
}

# Check Phoenix connectivity with timeout
check_phoenix() {
    local timeout=60  # Increased timeout
    local start_time=$(date +%s)
    
    debug_info "Checking Phoenix connectivity..."
    debug_info "Phoenix Host: $PHOENIX_HOST"
    debug_info "Phoenix Port: $PHOENIX_PORT"
    debug_info "Phoenix GRPC Port: $PHOENIX_GRPC_PORT"
    
    # First ensure Phoenix container is running
    check_phoenix_container
    
    while true; do
        # Check HTTP port
        if curl -s "http://${PHOENIX_HOST}:${PHOENIX_PORT}/health" >/dev/null; then
            debug_info "Phoenix HTTP port ($PHOENIX_PORT) is reachable"
            http_ok=true
        else
            error_info "Phoenix HTTP port ($PHOENIX_PORT) is not reachable"
            http_ok=false
        fi
        
        # Check GRPC port
        if nc -z -w 2 $PHOENIX_HOST $PHOENIX_GRPC_PORT 2>/dev/null; then
            debug_info "Phoenix GRPC port ($PHOENIX_GRPC_PORT) is reachable"
            grpc_ok=true
        else
            error_info "Phoenix GRPC port ($PHOENIX_GRPC_PORT) is not reachable"
            grpc_ok=false
        fi
        
        # Both ports are ready
        if [ "$http_ok" = true ] && [ "$grpc_ok" = true ]; then
            echo -e "${GREEN}Phoenix is ready!${RESET}"
            return 0
        fi
        
        # Check timeout
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -gt $timeout ]; then
            error_info "Timeout waiting for Phoenix after ${timeout} seconds"
            error_info "Container network info:"
            docker network inspect poker-net
            error_info "Phoenix container logs:"
            docker logs phoenix
            error_info "DNS resolution:"
            nslookup $PHOENIX_HOST || true
            error_info "Network connectivity:"
            netstat -tulpn
            return 1
        fi
        
        echo -n "."
        sleep 2
    done
}

# Main execution
echo -e "${GREEN}Starting Poker Bot...${RESET}"

# Wait for Phoenix with enhanced debugging
if ! check_phoenix; then
    error_info "Failed to connect to Phoenix. Exiting."
    exit 1
fi

debug_info "Initializing Phoenix integration..."
if ! python -c "from phoenix_config import init_phoenix; init_phoenix()"; then
    error_info "Failed to initialize Phoenix integration"
    exit 1
fi

debug_info "Starting main application..."
exec python -m poker_bot.main