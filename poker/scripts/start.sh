#!/bin/bash

# Initialize variables
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

# Wait for Phoenix to be ready
wait_for_phoenix() {
    echo -e "${YELLOW}Waiting for Phoenix to start...${RESET}"
    until curl -s http://phoenix:6006/health > /dev/null; do
        echo -n "."
        sleep 1
    done
    echo -e "\n${GREEN}Phoenix is ready!${RESET}"
}

# Main execution
echo -e "${GREEN}Starting Poker Bot...${RESET}"

# Wait for Phoenix
wait_for_phoenix

# Start the application
python -m poker_bot.main
#!/bin/bash
set -e

# Wait for Phoenix to be ready
echo "Waiting for Phoenix to be ready..."
until nc -z $PHOENIX_HOST $PHOENIX_PORT && nc -z $PHOENIX_HOST $PHOENIX_GRPC_PORT; do
    echo "Phoenix is not ready - sleeping"
    sleep 1
done

echo "Phoenix is ready! Starting Poker Bot..."

# Initialize Phoenix integration
python -c "from phoenix_config import init_phoenix; init_phoenix()"

# Start your application
exec python -m poker_bot.main
