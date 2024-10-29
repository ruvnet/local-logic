# Poker Bot Docker Setup

## Quick Start

### Automated Startup
```bash
# Make the startup script executable
chmod +x start_poker.sh

# Start the system
./start_poker.sh
```

### Manual Connection
If you need to manually connect to the poker_bot:
```bash
# Start the services
docker compose up -d

# Wait for services to be ready, then connect to poker_bot
docker exec -it poker_bot bash

# Inside the container, navigate to the poker_bot directory and start the interface
cd /app/poker_bot/src/poker_bot
python main.py
```

## Stopping the System
To stop all services:
```bash
docker compose down
```

For more details about the poker_bot system itself, see [poker_bot/README.md](poker_bot/README.md).
