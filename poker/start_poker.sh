#!/bin/bash

# Function to check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^$1$"
}

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# Stop any existing containers
if container_exists "phoenix" || container_exists "poker_bot"; then
    echo "Stopping existing containers..."
    docker compose down > /dev/null 2>&1
fi

# Start phoenix in the background
echo "Starting phoenix..."
docker compose up phoenix -d > /dev/null 2>&1

# Wait for Phoenix health check
echo "Waiting for Phoenix to be healthy..."
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:6006/health > /dev/null; then
        echo "Phoenix is healthy!"
        break
    fi
    echo "Waiting for Phoenix health check... (attempt $attempt/$max_attempts)"
    attempt=$((attempt + 1))
    sleep 2
    
    if [ $attempt -gt $max_attempts ]; then
        echo "Timeout waiting for Phoenix to be healthy"
        exit 1
    fi
done

# Start poker_bot container in the background
echo "Starting poker_bot container..."
docker compose up poker_bot -d > /dev/null 2>&1

# Wait for poker_bot container to be running
echo "Waiting for poker_bot container to be ready..."
max_attempts=15
attempt=1
while [ $attempt -le $max_attempts ]; do
    if container_running "poker_bot"; then
        echo "Poker_bot container is ready!"
        break
    fi
    echo "Waiting for poker_bot container... (attempt $attempt/$max_attempts)"
    attempt=$((attempt + 1))
    sleep 2
    
    if [ $attempt -gt $max_attempts ]; then
        echo "Timeout waiting for poker_bot container"
        exit 1
    fi
done

# Give containers a moment to fully initialize
sleep 2

# Start an interactive session in the poker_bot container
echo "Starting poker_bot interface..."
exec docker exec -it poker_bot bash -c "cd /app/poker_bot/src/poker_bot && python main.py"
