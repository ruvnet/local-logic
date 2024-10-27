#!/bin/bash
set -e

# Enhanced debugging
echo "Starting health check..."

# Check Phoenix HTTP endpoint
echo "Checking Phoenix HTTP endpoint..."
if ! curl -f -s http://${PHOENIX_HOST}:${PHOENIX_PORT}/health; then
    echo "Phoenix HTTP health check failed"
    exit 1
fi

# Check Phoenix GRPC port
echo "Checking Phoenix GRPC port..."
if ! nc -z ${PHOENIX_HOST} ${PHOENIX_GRPC_PORT}; then
    echo "Phoenix GRPC port check failed"
    exit 1
fi

# Check application health
echo "Checking application health..."
if ! python -c "import poker_bot; from phoenix_config import get_tracer; print('Health check passed')"; then
    echo "Application health check failed"
    exit 1
fi

echo "All health checks passed"
exit 0
#!/bin/bash
set -e

# Check Phoenix connectivity
nc -z $PHOENIX_HOST $PHOENIX_PORT || exit 1
nc -z $PHOENIX_HOST $PHOENIX_GRPC_PORT || exit 1

# Check application health
python -c "import poker_bot; from phoenix_config import get_tracer; print('Health check passed')" || exit 1
