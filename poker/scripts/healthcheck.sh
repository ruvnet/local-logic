#!/bin/bash

# Check Phoenix health
curl -f http://phoenix:6006/health || exit 1

# Check if poker bot process is running
pgrep -f "python -m poker_bot.main" || exit 1

exit 0
#!/bin/bash
set -e

# Check Phoenix connectivity
nc -z $PHOENIX_HOST $PHOENIX_PORT || exit 1
nc -z $PHOENIX_HOST $PHOENIX_GRPC_PORT || exit 1

# Check application health
python -c "import poker_bot; from phoenix_config import get_tracer; print('Health check passed')" || exit 1
