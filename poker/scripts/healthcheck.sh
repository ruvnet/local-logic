#!/bin/bash

# Check Phoenix health
curl -f http://phoenix:6006/health || exit 1

# Check if poker bot process is running
pgrep -f "python -m poker_bot.main" || exit 1

exit 0
