# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy application code and install dependencies
COPY poker_bot/src/setup.py .
COPY poker_bot/src/poker_bot ./poker_bot
RUN pip install --no-cache-dir -e .

# Final stage
FROM python:3.9-slim

# Add metadata labels
LABEL maintainer="Poker Bot Team" \
      version="1.0" \
      description="Poker Bot AI Assistant"

# Create non-root user
RUN useradd -m -s /bin/bash poker

WORKDIR /app

# Install runtime dependencies and tools for interactive use
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy application code and scripts
COPY poker_bot/src/poker_bot ./poker_bot
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set ownership
RUN chown -R poker:poker /app

# Switch to non-root user
USER poker

# Environment variables
ENV PHOENIX_HOST=phoenix \
    PHOENIX_PORT=6006 \
    PHOENIX_GRPC_PORT=4317 \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nc -z $PHOENIX_HOST $PHOENIX_PORT && \
        nc -z $PHOENIX_HOST $PHOENIX_GRPC_PORT && \
        python -c "import poker_bot; print('Health check passed')" || exit 1

# Set entrypoint to bash for interactive use
ENTRYPOINT ["/bin/bash"]
CMD ["/app/start.sh"]
