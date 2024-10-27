# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install core dependencies first
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    treys \
    pytest \
    dspy-ai[all] \
    scikit-learn \
    colorama \
    matplotlib \
    openai \
    seaborn \
    tqdm \
    python-dotenv

# Install OpenTelemetry dependencies
RUN pip install --no-cache-dir \
    opentelemetry-exporter-otlp \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation \
    opentelemetry-instrumentation-requests \
    openinference-instrumentation-dspy \
    openinference-instrumentation-litellm \
    arize-phoenix-otel

# Copy scripts and application code
COPY scripts/ /app/scripts/
COPY phoenix_config.py /app/phoenix_config.py
COPY poker_bot/src/poker_bot ./poker_bot
RUN chmod +x /app/scripts/*.sh

# Final stage
FROM python:3.10-slim

# Add metadata labels
LABEL maintainer="Poker Bot Team" \
      version="1.0" \
      description="Poker Bot AI Assistant"

# Create non-root user
RUN useradd -m -s /bin/bash poker

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/poker_bot ./poker_bot

# Copy scripts and config
COPY scripts/ /app/scripts/
COPY phoenix_config.py /app/phoenix_config.py
RUN chmod +x /app/scripts/*.sh

# Set ownership
RUN chown -R poker:poker /app

# Switch to non-root user
USER poker

# Environment variables
ENV PHOENIX_HOST=phoenix \
    PHOENIX_PORT=6006 \
    PHOENIX_GRPC_PORT=4317 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nc -z $PHOENIX_HOST $PHOENIX_PORT && \
        nc -z $PHOENIX_HOST $PHOENIX_GRPC_PORT && \
        python -c "import poker_bot; print('Health check passed')" || exit 1

# Set command
CMD ["/app/scripts/start.sh"]
