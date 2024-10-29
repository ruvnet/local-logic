from setuptools import setup, find_namespace_packages

setup(
    name="poker_bot",
    version="0.1.0",
    packages=find_namespace_packages(include=['poker_bot*']),
    install_requires=[
        "numpy",  # For numerical operations
        "pandas",  # For data manipulation
        "treys",  # For poker hand evaluation
        "pytest",  # For testing
        "dspy-ai[all]",   # For DSPy functionality with all dependencies
        "scikit-learn",  # For machine learning functionality
        "colorama",  # For colored terminal output
        "matplotlib",  # For plotting and visualization
        "openai",  # For OpenAI API integration
        "seaborn",  # For statistical visualization
        "tqdm",  # For progress bars
        "opentelemetry-api",  # OpenTelemetry support
        "opentelemetry-sdk",
        "opentelemetry-instrumentation",
        "openinference-instrumentation-dspy",
        "openinference-instrumentation-litellm"
    ],
    python_requires=">=3.8",
)
