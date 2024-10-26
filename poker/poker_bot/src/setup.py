from setuptools import setup, find_packages

setup(
    name="poker_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",  # For numerical operations
        "pandas",  # For data manipulation
        "treys",  # For poker hand evaluation
        "pytest",  # For testing
        "dspy",   # For DSPy functionality
        "scikit-learn",  # For machine learning functionality
        "colorama",  # For colored terminal output
        "matplotlib",  # For plotting and visualization
        "openai",  # For OpenAI API integration
        "seaborn"  # For statistical visualization
    ],
    python_requires=">=3.8",
)
