from setuptools import setup, find_packages

setup(
    name="poker_bot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",  # For numerical operations
        "pandas",  # For data manipulation
        "treys",  # For poker hand evaluation
        "pytest",  # For testing
        "dspy",   # For DSPy functionality
        "scikit-learn",  # For machine learning functionality
    ],
)
