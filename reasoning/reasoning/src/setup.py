from setuptools import setup, find_packages

setup(
    name="reasoning_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",  # For numerical operations
        "pandas",  # For data manipulation
        "scikit-learn",  # For machine learning
        "colorama",  # For colored terminal output
        "matplotlib",  # For visualization
        "seaborn",  # For statistical plots
        "openai",  # For OpenAI integration
        "dspy-ai[all]",  # For DSPy integration
    ],
    python_requires=">=3.8",
)
