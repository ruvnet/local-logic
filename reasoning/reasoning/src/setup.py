from setuptools import setup, find_packages

setup(
    name="reasoning_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",  # For numerical operations
        "pandas",  # For data manipulation
        "dspy-ai[all]",  # For DSPy functionality
        "scikit-learn",  # For machine learning
        "colorama",  # For colored terminal output
        "matplotlib",  # For visualization
        "seaborn",  # For statistical plots
        "openai",  # For OpenAI integration
        "tqdm"  # For progress bars
    ],
    python_requires=">=3.8",
)
