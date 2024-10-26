import os
import dspy

# Configure DSPy with your preferred LLM
gpt4 = dspy.OpenAI(model="gpt-4")
dspy.configure(lm=gpt4)

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY:
    dspy.configure(openai_api_key=OPENAI_API_KEY)
else:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
