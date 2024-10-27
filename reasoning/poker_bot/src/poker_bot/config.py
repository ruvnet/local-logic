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
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model configuration
DEFAULT_MODEL = 'gpt-4'
MAX_TOKENS = 256
TEMPERATURE = 0.7

# Training configuration
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 1000

# Game configuration
STARTING_STACK = 1000
MIN_BET = 20
