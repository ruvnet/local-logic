# Poker Training System

A sophisticated poker training system using AI and game theory optimization to help players improve their decision-making skills.

## Features

- **Real-time Decision Support**: Get instant analysis of poker situations
- **AI-Powered Training**: Learn from advanced AI models trained on poker strategy
- **Game Theory Optimization**: Decisions based on GTO principles
- **Position-based Strategy**: Contextual advice based on table position
- **Opponent Modeling**: Adapt to different player types and tendencies
- **Bankroll Management**: Stack size considerations and risk analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd poker/poker_bot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

```python
from poker_bot.poker_assistant import PokerAssistant

# Initialize the assistant
assistant = PokerAssistant()

# Get action recommendation
result = assistant.get_action(
    hand="AH KH",
    table_cards="QH JH 2C",
    position="BTN",
    pot_size=100,
    stack_size=1000,
    opponent_stack=1200,
    game_type="cash",
    opponent_history="aggressive"
)

print(f"Recommended Action: {result['recommended_action']}")
print(f"Reasoning: {result['reasoning']}")
```

## Training System Architecture

The training system consists of several key components:

1. **PokerAgent**: Core decision-making model using DSPy
2. **PokerTrainer**: Handles model training and optimization
3. **HyperparameterTuner**: Optimizes model parameters
4. **SafetyChecks**: Validates actions and bet sizes
5. **PokerEvaluator**: Measures model performance

### Training Process

```python
from poker_bot.trainer import PokerTrainer, TrainingConfig

# Configure training parameters
config = TrainingConfig(
    num_epochs=1000,
    batch_size=32,
    learning_rate=0.001
)

# Initialize and start training
trainer = PokerTrainer()
trainer.train(config)
```

## Customization

### Hyperparameter Tuning

```python
trainer = PokerTrainer()
trainer.tune_hyperparameters()
```

### Loading Checkpoints

```python
# List available checkpoints
checkpoints = trainer.list_checkpoints()

# Load specific checkpoint
trainer.load_checkpoint(checkpoints[0])
```

## Performance Monitoring

Monitor training progress and model performance:

```python
# Display training history
trainer.display_training_history()

# View latest metrics
evaluator = PokerEvaluator()
metrics = evaluator.evaluate(model, eval_data)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [DSPy](https://github.com/stanfordnlp/dspy)
- Uses [Treys](https://github.com/ihendley/treys) for hand evaluation
- Powered by OpenAI's GPT models
