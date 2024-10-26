# Advanced Poker AI Training System
## Game Theory Optimization & Machine Learning Framework

A sophisticated poker training system that combines game theory optimal (GTO) play with adaptive learning strategies using DSPy and advanced AI techniques. This system helps players improve their decision-making through real-time analysis and adaptive learning.

## 📊 System Architecture

![System Architecture](assets/architecture.png)

Key components:
- Decision Engine
- Training Framework 
- Analysis Tools
- Safety Checks

## 📈 Performance Metrics

![Training Progress](assets/training_progress.png)
- Win rate over time
- EV calculations
- Decision quality trends

## 🎯 System Overview

The Poker AI Training System is designed to:
- Analyze hand situations in real-time
- Provide GTO-based recommendations
- Adapt to different playing styles
- Learn from player decisions
- Optimize strategies based on position and stack depth

## 🎯 Features

### Core Capabilities
- **Advanced Decision Making Engine**
  - Real-time hand strength evaluation using Treys library
  - Position-based strategy optimization with DSPy
  - Dynamic opponent modeling through machine learning
  - Stack-size aware decisions with ICM considerations
  - Pot odds and implied odds calculations in real-time
  - Multi-street planning and hand reading
  - Range-based decision making

### Training Framework
- **Multi-Modal Learning System**
  - Supervised learning from expert gameplay
  - Reinforcement learning through self-play
  - Adversarial training against varied opponents
  - Real-time adaptation to opponent tendencies

### Analysis Tools
- **Performance Metrics**
  - Win rate tracking
  - Expected Value (EV) calculations
  - Decision quality assessment
  - Bluff efficiency analysis
  - Position-based performance metrics

### Customization Options
- **Training Parameters**
  - Learning rate adjustment
  - Batch size optimization
  - Temperature scaling
  - Early stopping criteria
  - Validation intervals

## 🛠 Advanced Customization

### Custom Hand Evaluator
```python
from poker_bot.hand_evaluator import HandEvaluator

class MyHandEvaluator(HandEvaluator):
    def calculate_hand_strength(self, cards):
        # Custom strength calculation
        strength = super().calculate_hand_strength(cards)
        # Add position-based adjustments
        return strength * self.position_multiplier()
```

### Custom Opponent Model
```python
from poker_bot.opponent_model import OpponentModel

class MyOpponentModel(OpponentModel):
    def analyze_opponent(self, history):
        tendencies = {
            'aggression': self.calculate_aggression(history),
            'bluff_frequency': self.detect_bluffs(history),
            'position_plays': self.analyze_position_play(history)
        }
        return self.generate_counter_strategy(tendencies)
```

## 🚀 Deployment

### Model Export
```python
# Export trained model
trainer.export_model('my_poker_model.pkl')
```

### API Integration
```python
from poker_bot import PokerAPI

api = PokerAPI(model_path='my_poker_model.pkl')
api.start(port=8000)
```

### Docker Deployment
```bash
# Build container
docker build -t poker-bot .

# Run API server
docker run -p 8000:8000 poker-bot
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/poker-ai-trainer.git

# Navigate to project directory
cd poker-ai-trainer

# Install dependencies
./poker/start.sh
```

### Basic Usage
```python
# Start training session
train

# Run hyperparameter tuning
tune

# Load specific checkpoint
load-checkpoint

# View training history
training-history
```

## 🎓 Training Tutorial

### 1. Understanding the Architecture

The system uses a multi-layered approach to poker decision making:
```python
class PokerAgent(dspy.Module):
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.opponent_model = OpponentModel()
        self.position_strategy = PositionStrategy()
```

### 2. Configuring Training Parameters

Optimize your training with custom configurations:
```python
config = TrainingConfig(
    num_epochs=1000,
    batch_size=32,
    learning_rate=0.001,
    validation_interval=50,
    patience=10,
    temperature=0.7
)
```

### 3. Data Generation & Augmentation

Create diverse training scenarios:
```python
def prepare_training_data():
    # Generate balanced hand distributions
    # Create multi-street scenarios
    # Vary stack sizes and positions
    return train_data, valid_data
```

## 🛠 Advanced Customization

### Game Theory Integration
- Implement Nash Equilibrium solvers
- Add range-based decision making
- Incorporate ICM modeling for tournaments

### Custom Evaluation Metrics
```python
class CustomEvaluator(PokerEvaluator):
    def __init__(self):
        self.metrics.extend([
            "fold_equity",
            "range_advantage",
            "stack_leverage"
        ])
```

### Opponent Modeling
```python
class OpponentModel:
    def analyze_opponent(self, history):
        # Pattern recognition
        # Tendency analysis
        # Exploit identification
```

## 📊 Performance Optimization

### 1. Hyperparameter Tuning
```python
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'temperature': [0.5, 0.7, 0.9]
}
```

### 2. Model Architecture Optimization
- Layer configuration
- Attention mechanisms
- Residual connections

### 3. Training Efficiency
- Batch processing
- GPU acceleration
- Distributed training

## 🔍 Monitoring & Analysis

### Training Metrics
- Loss curves
- Accuracy trends
- Validation performance
- Overfitting detection

### Performance Analysis
```python
class PerformanceAnalyzer:
    def analyze_session(self):
        # Win rate by position
        # Action distribution
        # EV analysis
        # Bluff success rate
```

## 🎮 Demo Mode

Practice and validate strategies:
```python
demo = DemoMode()
demo.simulate_game(
    opponent_level='expert',
    num_hands=100
)
```

## 🔧 Troubleshooting

Common issues and solutions:
- Training convergence problems
- Overfitting symptoms
- Memory optimization
- Performance bottlenecks

## 📚 Additional Resources

- [Poker Game Theory Fundamentals](link)
- [Advanced Training Techniques](link)
- [DSPy Documentation](link)
- [Community Forums](link)

## 🗺 Roadmap

### Upcoming Features
- Multi-table tournament support
- Real-time opponent modeling
- Advanced ICM calculations
- Hand range visualization
- Integration with popular poker platforms

### In Development
- Mobile client application
- Cloud training infrastructure
- Collaborative training framework

## 🤝 Contributing

### Issue Reporting
- Use the issue template
- Include hand histories when relevant
- Provide system information

### Pull Request Guidelines
- Follow PEP 8 style guide
- Include unit tests
- Update documentation
- Add to CHANGELOG.md

### Code Style
- Use type hints
- Document complex algorithms
- Follow project structure
- Include docstrings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- DSPy team for the core framework
- Poker theory contributors
- Community feedback and testing

---

For detailed documentation, visit our [Wiki](wiki-link).
For support, join our [Discord Community](discord-link).
# Poker Training System

A sophisticated poker training system using AI and game theory optimization to help players improve their decision-making skills.

## 🎯 Features

### Core Capabilities
- **Advanced Decision Engine**
  - Real-time hand strength evaluation using Treys library
  - Position-based strategy optimization with DSPy
  - Dynamic opponent modeling through machine learning
  - Stack-size aware decisions with ICM considerations
  - Pot odds and implied odds calculations in real-time
  - Multi-street planning and hand reading
  - Range-based decision making
  - Monte Carlo simulation for equity calculation
  - Bluff frequency optimization
  - Fold equity calculations

### Training Framework
- **Multi-Modal Learning System**
  - LLM-based initial training using GPT-4-mini
  - Efficient response caching to minimize API calls
  - Local model fine-tuning for faster predictions
  - Example-based learning with similarity matching
  - Comprehensive metrics tracking and analysis
  - Early stopping with configurable patience
  - Automatic checkpointing of best models
  - Training history visualization
  - Custom evaluation metrics

### Analysis Tools
- **Performance Metrics**
  - Win rate tracking and progression
  - Expected Value (EV) calculations
  - Decision quality assessment
  - Bluff efficiency analysis
  - Position-based performance metrics
  - Stack-to-pot ratio considerations
  - Opponent tendency analysis
  - Hand strength evaluation
  - Monte Carlo simulation results
  - Training convergence metrics

### Safety Features
- **Risk Management**
  - Action validation system
  - Bet size verification
  - Stack size protection
  - Bankroll management rules
  - Tilt detection and prevention
  - Emergency stop conditions
  - Input validation and sanitization
  - Error handling and recovery
  - Session limits and warnings
  - Automatic safety adjustments

### Interactive Features
- **User Interface**
  - Colored terminal output for better readability
  - Progress bars for long-running operations
  - Interactive command system
  - Detailed help and documentation
  - Training progress visualization
  - Real-time feedback and advice
  - Situation-specific recommendations
  - Position and stack size guidance
  - Opponent-specific adjustments
  - Demo mode with different difficulty levels

### Data Management
- **Storage and Analysis**
  - JSON-based result storage
  - Training history tracking
  - Checkpoint management
  - Performance analysis reports
  - Metric progression tracking
  - Configuration persistence
  - Session summaries
  - Analysis exports
  - Data visualization capabilities
  - Backup and recovery systems

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

## 🧠 Learning System Architecture

### How the AI Learning Works

The poker training system uses a sophisticated multi-stage learning approach:

1. **Initial Learning with LLM**
   - Uses GPT-4-mini as the foundation model
   - Processes poker situations into structured inputs
   - Generates initial decisions and reasoning
   - Learns from expert-level poker knowledge

2. **Efficient Learning Process**
   ```python
   # Example of how the system processes a poker situation
   game_state = {
       'hand': "AH KH",  # Ace-King of Hearts
       'position': "BTN",  # Button position
       'pot_size': 100,
       # ... other game details
   }
   ```

3. **Smart Caching System**
   - Stores previously seen situations
   - Reduces redundant LLM calls
   - Speeds up training process
   - Builds a knowledge base of decisions

4. **Local Model Development**
   - Creates a smaller, faster model from cached responses
   - Uses pattern matching for similar situations
   - Provides quick responses for familiar scenarios
   - Falls back to LLM for unique situations

5. **Continuous Improvement**
   ```python
   # How the system learns from each hand
   for each_hand:
       # Get AI decision
       decision = model.predict(game_state)
       
       # Store result
       store_in_cache(game_state, decision)
       
       # Update local model
       update_patterns(game_state, decision)
   ```

### Learning Features

1. **Pattern Recognition**
   - Identifies similar poker situations
   - Learns from past decisions
   - Adapts to playing styles
   - Builds decision patterns

2. **Real-time Adaptation**
   - Adjusts to opponent tendencies
   - Updates strategy based on results
   - Fine-tunes decision making
   - Learns from mistakes

3. **Performance Metrics**
   ```python
   metrics = {
       'win_rate': 0.65,      # 65% success rate
       'ev_calculation': 1.2,  # Positive expected value
       'decision_quality': 0.8 # High confidence
   }
   ```

### Training Process Visualization

```
Input → LLM Processing → Decision Cache → Local Model → Refined Output
   ↑                                                        |
   └────────────── Feedback Loop ───────────────────────────┘
```

### Key Benefits

1. **Efficient Learning**
   - Reduces API calls over time
   - Builds a specialized poker knowledge base
   - Improves response speed
   - Maintains decision quality

2. **Adaptive Strategy**
   - Learns from each game situation
   - Builds pattern recognition
   - Develops situational awareness
   - Improves decision consistency

3. **Safety and Validation**
   - Verifies decisions against poker rules
   - Prevents costly mistakes
   - Ensures bankroll protection
   - Maintains strategic discipline

### Technical Implementation

```python
class PokerAgent:
    def process_hand(self, game_state):
        # Check cache first
        if in_cache(game_state):
            return get_cached_decision(game_state)
            
        # Use LLM if needed
        if needs_llm_processing(game_state):
            decision = query_llm(game_state)
            store_in_cache(game_state, decision)
            return decision
            
        # Use local model for similar situations
        return local_model.predict(game_state)
```

### Learning Metrics Tracked

- Win Rate Progression
- Decision Quality Trends
- Pattern Recognition Success
- Adaptation Speed
- Strategy Consistency
- Error Rate Reduction

This learning system combines the power of large language models with efficient caching and pattern recognition to create a poker AI that continuously improves while maintaining quick response times and strategic accuracy.

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
