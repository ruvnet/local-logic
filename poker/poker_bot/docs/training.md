# Training System Documentation

## Overview
The poker bot training system uses a combination of deep learning and Monte Carlo simulation approaches to learn optimal poker strategies. The system is built on the DSPy framework and uses GPT-4-mini as the base language model.

## Components

### PokerTrainer
The main training controller class that orchestrates the entire training process.

#### Key Methods:
- `train()`: Full training pipeline
- `train_one_epoch()`: Single epoch training
- `prepare_training_data()`: Data preparation
- `display_training_history()`: Training progress visualization

### Training Configuration
The `TrainingConfig` class manages all training parameters:

```python
config = TrainingConfig(
    num_epochs=100,        # Total training epochs
    batch_size=32,         # Samples per batch
    learning_rate=0.001,   # Learning rate
    validation_interval=5,  # Epochs between validations
    patience=10,           # Early stopping patience
    min_delta=0.001,       # Minimum improvement for early stopping
    temperature=0.7,       # Model temperature
    max_tokens=256,        # Maximum tokens per response
    shuffle_data=True,     # Data shuffling
    num_simulations=500    # Monte Carlo simulations
)
```

#### Parameter Details:

1. **num_epochs** (default: 100)
   - Range: 50-500
   - Impact: Training duration and convergence
   - Higher values: Better convergence but longer training
   - Lower values: Faster training but potential underfitting

2. **batch_size** (default: 32)
   - Range: 16-128
   - Impact: Memory usage and training stability
   - Larger batches: More stable gradients, more memory
   - Smaller batches: Less memory, more noise in training

3. **learning_rate** (default: 0.001)
   - Range: 0.0001-0.1
   - Impact: Training convergence speed
   - Higher values: Faster learning but potential instability
   - Lower values: More stable but slower learning

4. **temperature** (default: 0.7)
   - Range: 0.1-1.0
   - Impact: Model creativity vs. consistency
   - Higher values: More diverse decisions
   - Lower values: More conservative play

5. **num_simulations** (default: 500)
   - Range: 100-10000
   - Impact: Equity calculation accuracy
   - Higher values: More accurate but slower
   - Lower values: Faster but less accurate

## Training Process

### 1. Initialization
```python
trainer = PokerTrainer()
config = TrainingConfig(...)
```

### 2. Data Preparation
The system generates diverse poker scenarios including:
- Different positions (BTN, CO, MP, UTG, BB, SB)
- Various stack sizes (1000-5000)
- Different pot sizes (100-500)
- Multiple hand types (premium, speculative, etc.)

### 3. Training Loop
Each epoch includes:
1. Batch processing
2. Model updates
3. Validation checks
4. Metric calculation
5. Checkpoint saving

### 4. Monitoring
Training progress can be monitored through:
- Real-time metrics
- Training history
- Validation results
- Performance plots

## Optimization Strategies

### 1. Basic Optimization
```python
# Start with default parameters
config = TrainingConfig()
trainer.train(config)

# Monitor initial performance
trainer.display_training_history()
```

### 2. Advanced Optimization
```python
# Increase simulation accuracy
config = TrainingConfig(
    num_simulations=1000,
    temperature=0.5,
    patience=15
)

# Use hyperparameter tuning
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'temperature': [0.5, 0.7, 0.9]
}
tuner = HyperparameterTuner()
results = tuner.tune_hyperparameters(param_grid)
```

### 3. Specialized Training
For specific game types:
```python
config = TrainingConfig(
    temperature=0.4,  # More conservative for tournaments
    num_simulations=2000,  # Higher accuracy
    patience=20  # Allow more time to converge
)
```

## Best Practices

1. **Start Conservative**
   - Use default parameters
   - Monitor initial performance
   - Adjust gradually

2. **Systematic Tuning**
   - Test one parameter at a time
   - Use wide ranges initially
   - Narrow down based on results

3. **Resource Management**
   - Balance num_simulations with available compute
   - Monitor memory usage with batch_size
   - Consider training time vs. accuracy tradeoffs

4. **Validation Strategy**
   - Use diverse validation scenarios
   - Monitor multiple metrics
   - Save checkpoints regularly

## Common Issues and Solutions

1. **Overfitting**
   - Decrease num_epochs
   - Increase early stopping patience
   - Add more diverse training data

2. **Unstable Training**
   - Reduce learning_rate
   - Increase batch_size
   - Decrease temperature

3. **Slow Convergence**
   - Increase learning_rate
   - Adjust batch_size
   - Increase patience

4. **Memory Issues**
   - Reduce batch_size
   - Decrease num_simulations
   - Optimize data loading
