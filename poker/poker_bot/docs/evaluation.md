# Evaluation System Documentation

## Overview
The poker bot evaluation system provides comprehensive metrics and analysis tools to assess model performance across multiple dimensions of poker strategy.

## Components

### PokerEvaluator
Main evaluation class inheriting from dspy.Evaluate.

#### Core Metrics:

1. **Win Rate**
   - Primary performance indicator
   - Calculation: Monte Carlo simulation results
   - Range: 0.0-1.0
   - Target: >0.55 for profitable play

2. **Expected Value (EV)**
   - Measures decision profitability
   - Calculation: (Win% * Pot) - (Loss% * Bet)
   - Range: Negative to positive
   - Target: Positive EV over time

3. **Decision Quality**
   - Measures strategic correctness
   - Factors:
     - Hand strength
     - Position
     - Stack-to-pot ratio
     - Opponent tendencies
   - Range: 0.0-1.0
   - Target: >0.7

4. **Bluff Efficiency**
   - Measures deception success
   - Calculation: (Successful bluffs / Total bluffs)
   - Range: 0.0-1.0
   - Target: >0.4

## Evaluation Methods

### 1. Basic Evaluation
```python
evaluator = PokerEvaluator()
metrics = evaluator.evaluate(model, valid_data)
```

### 2. Detailed Analysis
```python
analysis = trainer._generate_analysis(history, final_metrics)
```

### 3. Performance Visualization
```python
trainer.plot_results(results)
```

## Metrics Details

### Win Rate Calculation
```python
def calculate_win_rate(self, prediction, game_state):
    # Monte Carlo simulation
    wins = 0
    for _ in range(self.config.num_simulations):
        # Simulate opponent hands
        # Evaluate outcomes
        # Track wins
    return wins / self.config.num_simulations
```

### Expected Value Calculation
```python
def calculate_ev(self, action, pot_size, stack_size, hand_strength):
    if action == 'fold':
        return 0
    elif action == 'call':
        return pot_size * (hand_strength - 0.5)
    elif action == 'raise':
        fold_equity = 0.3
        return (pot_size * fold_equity) + 
               (pot_size * 2 * (1 - fold_equity) * 
                (hand_strength - 0.5))
```

## Performance Benchmarks

### 1. Basic Proficiency
- Win Rate: >0.52
- EV: >0
- Decision Quality: >0.6
- Bluff Efficiency: >0.3

### 2. Intermediate Level
- Win Rate: >0.54
- EV: >5bb/100
- Decision Quality: >0.7
- Bluff Efficiency: >0.35

### 3. Advanced Level
- Win Rate: >0.56
- EV: >10bb/100
- Decision Quality: >0.8
- Bluff Efficiency: >0.4

## Evaluation Scenarios

### 1. Standard Evaluation
- Regular game situations
- Various positions
- Different stack sizes
- Multiple opponent types

### 2. Stress Testing
- Extreme stack sizes
- Complex board textures
- Multi-way pots
- Tournament bubbles

### 3. Specialized Testing
- Heads-up situations
- Short stack play
- Deep stack play
- ICM situations

## Analysis Tools

### 1. Training History
```python
trainer.display_training_history()
```
Shows:
- Epoch-by-epoch metrics
- Convergence trends
- Performance peaks
- Problem areas

### 2. Performance Analysis
```python
analysis = trainer._generate_analysis(history, final_metrics)
```
Provides:
- Convergence rate
- Stage-by-stage performance
- Metric progression
- Key improvements

### 3. Visualization
```python
trainer.plot_results(results)
```
Displays:
- Learning curves
- Metric correlations
- Performance distribution
- Trend analysis

## Best Practices

### 1. Regular Evaluation
- Evaluate every N epochs
- Track multiple metrics
- Save detailed logs
- Compare to benchmarks

### 2. Comprehensive Testing
- Use diverse scenarios
- Test edge cases
- Validate against known strategies
- Cross-validate results

### 3. Performance Monitoring
- Track long-term trends
- Identify weaknesses
- Monitor for overfitting
- Compare across versions

## Troubleshooting

### 1. Poor Performance
- Check training data quality
- Verify evaluation metrics
- Analyze specific scenarios
- Review hyperparameters

### 2. Inconsistent Results
- Increase num_simulations
- Verify random seed
- Check for data leakage
- Validate evaluation code

### 3. Metric Discrepancies
- Verify calculation methods
- Check for edge cases
- Validate assumptions
- Cross-reference results

## Advanced Analysis

### 1. Decision Analysis
```python
def analyze_decision(self, game_state, prediction):
    # Detailed analysis of single decisions
    # Compare to optimal play
    # Calculate decision error
    # Provide improvement suggestions
```

### 2. Strategy Profiling
```python
def profile_strategy(self, model, test_suite):
    # Analyze playing style
    # Identify patterns
    # Compare to known strategies
    # Generate recommendations
```

### 3. Performance Attribution
```python
def attribute_performance(self, metrics_history):
    # Break down performance factors
    # Identify key contributors
    # Analyze failure modes
    # Suggest improvements
```
