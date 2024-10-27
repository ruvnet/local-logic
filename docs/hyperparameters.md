# Hyperparameter Tuning Documentation

## Overview
The hyperparameter tuning system optimizes the poker bot's learning parameters using grid search and performance analysis.

## HyperparameterTuner Class

### Key Features
- Grid search optimization
- Performance visualization
- Result persistence
- Analysis tools

## Tunable Parameters

### 1. Learning Rate
```python
'learning_rate': [0.0001, 0.001, 0.01]
```
- Controls training speed
- Impact on convergence
- Recommended ranges:
  - Conservative: 0.0001-0.001
  - Aggressive: 0.001-0.01
  - Experimental: 0.01-0.1

### 2. Batch Size
```python
'batch_size': [16, 32, 64, 128]
```
- Affects training stability
- Memory usage implications
- Recommended ranges:
  - Small: 16-32 (less memory, more noise)
  - Medium: 32-64 (balanced)
  - Large: 64-128 (more stable, more memory)

### 3. Temperature
```python
'temperature': [0.5, 0.7, 0.9]
```
- Controls decision randomness
- Affects exploration/exploitation
- Recommended ranges:
  - Conservative: 0.3-0.5
  - Balanced: 0.5-0.7
  - Exploratory: 0.7-0.9

### 4. Max Tokens
```python
'max_tokens': [128, 256, 512]
```
- Limits response length
- Affects model complexity
- Recommended ranges:
  - Minimal: 128
  - Standard: 256
  - Extended: 512

## Tuning Process

### 1. Basic Tuning
```python
param_grid = {
    'learning_rate': [0.001, 0.01],
    'batch_size': [32, 64],
    'temperature': [0.5, 0.7]
}
tuner = HyperparameterTuner()
results = tuner.tune_hyperparameters(param_grid)
```

### 2. Advanced Tuning
```python
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'temperature': [0.5, 0.7, 0.9],
    'max_tokens': [128, 256, 512]
}
```

### 3. Specialized Tuning
```python
# Tournament play optimization
param_grid = {
    'learning_rate': [0.0001, 0.001],
    'temperature': [0.3, 0.5],
    'batch_size': [32, 64]
}

# Cash game optimization
param_grid = {
    'learning_rate': [0.001, 0.01],
    'temperature': [0.7, 0.9],
    'batch_size': [16, 32]
}
```

## Performance Metrics

### 1. Primary Metrics
- Win rate
- Expected value
- Decision quality
- Bluff efficiency

### 2. Secondary Metrics
- Convergence speed
- Memory usage
- Training stability
- Generalization

## Result Analysis

### 1. Basic Analysis
```python
results = tuner.tune_hyperparameters(param_grid)
print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']}")
```

### 2. Detailed Analysis
```python
tuner.plot_results(results)
```

### 3. Performance Comparison
```python
for result in results['scores']:
    print(f"Parameters: {result['params']}")
    print(f"Score: {result['score']}")
```

## Optimization Strategies

### 1. Coarse-to-Fine
1. Start with wide parameter ranges
2. Identify promising regions
3. Narrow search around best results
4. Fine-tune final parameters

### 2. Resource-Aware
1. Start with smaller models/datasets
2. Validate approaches quickly
3. Scale up promising configurations
4. Optimize for production

### 3. Scenario-Specific
1. Identify key scenarios
2. Customize parameter ranges
3. Optimize for specific conditions
4. Validate across scenarios

## Best Practices

### 1. Parameter Selection
- Start with defaults
- Change one parameter at a time
- Monitor impact carefully
- Document findings

### 2. Resource Management
- Consider memory limits
- Monitor training time
- Balance accuracy vs. speed
- Plan for scaling

### 3. Validation Strategy
- Use diverse scenarios
- Cross-validate results
- Monitor overfitting
- Test edge cases

## Common Issues

### 1. Overfitting
- Symptoms:
  - High training performance
  - Poor validation results
  - Inconsistent play
- Solutions:
  - Reduce model complexity
  - Increase regularization
  - Add more diverse data

### 2. Underfitting
- Symptoms:
  - Poor training performance
  - Limited improvement
  - Simplistic play
- Solutions:
  - Increase model capacity
  - Extend training time
  - Adjust learning rate

### 3. Instability
- Symptoms:
  - Erratic performance
  - Training crashes
  - Inconsistent results
- Solutions:
  - Reduce learning rate
  - Increase batch size
  - Adjust temperature

## Advanced Topics

### 1. Custom Parameter Spaces
```python
def create_custom_grid(self, base_params, scaling_factor):
    return {
        'learning_rate': [p * scaling_factor 
                         for p in base_params['learning_rate']],
        'batch_size': [int(p * scaling_factor) 
                      for p in base_params['batch_size']]
    }
```

### 2. Adaptive Tuning
```python
def adaptive_tune(self, initial_grid, num_iterations):
    for i in range(num_iterations):
        results = self.tune_hyperparameters(initial_grid)
        initial_grid = self.refine_grid(results)
```

### 3. Multi-Objective Optimization
```python
def multi_objective_tune(self, param_grid, objectives):
    results = []
    for params in param_grid:
        scores = []
        for objective in objectives:
            score = self.evaluate_objective(params, objective)
            scores.append(score)
        results.append((params, scores))
```

## Visualization Tools

### 1. Learning Curves
```python
def plot_learning_curves(self, results):
    # Plot training progress
    # Show convergence
    # Highlight best results
```

### 2. Parameter Impact
```python
def plot_parameter_impact(self, results):
    # Show parameter effects
    # Visualize interactions
    # Identify trends
```

### 3. Performance Distribution
```python
def plot_performance_distribution(self, results):
    # Show score distribution
    # Highlight outliers
    # Compare configurations
```
