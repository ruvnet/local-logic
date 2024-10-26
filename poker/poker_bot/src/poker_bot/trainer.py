import os
import json
import numpy as np
from colorama import Fore, Style
from poker_bot.poker_agent import PokerAgent
from poker_bot.hyperparameter_tuner import HyperparameterTuner
import dspy
from dspy.evaluate import Evaluate
from typing import List, Dict, Tuple

class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self, **kwargs):
        # Default optimal values
        self.num_epochs = kwargs.get('num_epochs', 1000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.validation_interval = kwargs.get('validation_interval', 50)
        self.patience = kwargs.get('patience', 10)
        self.min_delta = kwargs.get('min_delta', 0.001)
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 256)

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class PokerEvaluator(dspy.Evaluate):
    """Evaluator for poker model performance"""
    def __init__(self):
        # Create a minimal dev set for evaluation
        devset = [
            {
                'input': {
                    'hand': 'AH KH',
                    'table_cards': 'QH JH 2C',
                    'position': 'BTN',
                    'pot_size': 1000,
                    'stack_size': 2000,
                    'opponent_stack': 2000,
                    'game_type': 'cash',
                    'opponent_tendency': 'aggressive'
                },
                'output': {
                    'action': 'raise',
                    'reasoning': 'Flush draw with overcards'
                }
            }
        ]
        super().__init__(devset=devset)
        self.metrics = [
            "win_rate",
            "expected_value", 
            "decision_quality",
            "bluff_efficiency"
        ]
    
    def evaluate(self, model, eval_data) -> Dict[str, float]:
        results = {metric: 0.0 for metric in self.metrics}
        
        for game in eval_data:
            prediction = model(game)
            
            # Calculate various metrics
            results["win_rate"] += self.calculate_win_rate(prediction, game)
            results["expected_value"] += self.calculate_ev(prediction, game)
            results["decision_quality"] += self.evaluate_decision_quality(prediction, game)
            results["bluff_efficiency"] += self.evaluate_bluff_efficiency(prediction, game)
            
        # Average the results
        for metric in results:
            results[metric] /= len(eval_data)
            
        return results
    
    def calculate_win_rate(self, prediction, game):
        # Implement actual win rate calculation
        return 0.75  # Placeholder
        
    def calculate_ev(self, prediction, game):
        # Implement actual EV calculation
        return 0.5  # Placeholder
        
    def evaluate_decision_quality(self, prediction, game):
        # Implement decision quality evaluation
        return 0.8  # Placeholder
        
    def evaluate_bluff_efficiency(self, prediction, game):
        # Implement bluff efficiency calculation
        return 0.6  # Placeholder

class PokerTrainer:
    def __init__(self):
        # Configure DSPy to use GPT-4
        dspy.configure(lm='gpt-4')
        
        self.agent = PokerAgent()
        self.save_dir = os.path.join(os.path.dirname(__file__), 'training_data')
        os.makedirs(self.save_dir, exist_ok=True)
        self.tuner = HyperparameterTuner()
        self.evaluator = PokerEvaluator()
        self.config = TrainingConfig()
        
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []
        if os.path.exists(self.save_dir):
            for file in os.listdir(self.save_dir):
                if file.startswith('checkpoint_epoch_') and file.endswith('.json'):
                    checkpoints.append(file)
        return sorted(checkpoints)

    def load_checkpoint(self, checkpoint_name):
        """Load a specific checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                
            print(f"\n{Fore.GREEN}Loading checkpoint from epoch {checkpoint['epoch']}")
            print(f"Win Rate: {checkpoint['metrics']['win_rate']:.2%}{Style.RESET_ALL}")
            
            if checkpoint.get('model_state'):
                self.agent.load_state_dict(checkpoint['model_state'])
            
            return True
        return False

    def display_training_history(self):
        """Display training history"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            print(f"\n{Fore.YELLOW}Training History:")
            print(f"{Fore.GREEN}{'='*60}")
            for entry in history:
                print(f"\nEpoch {entry['epoch']}:")
                for metric, value in entry['train_metrics'].items():
                    print(f"Training {metric}: {value:.2%}")
                for metric, value in entry['valid_metrics'].items():
                    print(f"Validation {metric}: {value:.2%}")
            print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            return True
        return False

    def tune_hyperparameters(self):
        """Run hyperparameter tuning"""
        print(f"\n{Fore.YELLOW}Starting hyperparameter tuning...")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'temperature': [0.5, 0.7, 0.9],
            'max_tokens': [128, 256, 512]
        }
        
        results = self.tuner.tune_hyperparameters(param_grid)
        self.tuner.plot_results(results)
        
        print(f"\n{Fore.GREEN}Tuning complete! Results saved to tuning_results/")
        print(f"Check tuning_plot.png for visualizations{Style.RESET_ALL}")
        
        return results

    def prepare_training_data(self):
        """Prepare real training data"""
        # This should load/generate actual poker hands and situations
        train_data = []
        valid_data = []
        
        # Example data structure
        game_state = {
            "hand": ["Ah", "Kh"],
            "table": ["Qh", "Jh", "2c"],
            "pot_size": 1000,
            "position": "BTN",
            "optimal_action": "raise",
            "reasoning": "Flush draw with overcards"
        }
        
        # Add more varied game states
        train_data.append(game_state)
        valid_data.append(game_state)
        
        return train_data, valid_data

    def train(self, config: TrainingConfig = None):
        """Actual training implementation"""
        if config:
            self.config = config
            
        print(f"\n{Fore.YELLOW}Starting training with configuration:")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        # Prepare data
        train_data, valid_data = self.prepare_training_data()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        # Training loop
        history = []
        best_metric = float('-inf')
        
        for epoch in range(self.config.num_epochs):
            # Train on batches
            train_metrics = self._train_epoch(train_data)
            
            # Validate
            if epoch % self.config.validation_interval == 0:
                valid_metrics = self.evaluator.evaluate(self.agent, valid_data)
                
                # Save checkpoint if improved
                if valid_metrics['win_rate'] > best_metric:
                    best_metric = valid_metrics['win_rate']
                    self._save_checkpoint(epoch, valid_metrics)
                
                # Check for early stopping
                if early_stopping(1.0 - valid_metrics['win_rate']):
                    print(f"\n{Fore.YELLOW}Early stopping triggered at epoch {epoch}")
                    break
                
                # Record metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_metrics': train_metrics,
                    'valid_metrics': valid_metrics
                }
                history.append(metrics)
                
                # Display progress
                self._display_metrics(metrics)
        
        # Save final history
        self._save_history(history)
        
    def _train_epoch(self, train_data) -> Dict[str, float]:
        """Train for one epoch"""
        total_metrics = {metric: 0.0 for metric in self.evaluator.metrics}
        num_batches = 0
        
        # Process in batches
        for i in range(0, len(train_data), self.config.batch_size):
            batch = train_data[i:i + self.config.batch_size]
            batch_metrics = self._train_batch(batch)
            
            # Accumulate metrics
            for metric in total_metrics:
                total_metrics[metric] += batch_metrics[metric]
            num_batches += 1
            
        # Average metrics
        for metric in total_metrics:
            total_metrics[metric] /= num_batches
            
        return total_metrics
    
    def _train_batch(self, batch) -> Dict[str, float]:
        """Train on a single batch"""
        # Implement actual batch training logic here
        return {metric: 0.8 for metric in self.evaluator.metrics}  # Placeholder
        
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save a checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': self.agent.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = os.path.join(
            self.save_dir, 
            f'checkpoint_epoch_{epoch+1:03d}.json'
        )
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
    def _save_history(self, history: List[Dict]):
        """Save training history"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
    def _display_metrics(self, metrics: Dict):
        """Display current metrics"""
        print(f"\nEpoch {metrics['epoch']}:")
        print(f"Training Metrics:")
        for metric, value in metrics['train_metrics'].items():
            print(f"  {metric}: {value:.2%}")
        print(f"Validation Metrics:")
        for metric, value in metrics['valid_metrics'].items():
            print(f"  {metric}: {value:.2%}")
