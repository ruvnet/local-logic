import os
import json
from colorama import Fore, Style
from poker_bot.poker_agent import PokerAgent
from poker_bot.hyperparameter_tuner import HyperparameterTuner

class PokerTrainer:
    def __init__(self):
        self.agent = PokerAgent()
        self.save_dir = os.path.join(os.path.dirname(__file__), 'training_data')
        os.makedirs(self.save_dir, exist_ok=True)
        self.tuner = HyperparameterTuner()
        
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
            print(f"Validation accuracy: {checkpoint['metrics']['accuracy']:.2%}{Style.RESET_ALL}")
            
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
                print(f"Training Accuracy: {entry['train_metrics']['accuracy']:.2%}")
                print(f"Validation Accuracy: {entry['valid_metrics']['accuracy']:.2%}")
            print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
            return True
        return False

    def tune_hyperparameters(self):
        """Run hyperparameter tuning"""
        print(f"\n{Fore.YELLOW}Starting hyperparameter tuning...")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64]
        }
        
        results = self.tuner.tune_hyperparameters(param_grid)
        self.tuner.plot_results(results)
        
        print(f"\n{Fore.GREEN}Tuning complete! Results saved to tuning_results/")
        print(f"Check tuning_plot.png for visualizations{Style.RESET_ALL}")
        
        return results

    def train(self, num_epochs=10, batch_size=32, learning_rate=0.01):
        """Train the poker agent"""
        print(f"\n{Fore.YELLOW}Starting training for {num_epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        # Training implementation will go here
        # This is a placeholder that saves dummy metrics
        history = []
        for epoch in range(num_epochs):
            metrics = {
                'epoch': epoch + 1,
                'train_metrics': {'accuracy': 0.75 + epoch * 0.02},
                'valid_metrics': {'accuracy': 0.70 + epoch * 0.02}
            }
            history.append(metrics)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state': None,  # Add actual model state here
                'metrics': metrics['valid_metrics']
            }
            
            checkpoint_path = os.path.join(
                self.save_dir, 
                f'checkpoint_epoch_{epoch+1:03d}.json'
            )
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"Training Accuracy: {metrics['train_metrics']['accuracy']:.2%}")
            print(f"Validation Accuracy: {metrics['valid_metrics']['accuracy']:.2%}")
        
        # Save training history
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
