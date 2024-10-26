from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class HyperparameterTuner:
    def __init__(self):
        self.results_dir = os.path.join(os.path.dirname(__file__), 'tuning_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def tune_hyperparameters(self, param_grid):
        """Run grid search over hyperparameters"""
        # Placeholder for actual grid search implementation
        results = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'accuracy': [0.75, 0.82, 0.78]
        }
        
        # Save results
        results_path = os.path.join(self.results_dir, 'tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def plot_results(self, results):
        """Generate visualization of tuning results"""
        plt.figure(figsize=(10, 6))
        
        # Plot learning rate vs accuracy
        plt.subplot(1, 2, 1)
        sns.lineplot(x=results['learning_rate'], y=results['accuracy'])
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title('Learning Rate Impact')
        
        # Plot batch size vs accuracy 
        plt.subplot(1, 2, 2)
        sns.lineplot(x=results['batch_size'], y=results['accuracy'])
        plt.xlabel('Batch Size')
        plt.ylabel('Accuracy')
        plt.title('Batch Size Impact')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'tuning_plot.png')
        plt.savefig(plot_path)
        plt.close()
import os
import json
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting functionality will be disabled.")

class HyperparameterTuner:
    def __init__(self):
        self.results_dir = os.path.join(os.path.dirname(__file__), 'tuning_results')
        os.makedirs(self.results_dir, exist_ok=True)

    def tune_hyperparameters(self, param_grid):
        """Run hyperparameter tuning"""
        # Placeholder implementation
        results = {
            'best_params': {
                'learning_rate': 0.01,
                'batch_size': 32
            },
            'scores': [
                {'params': {'learning_rate': 0.001, 'batch_size': 16}, 'score': 0.75},
                {'params': {'learning_rate': 0.01, 'batch_size': 32}, 'score': 0.85},
                {'params': {'learning_rate': 0.1, 'batch_size': 64}, 'score': 0.80}
            ]
        }
        
        # Save results
        results_path = os.path.join(self.results_dir, 'tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results

    def plot_results(self, results):
        """Plot tuning results"""
        if not HAS_MATPLOTLIB:
            print("Cannot create plot: matplotlib is not installed")
            return

        try:
            plt.figure(figsize=(10, 6))
            
            # Extract scores and parameters
            scores = [r['score'] for r in results['scores']]
            params = [f"lr={r['params']['learning_rate']}\nb={r['params']['batch_size']}" 
                     for r in results['scores']]
            
            # Create bar plot
            plt.bar(range(len(scores)), scores)
            plt.xticks(range(len(scores)), params, rotation=45)
            plt.ylabel('Score')
            plt.title('Hyperparameter Tuning Results')
            
            # Add value labels on top of bars
            for i, score in enumerate(scores):
                plt.text(i, score, f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.results_dir, 'tuning_plot.png')
            plt.savefig(plot_path)
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
