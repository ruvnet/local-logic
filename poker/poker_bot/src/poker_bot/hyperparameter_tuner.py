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
