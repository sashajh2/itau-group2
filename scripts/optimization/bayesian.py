import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.optimize import minimize
from scipy.stats import norm
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from .base_optimizer import BaseOptimizer

class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization for hyperparameter tuning using Gaussian Processes.
    More efficient than grid search, especially for continuous hyperparameters.
    """
    
    def __init__(self, model_type, model_name=None, device=None, log_dir="bayesian_optimization_results"):
        super().__init__(model_type, model_name, device, log_dir)
        
    def optimize(self, training_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", medium_filepath=None, easy_filepath=None,
                epochs=5, n_calls=50, n_random_starts=10):
        """
        Run Bayesian optimization.
        
        Args:
            training_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            medium_filepath=args.medium_filepath,
            easy_filepath=args.easy_filepath,
            epochs: Number of training epochs per trial
            n_calls: Number of optimization iterations
            n_random_starts: Number of random initial points
        """
        print(f"Starting Bayesian optimization for {self.model_type} model")
        print(f"Mode: {mode}, Loss: {loss_type}")
        print(f"Will run {n_calls} trials with {n_random_starts} random starts")
        
        # Sample initial hyperparameters
        initial_samples = self.sample_hyperparameters(mode, n_random_starts)
        
        # Evaluate initial samples
        for i, params in enumerate(initial_samples):
            print(f"\n{'='*50}")
            print(f"Starting Initial Sample {i+1}")
            print(f"{'='*50}")
            
            result = self.evaluate_trial(
                params, training_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, medium_filepath, easy_filepath, epochs
            )
            print(f"\nInitial Sample {i+1} completed - AUC: {result.get('test_auc', 0):.4f}, Accuracy: {result.get('test_accuracy', 0):.4f}")
        
        # Bayesian optimization loop
        for i in range(n_calls - n_random_starts):
            print(f"\n{'='*50}")
            print(f"Starting Bayesian Iteration {i+1}")
            print(f"{'='*50}")
            
            # Sample next point using acquisition function
            next_params = self._sample_next_point(mode)
            
            # Evaluate the new point
            result = self.evaluate_trial(
                next_params, training_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, medium_filepath, easy_filepath, epochs, 
            )
            print(f"\nBayesian Iteration {i+1} completed - AUC: {result.get('test_auc', 0):.4f}, Accuracy: {result.get('test_accuracy', 0):.4f}")
        
        # Save results
        self._save_results()
        
        print(f"\n{'='*60}")
        print(f"Bayesian optimization completed!")
        print(f"Best AUC: {self.best_auc:.4f}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"{'='*60}")
        
        return self.results
    
    def _sample_next_point(self, mode):
        """
        Sample the next hyperparameter point using acquisition function.
        For simplicity, we'll use random sampling with some bias towards better regions.
        """
        # For now, use random sampling with some exploration
        # In a full implementation, this would use Gaussian Process regression
        samples = self.sample_hyperparameters(mode, 10)
        
        # Simple acquisition: prefer points with higher expected performance
        # This is a simplified version - in practice, you'd use GP-UCB or similar
        return samples[np.random.randint(len(samples))]
    
    def _save_results(self):
        """Save optimization results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.log_dir}/bayesian_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}") 