import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.optimize import minimize
from scipy.stats import norm
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.siamese import SiameseCLIPModelPairs, SiameseCLIPTriplet
from model_utils.models.supcon import SiameseCLIPSupCon
from model_utils.models.infonce import SiameseCLIPInfoNCE

class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning using Gaussian Processes.
    More efficient than grid search, especially for continuous hyperparameters.
    """
    
    def __init__(self, model_class, device, log_dir="bayesian_optimization_results"):
        self.model_class = model_class
        self.device = device
        self.log_dir = log_dir
        self.results = []
        self.best_auc = 0.0  # Track best AUC across all trials
        self.best_accuracy = 0.0  # Track best accuracy across all trials
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
    def get_loss_class(self, mode, loss_type):
        """Get appropriate loss class based on mode and type"""
        if mode == "pair":
            if loss_type == "cosine":
                from model_utils.loss.pair_losses import CosineLoss
                return CosineLoss
            elif loss_type == "euclidean":
                from model_utils.loss.pair_losses import EuclideanLoss
                return EuclideanLoss
        elif mode == "triplet":
            if loss_type == "cosine":
                from model_utils.loss.triplet_losses import CosineTripletLoss
                return CosineTripletLoss
            elif loss_type == "euclidean":
                from model_utils.loss.triplet_losses import EuclideanTripletLoss
                return EuclideanTripletLoss
            elif loss_type == "hybrid":
                from model_utils.loss.triplet_losses import HybridTripletLoss
                return HybridTripletLoss
        elif mode == "supcon":
            if loss_type == "supcon":
                from model_utils.loss.supcon_loss import SupConLoss
                return SupConLoss
            elif loss_type == "infonce":
                from model_utils.loss.infonce_loss import InfoNCELoss
                return InfoNCELoss
        elif mode == "infonce":
            from model_utils.loss.infonce_loss import InfoNCELoss
            return InfoNCELoss
        raise ValueError(f"Unsupported mode/loss_type combination: {mode}/{loss_type}")
    
    def create_dataloader(self, dataframe, batch_size, mode):
        """Create appropriate dataloader based on mode"""
        if mode == "pair":
            from utils.data import TextPairDataset
            dataset = TextPairDataset(dataframe)
        elif mode == "triplet":
            from utils.data import TripletDataset
            dataset = TripletDataset(dataframe)
        elif mode == "supcon":
            from utils.data import SupConDataset
            dataset = SupConDataset(dataframe)
        elif mode == "infonce":
            from utils.data import InfoNCEDataset
            dataset = InfoNCEDataset(dataframe)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        from torch.utils.data import DataLoader
        # Use num_workers=0 for pair mode to avoid device mismatch issues
        # num_workers = 0 if mode == "pair" else 4
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    def sample_hyperparameters(self, mode, n_samples):
        """
        Sample hyperparameters for Bayesian optimization.
        
        Args:
            mode: Training mode
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        samples = []
        
        for _ in range(n_samples):
            # Sample learning rate (log-uniform)
            lr = np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2)))
            
            # Sample batch size (discrete)
            batch_size = np.random.choice([16, 32, 64, 128])
            
            # Sample internal layer size (discrete)
            internal_layer_size = np.random.choice([64, 128, 256, 512])
            
            if mode in ["supcon", "infonce"]:
                # Sample temperature (log-uniform)
                temperature = np.exp(np.random.uniform(np.log(0.01), np.log(1.0)))
                samples.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'temperature': temperature,
                    'internal_layer_size': internal_layer_size
                })
            else:
                # Sample margin (uniform)
                margin = np.random.uniform(0.1, 2.0)
                samples.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'margin': margin,
                    'internal_layer_size': internal_layer_size
                })
        
        return samples
    
    def evaluate_trial(self, params, reference_filepath, test_reference_filepath,
                      test_filepath, mode, loss_type, medium_filepath=None, easy_filepath=None,
                      epochs=5, ):
        """
        Evaluate a single hyperparameter configuration.
        
        Returns:
            float: Negative accuracy (for minimization)
        """
        try:
            # Convert numpy types to Python types
            batch_size = int(params['batch_size'])
            internal_layer_size = int(params['internal_layer_size'])
            lr = float(params['lr'])
            
            # Load data
            dataframe = pd.read_pickle(reference_filepath)
            warmup_dataframe = None
            if easy_filepath and medium_filepath:
                easy_dataframe = pd.read_pickle(easy_filepath)
                medium_dataframe = pd.read_pickle(medium_filepath)
            
            # Create dataloaders
            dataloader = self.create_dataloader(dataframe, batch_size, mode)
            medium_loader = None
            hard_loader = None
            if medium_dataframe is not None and easy_dataframe is not None:
                medium_loader = self.create_dataloader(medium_dataframe, batch_size, mode)
                easy_loader = self.create_dataloader(easy_dataframe, batch_size, mode)
            
            # Create model and optimizer
            model = self.model_class(
                embedding_dim=512,
                projection_dim=internal_layer_size
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Get loss class and create criterion
            loss_class = self.get_loss_class(mode, loss_type)
            if mode in ["supcon", "infonce"]:
                temperature = float(params['temperature'])
                criterion = loss_class(temperature=temperature)
            elif mode == "triplet" and loss_type == "hybrid":
                margin = float(params['margin'])
                criterion = loss_class(margin=margin, alpha=0.5)
            else:
                margin = float(params['margin'])
                criterion = loss_class(margin=margin)
            
            # Create trainer and evaluator
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                log_csv_path=f"{self.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            evaluator = Evaluator(model, batch_size=batch_size)
            
            # Train model
            best_metrics = trainer.train(
                dataloader=dataloader,
                test_reference_filepath=test_reference_filepath,
                test_filepath=test_filepath,
                mode=mode,
                epochs=epochs,
                medium_loader=medium_loader,
                easy_loader=easy_loader,
                medium=medium_loader
            )
            
            # Log results
            result = {
                "timestamp": datetime.now(),
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": internal_layer_size,
                "epochs": epochs,
                "train_loss": best_metrics.get('loss', 0.0),  # Use loss from best metrics if available
                "test_accuracy": best_metrics['accuracy'],
                "test_auc": best_metrics['roc_auc'],
                "threshold": best_metrics.get('threshold', 0.5),  # Default threshold
                "loss_type": loss_type,
                **{k: v for k, v in locals().items() if k in ['temperature', 'margin'] and v is not None}
            }
            self.results.append(result)
            
            # Track best AUC
            current_auc = best_metrics['roc_auc']
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                print(f"Trial {len(self.results)}: New best AUC = {self.best_auc:.4f}")
            else:
                print(f"Trial {len(self.results)}: AUC = {current_auc:.4f} (Best = {self.best_auc:.4f})")
            
            # Track best accuracy
            current_accuracy = best_metrics['accuracy']
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                print(f"Trial {len(self.results)}: New best accuracy = {self.best_accuracy:.4f}")
            else:
                print(f"Trial {len(self.results)}: Accuracy = {current_accuracy:.4f} (Best = {self.best_accuracy:.4f})")
            
            return -best_metrics['accuracy']
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return -0.0  # Return worst possible score
    
    def optimize(self, reference_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", medium_filepath=None, easy_filepath=None,
                epochs=5, n_calls=50, n_random_starts=10):
        """
        Perform Bayesian optimization over hyperparameters.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair", "triplet", "supcon", or "infonce"
            loss_type: Type of loss function to use
            epochs: Number of training epochs
            medium_filepath: optional filepath to medium 
            easy_filepath: options filepath to easy
            n_calls: Number of optimization iterations
            n_random_starts: Number of random initial points
        """
        print(f"Starting Bayesian optimization for {mode} mode with {loss_type} loss")
        print(f"Will perform {n_calls} trials with {n_random_starts} random starts")
        
        # Sample initial hyperparameters
        initial_samples = self.sample_hyperparameters(mode, n_random_starts)
        
        best_config = None
        
        # Evaluate initial random samples
        for i, params in enumerate(initial_samples):
            print(f"\nInitial trial {i+1}/{n_random_starts}")
            print(f"Parameters: {params}")
            
            accuracy = -self.evaluate_trial(
                params, reference_filepath, test_reference_filepath,
                test_filepath, mode, loss_type, easy_filepath, medium_filepath, epochs,
            )
            
            if accuracy > self.best_accuracy:
                best_config = {**params, "best_accuracy": accuracy}
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Best accuracy so far: {self.best_accuracy:.4f}")
            print(f"Best AUC so far: {self.best_auc:.4f}")
        
        # Continue with additional random sampling (simplified Bayesian optimization)
        remaining_trials = n_calls - n_random_starts
        additional_samples = self.sample_hyperparameters(mode, remaining_trials)
        
        for i, params in enumerate(additional_samples):
            print(f"\nTrial {n_random_starts + i + 1}/{n_calls}")
            print(f"Parameters: {params}")
            
            accuracy = -self.evaluate_trial(
                params, reference_filepath, test_reference_filepath,
                test_filepath, mode, loss_type, easy_filepath, medium_filepath, epochs,
            )
            
            if accuracy > self.best_accuracy:
                best_config = {**params, "best_accuracy": accuracy}
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Best accuracy so far: {self.best_accuracy:.4f}")
            print(f"Best AUC so far: {self.best_auc:.4f}")
        
        print(f"\nOptimization completed!")
        print(f"Best accuracy: {self.best_accuracy:.4f}")
        print(f"Best parameters: {best_config}")
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.log_dir}/bayesian_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         index=False)
        
        # Save best configuration
        best_config = {
            **best_config,
            "mode": mode,
            "loss_type": loss_type,
            "optimization_method": "bayesian"
        }
        
        return best_config, results_df 