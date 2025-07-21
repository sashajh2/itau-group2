import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.model_factory import ModelFactory
from model_utils.models.learning import (
    SiameseModelPairs, 
    SiameseModelTriplet,
    SiameseModelSupCon,
    SiameseModelInfoNCE
)

class BaseOptimizer:
    """
    Base class for hyperparameter optimization that works with any model type.
    """
    
    def __init__(self, model_type, model_name=None, device=None, log_dir="optimization_results"):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir
        self.results = []
        self.best_auc = 0.0
        self.best_accuracy = 0.0
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a sample model to get embedding dimension
        self._init_model_info()
    
    def _init_model_info(self):
        """Initialize model information by creating a sample model."""
        try:
            # Create a sample model wrapper to get embedding dimension
            sample_wrapper = ModelFactory.create_model(self.model_type, self.model_name, self.device)
            self.embedding_dim = sample_wrapper.embedding_dim
            print(f"Model type: {self.model_type}, Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Warning: Could not determine embedding dimension for {self.model_type}: {e}")
            # Fallback to default embedding dimension
            self.embedding_dim = 512

    def create_model(self, projection_dim=128):
        """
        Create a model with the specified backbone and projection dimension.
        
        Args:
            projection_dim: Dimension of the projection layer
            
        Returns:
            Model wrapper instance
        """
        # Create the backbone model wrapper
        backbone = ModelFactory.create_model(self.model_type, self.model_name, self.device)
        return backbone
    
    def create_siamese_model(self, mode, projection_dim=128):
        """
        Create a siamese model for the specified mode.
        
        Args:
            mode: Training mode ("pair", "triplet", "supcon", "infonce")
            projection_dim: Dimension of the projection layer
            
        Returns:
            Siamese model instance
        """
        try:
            backbone = self.create_model(projection_dim)
            
            if mode == "pair":
                return SiameseModelPairs(self.embedding_dim, projection_dim, backbone)
            elif mode == "triplet":
                return SiameseModelTriplet(self.embedding_dim, projection_dim, backbone)
            elif mode == "supcon":
                return SiameseModelSupCon(self.embedding_dim, projection_dim, backbone)
            elif mode == "infonce":
                return SiameseModelInfoNCE(self.embedding_dim, projection_dim, backbone)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            print(f"Error creating siamese model for mode {mode}: {e}")
            raise RuntimeError(f"Failed to create siamese model for mode {mode}: {str(e)}")
    
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
        num_workers = 0 if mode == "pair" else 4
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    def create_optimizer(self, model, params):
        """Create optimizer based on parameters"""
        if params['optimizer'] == "adam":
            return torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        elif params['optimizer'] == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        else:  # sgd
            return torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    def sample_hyperparameters(self, mode, n_samples):
        """
        Sample hyperparameters for optimization.
        
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
            
            # Sample optimizer
            optimizer_name = np.random.choice(["adam", "adamw", "sgd"])
            
            # Sample weight decay
            weight_decay = np.exp(np.random.uniform(np.log(1e-6), np.log(1e-3)))
            
            if mode in ["supcon", "infonce"]:
                # Sample temperature (log-uniform)
                temperature = np.exp(np.random.uniform(np.log(0.01), np.log(1.0)))
                samples.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'temperature': temperature,
                    'internal_layer_size': internal_layer_size,
                    'optimizer': optimizer_name,
                    'weight_decay': weight_decay
                })
            else:
                # Sample margin (uniform)
                margin = np.random.uniform(0.1, 2.0)
                samples.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'margin': margin,
                    'internal_layer_size': internal_layer_size,
                    'optimizer': optimizer_name,
                    'weight_decay': weight_decay
                })
        
        return samples
    
    def sample_initial_hyperparameters(self, mode, n_samples):
        """
        Sample initial hyperparameters for optimization (alias for sample_hyperparameters).
        
        Args:
            mode: Training mode
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        return self.sample_hyperparameters(mode, n_samples)
    
    def evaluate_trial(self, params, training_filepath, test_reference_filepath,
                      test_filepath, mode, loss_type, medium_filepath=None, easy_filepath=None,
                      epochs=5):
        """
        Evaluate a single hyperparameter configuration.
        
        Returns:
            Dictionary with results
        """
        try:
            # Convert numpy types to Python types
            batch_size = int(params['batch_size'])
            internal_layer_size = int(params['internal_layer_size'])
            lr = float(params['lr'])
            
            # Log parameters being tested
            param_str = f"LR: {lr:.6f}, Batch: {batch_size}, Layer: {internal_layer_size}, Opt: {params['optimizer']}, WD: {params['weight_decay']:.6f}"
            if mode in ["supcon", "infonce"]:
                param_str += f", Temp: {params['temperature']:.4f}"
            else:
                param_str += f", Margin: {params['margin']:.4f}"
            print(f"Testing parameters: {param_str}")
            
            # Load data
            dataframe = pd.read_parquet(training_filepath)
            medium_dataframe = None
            easy_dataframe = None
            if medium_filepath and easy_filepath:
                medium_dataframe = pd.read_parquet(medium_filepath)
                easy_dataframe = pd.read_parquet(easy_filepath)
            
            # Create dataloaders
            dataloader = self.create_dataloader(dataframe, batch_size, mode)
            medium_loader = None
            easy_loader = None
            if medium_dataframe is not None and easy_dataframe is not None:
                medium_loader = self.create_dataloader(medium_dataframe, batch_size, mode)
                easy_loader = self.create_dataloader(easy_dataframe, batch_size, mode)
            
            # Create model and optimizer
            model = self.create_siamese_model(mode, internal_layer_size).to(self.device)
            optimizer = self.create_optimizer(model, params)
            
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
                log_csv_path=f"{self.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                model_type=mode
            )
            evaluator = Evaluator(model, batch_size=batch_size, model_type=mode)
            
            # Train model
            best_metrics = trainer.train(
                dataloader=dataloader,
                test_reference_filepath=test_reference_filepath,
                test_filepath=test_filepath,
                mode=mode,
                epochs=epochs,
                medium_loader = medium_loader,
                easy_loader=easy_loader
            )
            
            # Log results
            result = {
                "timestamp": datetime.now(),
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": internal_layer_size,
                "optimizer": params['optimizer'],
                "weight_decay": params['weight_decay'],
                "mode": mode,
                "loss_type": loss_type,
                "model_type": self.model_type,
                "model_name": self.model_name,
                "test_auc": best_metrics.get('roc_auc', 0),
                "test_accuracy": best_metrics.get('accuracy', 0),
                **best_metrics
            }
            
            # Add mode-specific parameters
            if mode in ["supcon", "infonce"]:
                result["temperature"] = float(params['temperature'])
            else:
                result["margin"] = float(params['margin'])
            
            self.results.append(result)
            
            # Update best metrics
            if best_metrics.get('roc_auc', 0) > self.best_auc:
                self.best_auc = best_metrics['roc_auc']
            if best_metrics.get('accuracy', 0) > self.best_accuracy:
                self.best_accuracy = best_metrics['accuracy']
            
            return result
            
        except Exception as e:
            print(f"\nTrial failed with error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback:")
            traceback.print_exc()
            return {
                "timestamp": datetime.now(),
                "error": str(e),
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": internal_layer_size,
                "mode": mode,
                "loss_type": loss_type,
                "model_type": self.model_type,
                "model_name": self.model_name,
                "test_auc": 0.0,
                "test_accuracy": 0.0
            } 