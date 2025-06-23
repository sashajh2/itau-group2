import torch
import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_curve
from utils.evals import find_best_threshold_youden
from scripts.evaluation.evaluator import Evaluator
from model_utils.loss.supcon_loss import SupConLoss
from model_utils.loss.infonce_loss import InfoNCELoss
from utils.curriculum import ManualCurriculum, SelfPacedCurriculum, BanditCurriculum, CurriculumDataLoader


class CurriculumTrainer:
    """
    Enhanced trainer with curriculum learning support.
    Extends the base trainer with curriculum learning capabilities.
    """
    
    def __init__(self, model, criterion, optimizer, device, log_csv_path="curriculum_training_log.csv"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_csv_path = log_csv_path
        self.model.to(device)
        self.evaluator = Evaluator(model)
        
    def create_curriculum_dataloader(self, dataframe, curriculum_type, batch_size, mode, **curriculum_params):
        """Create curriculum dataloader based on type."""
        # Create appropriate dataset
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
        
        # Create curriculum strategy
        if curriculum_type == "manual":
            curriculum = ManualCurriculum(dataset, mode, **curriculum_params)
        elif curriculum_type == "self_paced":
            curriculum = SelfPacedCurriculum(dataset, mode, **curriculum_params)
        elif curriculum_type == "bandit":
            curriculum = BanditCurriculum(dataset, mode, **curriculum_params)
        else:
            raise ValueError(f"Unknown curriculum type: {curriculum_type}")
        
        return CurriculumDataLoader(dataset, curriculum, batch_size, mode)
    
    def train_epoch_with_curriculum(self, curriculum_loader, mode="pair"):
        """Train for one epoch using curriculum learning."""
        self.model.train()
        epoch_loss = 0.0
        batch_losses = []
        batch_indices = []
        
        # Set epoch for curriculum
        curriculum_loader.set_epoch(self.current_epoch)
        
        # Iterate through batches
        for i, batch in enumerate(curriculum_loader):
            # Extract data and indices
            if mode == "pair":
                text1, text2, labels, indices = batch
                outputs = self.model(text1, text2)
                loss = self.criterion(*outputs, labels)
            elif mode == "triplet":
                anchor, positive, negative, indices = batch
                outputs = self.model(anchor, positive, negative)
                loss = self.criterion(*outputs)
            elif mode == "supcon":
                anchor, positives, negatives, indices = batch
                outputs = self.model(anchor, positives, negatives)
                loss = self.criterion(*outputs)
            elif mode == "infonce":
                anchor, positive, negatives, indices = batch
                outputs = self.model(anchor, positive, negatives)
                loss = self.criterion(*outputs)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Store loss and indices for curriculum update
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.extend([batch_loss] * len(indices))
            batch_indices.extend(indices)
            
            if i % 100 == 0:
                print(f"Step {i} complete out of {len(curriculum_loader)}")
        
        # Update curriculum with batch losses
        curriculum_loader.update_curriculum(batch_losses, batch_indices)
        
        return epoch_loss / len(curriculum_loader)
    
    def evaluate(self, test_reference_filepath, test_filepath):
        """Evaluate model on test set."""
        self.model.eval()
        results_df, metrics = self.evaluator.evaluate(test_reference_filepath, test_filepath)
        return metrics
    
    def train_with_curriculum(self, reference_filepath, test_reference_filepath, test_filepath,
                            mode="pair", curriculum_type="manual", epochs=30, batch_size=32,
                            warmup_filepath=None, warmup_epochs=5, **curriculum_params):
        """
        Main training loop with curriculum learning.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair", "triplet", "supcon", or "infonce"
            curriculum_type: "manual", "self_paced", or "bandit"
            epochs: Number of training epochs
            batch_size: Batch size for training
            warmup_filepath: Optional warmup data path
            warmup_epochs: Number of warmup epochs
            **curriculum_params: Additional parameters for curriculum strategy
        """
        import pandas as pd
        
        # Load data
        dataframe = pd.read_pickle(reference_filepath)
        warmup_dataframe = None
        if warmup_filepath:
            warmup_dataframe = pd.read_pickle(warmup_filepath)
        
        # Create curriculum dataloaders
        main_loader = self.create_curriculum_dataloader(
            dataframe, curriculum_type, batch_size, mode, **curriculum_params
        )
        
        warmup_loader = None
        if warmup_dataframe is not None:
            warmup_loader = self.create_curriculum_dataloader(
                warmup_dataframe, curriculum_type, batch_size, mode, **curriculum_params
            )
        
        # Training tracking
        best_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        best_epochs = {
            'accuracy': -1,
            'precision': -1,
            'recall': -1
        }
        best_epoch_loss = float('inf')
        
        # Logging
        with open(self.log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall", "Curriculum_Type"])
            
            for epoch in range(epochs):
                self.current_epoch = epoch
                
                # Use warmup loader if provided and in warmup phase
                current_loader = warmup_loader if warmup_loader and epoch < warmup_epochs else main_loader
                
                # Train epoch with curriculum
                avg_loss = self.train_epoch_with_curriculum(current_loader, mode)
                print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
                
                if avg_loss < best_epoch_loss:
                    best_epoch_loss = avg_loss
                
                # Evaluate
                metrics = self.evaluate(test_reference_filepath, test_filepath)
                
                # Log metrics
                writer.writerow([
                    epoch + 1, 
                    avg_loss, 
                    metrics['accuracy'], 
                    metrics['precision'], 
                    metrics['recall'],
                    curriculum_type
                ])
                
                print(f"Epoch {epoch+1} - Test Accuracy: {metrics['accuracy']:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f}")
                
                # Track best metrics
                for metric in ['accuracy', 'precision', 'recall']:
                    if metrics[metric] > best_metrics[metric]:
                        best_metrics[metric] = metrics[metric]
                        best_epochs[metric] = epoch + 1
        
        # Print summary
        print("\n=== Curriculum Training Summary ===")
        print(f"Curriculum Type: {curriculum_type}")
        for metric in ['accuracy', 'precision', 'recall']:
            print(f"Best {metric.capitalize()}: {best_metrics[metric]:.4f} "
                  f"at epoch {best_epochs[metric]}")
        
        return best_epoch_loss, best_metrics
    
    def compare_curriculum_methods(self, reference_filepath, test_reference_filepath, test_filepath,
                                 mode="pair", epochs=20, batch_size=32, **curriculum_params):
        """
        Compare different curriculum learning methods.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: Training mode
            epochs: Number of epochs per method
            batch_size: Batch size
            **curriculum_params: Additional curriculum parameters
        """
        curriculum_types = ["manual", "self_paced", "bandit"]
        results = {}
        
        # Save initial model and optimizer state before any training
        initial_model_state = self.model.state_dict().copy()
        initial_optimizer_state = self.optimizer.state_dict().copy()
        
        for curriculum_type in curriculum_types:
            print(f"\n=== Training with {curriculum_type} curriculum ===")
            
            # Reset model and optimizer to initial state for fair comparison
            self.model.load_state_dict(initial_model_state)
            self.optimizer.load_state_dict(initial_optimizer_state)
            
            # Train with current curriculum
            best_loss, best_metrics = self.train_with_curriculum(
                reference_filepath=reference_filepath,
                test_reference_filepath=test_reference_filepath,
                test_filepath=test_filepath,
                mode=mode,
                curriculum_type=curriculum_type,
                epochs=epochs,
                batch_size=batch_size,
                **curriculum_params
            )
            
            results[curriculum_type] = {
                'best_loss': best_loss,
                'best_metrics': best_metrics
            }
        
        # Print comparison results
        print("\n=== Curriculum Methods Comparison ===")
        for curriculum_type, result in results.items():
            print(f"\n{curriculum_type.upper()}:")
            print(f"  Best Loss: {result['best_loss']:.4f}")
            for metric, value in result['best_metrics'].items():
                print(f"  Best {metric.capitalize()}: {value:.4f}")
        
        return results 