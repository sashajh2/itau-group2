import torch
import csv
from sklearn.metrics import precision_score, recall_score, roc_curve
from utils.evals import find_best_threshold_youden
from scripts.evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import random
from model_utils.loss.supcon_loss import SupConLoss
from model_utils.loss.infonce_loss import InfoNCELoss
from utils.curriculum import get_curriculum_ratios

class Trainer:
    """
    Unified training interface for both pair and triplet models.
    Handles training, validation, and logging.
    """
    def __init__(self, model, criterion, optimizer, device, log_csv_path="training_log.csv", model_type=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_csv_path = log_csv_path
        self.model_type = model_type
        self.model.to(device)
        self.evaluator = Evaluator(model, model_type=model_type)

    def train_epoch(self, dataloader, mode="pair", track_pg = False):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        total_pg = 0
        pg_count = 0
        
        for i, batch in enumerate(dataloader):
            # Handle different modes based on expected inputs
            if mode == "triplet":
                anchor_texts, positive_texts, negative_texts = batch
                outputs = self.model(anchor_texts, positive_texts, negative_texts)
            elif mode == "supcon":
                anchor_texts, positive_texts, negative_texts = batch
                outputs = self.model(anchor_texts, positive_texts, negative_texts)
            elif mode == "infonce":
                anchor_texts, positive_texts, negative_texts = batch
                outputs = self.model(anchor_texts, positive_texts, negative_texts)
            else:  # pair mode
                outputs = self.model(*batch)
            loss = self.criterion(*outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if track_pg:
                with torch.no_grad():
                    outputs_after = self.model(*batch)
                    loss_after = self.criterion(*outputs_after)
                    total_pg += (loss.item() - loss_after.item())
                    pg_count += 1

            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Step {i} complete out of {len(dataloader)}")

        avg_pg = total_pg/pg_count if track_pg and pg_count > 0 else None
        return epoch_loss / len(dataloader), avg_pg

    def evaluate(self, test_reference_filepath, test_filepath):
        """Evaluate model on test set"""
        self.model.eval()
        results_df, metrics = self.evaluator.evaluate(test_reference_filepath, test_filepath)
        return metrics

    # pass in curriculum learning parameter 
    def train(self, dataloader, test_reference_filepath, test_filepath, 
             mode="pair", epochs=30, medium_loader=None, easy_loader=None, curriculum = None):
        """
        Main training loop with optional warmup.
        
        Args:
            dataloader: Main training dataloader
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair" or "triplet"
            epochs: Number of training epochs
            
        Returns:
            dict: Best metrics achieved during training
        """
        # bandit learning setup - only if warmup_loader is provided
        if medium_loader is not None and easy_loader is not None:
            datasets = {
                "easy": easy_loader.dataset,
                "medium": medium_loader.dataset,
                "hard": dataloader.dataset
            }
            # bandit learning tracking
            rewards = {k: [] for k in datasets}
        else:
            datasets = {}
            rewards = {}
        
        # keeping track of accuracy
        prev_accuracy = 0.0
        best_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'roc_auc': 0.0
        }
        best_epochs = {
            'accuracy': -1,
            'precision': -1,
            'recall': -1,
            'roc_auc': -1
        }
        best_epoch_loss = float('inf')

        with open(self.log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall", "AUC"])

            for epoch in range(epochs):
                
                # self paced curriculum learning - only if warmup_loader is provided
                if curriculum == "self" and medium_loader is not None and easy_loader is not None:
                    # new ratios for three dataset
                    ratios = get_curriculum_ratios(epoch, epochs)
                    total_samples = len(dataloader.dataset)
                    easy_n = int(ratios["easy"] * total_samples)
                    medium_n = int(ratios["medium"] * total_samples)
                    hard_n = int(ratios["hard"] * total_samples)

                    easy_idx = np.random.choice(len(easy_loader.dataset), easy_n, replace=False)
                    med_idx = np.random.choice(len(medium_loader.dataset), medium_n, replace=False)
                    hard_idx = np.random.choice(len(dataloader.dataset), hard_n, replace=False)

                    mixed_dataset = ConcatDataset([
                        Subset(easy_loader.dataset, easy_idx),
                        Subset(medium_loader.dataset, med_idx),
                        Subset(dataloader.dataset, hard_idx)
                    ])

                    current_loader = DataLoader(mixed_dataset, batch_size=dataloader.batch_size, shuffle=True)
                
                # bandit curriculum learning - only if warmup_loader is provided
                elif curriculum == "bandit" and medium_loader is not None and easy_loader is not None:

                    # exploration rate
                    epsilon = 0.1 
                    reward_window = 3
                    
                    avg_rewards = {
                        k: np.mean(v[-reward_window:]) if v else 0.0
                        for k, v in rewards.items()
                    }

                     # Bandit curriculum learning
                    if random.random() < epsilon:
                        chosen = random.choice(list(datasets.keys()))

                    else:
                        # reward estimation
                        chosen = max(avg_rewards, key=avg_rewards.get)

                    current_loader = DataLoader(
                        datasets[chosen],
                        batch_size=dataloader.batch_size,
                        shuffle=True
                        )

                else:
                    # non curriculum mode
                    phase_len = epochs // 3
                    if epoch < phase_len and easy_loader:
                        current_loader = easy_loader
                    elif epoch < 2 * phase_len and medium_loader:
                        current_loader = medium_loader
                    else:
                        current_loader = dataloader

                # Train epoch
                avg_loss, avg_pg = self.train_epoch(current_loader, track_pg = (curriculum == "bandit"))
                print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

                if avg_loss < best_epoch_loss:
                    best_epoch_loss = avg_loss

                # Evaluate
                metrics = self.evaluate(test_reference_filepath, test_filepath)

                if curriculum == "bandit" and avg_pg is not None and easy_loader is not None and medium_loader is not None:
                    rewards[chosen].append(avg_pg)
                
                # Log metrics
                writer.writerow([
                    epoch + 1, 
                    avg_loss, 
                    metrics['accuracy'], 
                    metrics['precision'], 
                    metrics['recall'],
                    metrics.get('roc_auc', None)
                ])

                print(f"Epoch {epoch+1} - Test Accuracy: {metrics['accuracy']:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f} | "
                      f"AUC: {metrics.get('roc_auc', float('nan')):.4f}")

                # Track best metrics
                for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
                    if metric in metrics and metrics[metric] > best_metrics[metric]:
                        best_metrics[metric] = metrics[metric]
                        best_epochs[metric] = epoch + 1

        # Print summary
        print("\n=== Best Epochs Summary ===")
        for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
            if best_epochs[metric] != -1:
                print(f"Best {metric.capitalize()}: {best_metrics[metric]:.4f} "
                      f"at epoch {best_epochs[metric]}")
        print(f"Best Loss: {best_epoch_loss:.4f}")

        # Add loss to best_metrics
        best_metrics['loss'] = best_epoch_loss

        return best_metrics 