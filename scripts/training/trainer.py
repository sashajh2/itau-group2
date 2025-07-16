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

class Trainer:
    """
    Unified training interface for both pair and triplet models.
    Handles training, validation, and logging.
    """
    def __init__(self, model, criterion, optimizer, device, log_csv_path="training_log.csv"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_csv_path = log_csv_path
        self.model.to(device)
        self.evaluator = Evaluator(model)

    def train_epoch(self, dataloader, mode="pair"):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            # Unified logic: model and criterion handle all modes
            outputs = self.model(*batch)
            loss = self.criterion(*outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Step {i} complete out of {len(dataloader)}")

        return epoch_loss / len(dataloader)

    def evaluate(self, test_reference_filepath, test_filepath):
        """Evaluate model on test set"""
        self.model.eval()
        results_df, metrics = self.evaluator.evaluate(test_reference_filepath, test_filepath)
        return metrics

    # pass in curriculum learning parameter 
    def train(self, medium_laoder, hard_loader, test_reference_filepath, test_filepath, 
             mode="pair", epochs=30, warmup_loader=None, warmup_epochs=5, curriculum = None):
        """
        Main training loop with optional warmup.
        
        Args:
            dataloader: Main training dataloader
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair" or "triplet"
            epochs: Number of training epochs
            warmup_loader: Optional warmup dataloader
            warmup_epochs: Number of warmup epochs
            
        Returns:
            dict: Best metrics achieved during training
        """
        # bandit learning setup 
        datasets = {
            "easy": warmup_loader.dataset,
            "medium": medium_loader.dataset,
            "hard": dataloader.dataset
        }

        # bandit learning tracking
        rewards = {k: [] for k in datasets}
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
                
                # self paced curriculum learning
                if curriculum == "self":

                    # new ratios for three dataset
                    
                    ratios = {
                        "easy": max(0.0, 1.0 - 0.1 * epoch),     # decay easy
                        "medium": min(0.5, 0.1 * epoch),         # increase medium
                        "hard": min(0.4, 0.05 * epoch)           # slowly ramp hard
                    }

                    total_samples = len(hard_loader.dataset)
                    easy_n = int(ratios["easy"] * total_samples)
                    medium_n = int(ratios["medium"] * total_samples)
                    hard_n = int(ratios["hard"] * total_samples)

                    easy_idx = np.random.choice(len(warmup_loader.dataset), easy_n, replace=False)
                    med_idx = np.random.choice(len(medium_loader.dataset), medium_n, replace=False)
                    hard_idx = np.random.choice(len(hard_loader.dataset), hard_n, replace=False)

                    mixed_dataset = ConcatDataset([
                        Subset(warmup_loader.dataset, easy_idx),
                        Subset(medium_loader.dataset, med_idx),
                        Subset(hard_loader.dataset, hard_idx)
                    ])

                    current_loader = DataLoader(mixed_dataset, batch_size=hard_loader.batch_size, shuffle=True)

                    
                # bandit curriculum learning
                elif curriculum == "bandit":

                    # exploration rate
                    epsilon = 0.1 
                    reward_window = 5
                    
                    avg_rewards = {
                        k: np.mean(v[-reward_window:]) if v else 0.0
                        for k, v in rewards.items()
                    }

                     # Bandit curriculum learning
                    if random.random() < epsilon:
                        chosen = random.choice(list(datasets.keys()))
                    else:
                        chosen = max(avg_rewards, key=avg_rewards.get)

                    batch_size = hard_loader.batch_size
                    alloc = {
                        "easy": batch_size // 3 if chosen == "easy" else 0,
                        "medium": batch_size // 3 if chosen == "medium" else 0,
                        "hard": batch_size if chosen == "hard" else batch_size // 3
                    }
                    remaining = batch_size - sum(alloc.values())
                    alloc[chosen] += remaining  # fill up batch size

                    sampled = {
                        k: np.random.choice(len(datasets[k]), alloc[k], replace=False)
                        for k in datasets
                        if alloc[k] > 0
                    }

                    mixed_dataset = ConcatDataset([
                        Subset(datasets[k], sampled[k]) for k in sampled
                    ])

                    current_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True)
                    print(f"Bandit (chose {chosen}) - batch alloc: {alloc}")

                else:
                    # === CHANGED: use warmup_loader (easy), medium_loader, or hard_loader ===
                    if epoch < warmup_epochs and warmup_loader:
                        current_loader = warmup_loader
                    elif epoch < warmup_epochs + 3:
                        current_loader = medium_loader
                    else:
                        current_loader = hard_loader

                # Train epoch
                avg_loss = self.train_epoch(current_loader, mode)
                print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

                if avg_loss < best_epoch_loss:
                    best_epoch_loss = avg_loss

                # Evaluate
                metrics = self.evaluate(test_reference_filepath, test_filepath)

                if curriculum == "bandit":
                    delta_acc = metrics['accuracy'] - prev_accuracy
                    prev_accuracy = metrics['accuracy']
                    rewards[chosen_dataset_name].append(delta_acc)
                
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