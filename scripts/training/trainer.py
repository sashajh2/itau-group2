import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from scripts.evaluation.evaluator import Evaluator
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


    def evaluate(self, test_filepath):
        """Evaluate model on test set (pairwise only)"""
        self.model.eval()
        _, metrics = self.evaluator.evaluate(test_filepath)
        return metrics


    def train(self, dataloader, test_filepath, 
             mode="pair", epochs=30, medium_loader=None, easy_loader=None, curriculum = None, validate_filepath=None):
        """
        Main training loop with optional warmup and validation.
        """
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
        best_val_metrics = None
        best_val_epoch = -1
        val_metrics_at_halfway = None
        halfway_epoch = (epochs - 1) // 2

        if medium_loader is not None and easy_loader is not None:
            datasets = {
                "easy": easy_loader.dataset,
                "medium": medium_loader.dataset,
                "hard": dataloader.dataset
            }
            
            rewards = {k: [] for k in datasets}
        else:
            datasets = {}
            rewards = {}

        for epoch in range(epochs):

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

            elif curriculum == "bandit" and medium_loader is not None and easy_loader is not None:
                epsilon = 0.1
                reward_window = 3

                # Bandit curriculum learning
                # For simplicity, use a local rewards dict
                avg_rewards = {
                    k: np.mean(v[-reward_window:]) if v else 0.0
                    for k, v in rewards.items()
                }

                if random.random() < epsilon:
                    chosen = random.choice(list(datasets.keys()))
                else:
                   chosen = max(avg_rewards, key=avg_rewards.get)

                current_loader = DataLoader(
                    datasets[chosen],
                    batch_size=dataloader.batch_size,
                    shuffle=True
                )
            
            else:
                phase_len = epochs // 3
                if epoch < phase_len and easy_loader:
                    current_loader = easy_loader
                elif epoch < 2 * phase_len and medium_loader:
                    current_loader = medium_loader
                else:
                    current_loader = dataloader

            avg_loss, avg_pg = self.train_epoch(current_loader, track_pg = (curriculum == "bandit"))            
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
            
            if curriculum == "bandit" and avg_pg is not None and easy_loader is not None and medium_loader is not None:
                rewards[chosen].append(avg_pg)

            val_metrics = None
            if validate_filepath is not None and (epoch == halfway_epoch or epoch == epochs - 1):
                print(f"[DEBUG] Evaluating on validation set at epoch {epoch+1}")
                val_metrics = self.evaluator.evaluate(validate_filepath)[1]

                if epoch == halfway_epoch:
                    val_metrics_at_halfway = val_metrics

                if best_val_metrics is None or (val_metrics and val_metrics.get('roc_auc', 0) > best_val_metrics.get('roc_auc', 0)):
                    best_val_metrics = val_metrics
                    best_val_epoch = epoch + 1

            if val_metrics:
                print(f"Epoch {epoch+1} - Val Accuracy: {val_metrics['accuracy']:.4f} | "
                        f"Precision: {val_metrics['precision']:.4f} | "
                        f"Recall: {val_metrics['recall']:.4f} | "
                        f"AUC: {val_metrics['roc_auc']:.4f}")

        print("\n=== Best Epochs Summary ===")
        for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
            if best_metrics[metric] < best_val_metrics.get(metric, 0):
                best_metrics[metric] = best_val_metrics.get(metric, 0)
                best_epochs[metric] = best_val_epoch
                print(f"Best {metric.capitalize()}: {best_metrics[metric]:.4f} at epoch {best_epochs[metric]}")

        print(f"Best Loss: {best_epoch_loss:.4f}")

        best_metrics['loss'] = best_epoch_loss
        if best_val_metrics:
            best_metrics['val_auc'] = best_val_metrics.get('roc_auc', None)
            best_metrics['val_accuracy'] = best_val_metrics.get('accuracy', None)
            best_metrics['val_precision'] = best_val_metrics.get('precision', None)
            best_metrics['val_recall'] = best_val_metrics.get('recall', None)
            best_metrics['val_epoch'] = best_val_epoch
        if val_metrics_at_halfway:
            pass # Removed print statement

        return best_metrics 