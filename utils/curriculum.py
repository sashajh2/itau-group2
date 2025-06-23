import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import random
from collections import defaultdict
import math


class CurriculumBase(ABC):
    """Base class for curriculum learning strategies."""
    
    def __init__(self, dataset, mode="pair"):
        self.dataset = dataset
        self.mode = mode
        self.current_epoch = 0
        self.sample_weights = None
        self.difficulty_scores = None
        
    @abstractmethod
    def get_curriculum_batch(self, batch_size: int, epoch: int) -> Tuple[List, List]:
        """Get a batch of samples according to curriculum strategy."""
        pass
    
    @abstractmethod
    def update_curriculum(self, epoch: int, losses: List[float], indices: List[int]):
        """Update curriculum based on training progress."""
        pass
    
    def compute_difficulty_scores(self) -> np.ndarray:
        """Compute difficulty scores for all samples."""
        if self.difficulty_scores is not None:
            return self.difficulty_scores
            
        if self.mode == "pair":
            return self._compute_pair_difficulty()
        elif self.mode == "triplet":
            return self._compute_triplet_difficulty()
        elif self.mode in ["supcon", "infonce"]:
            return self._compute_contrastive_difficulty()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _compute_pair_difficulty(self) -> np.ndarray:
        """Compute difficulty for pair samples based on text similarity."""
        difficulties = []
        
        for i in range(len(self.dataset)):
            name1, name2, label = self.dataset[i]
            
            # Simple difficulty based on string similarity
            if isinstance(name1, str) and isinstance(name2, str):
                # Length difference as difficulty proxy
                length_diff = abs(len(name1) - len(name2))
                # Character overlap
                char_overlap = len(set(name1.lower()) & set(name2.lower()))
                total_chars = len(set(name1.lower()) | set(name2.lower()))
                overlap_ratio = char_overlap / total_chars if total_chars > 0 else 0
                
                # Higher difficulty for similar length but different labels
                if label == 0:  # Different entities
                    difficulty = length_diff + (1 - overlap_ratio) * 10
                else:  # Same entity
                    difficulty = abs(length_diff) + overlap_ratio * 5
            else:
                difficulty = 1.0
                
            difficulties.append(difficulty)
            
        return np.array(difficulties)
    
    def _compute_triplet_difficulty(self) -> np.ndarray:
        """Compute difficulty for triplet samples."""
        difficulties = []
        
        for i in range(len(self.dataset)):
            anchor, positive, negative = self.dataset[i]
            
            # Difficulty based on anchor-positive vs anchor-negative similarity
            if isinstance(anchor, str) and isinstance(positive, str) and isinstance(negative, str):
                # Anchor-positive similarity
                anchor_pos_overlap = len(set(anchor.lower()) & set(positive.lower()))
                anchor_pos_total = len(set(anchor.lower()) | set(positive.lower()))
                anchor_pos_sim = anchor_pos_overlap / anchor_pos_total if anchor_pos_total > 0 else 0
                
                # Anchor-negative similarity
                anchor_neg_overlap = len(set(anchor.lower()) & set(negative.lower()))
                anchor_neg_total = len(set(anchor.lower()) | set(negative.lower()))
                anchor_neg_sim = anchor_neg_overlap / anchor_neg_total if anchor_neg_total > 0 else 0
                
                # Difficulty: harder when anchor-positive is low and anchor-negative is high
                difficulty = (1 - anchor_pos_sim) + anchor_neg_sim
            else:
                difficulty = 1.0
                
            difficulties.append(difficulty)
            
        return np.array(difficulties)
    
    def _compute_contrastive_difficulty(self) -> np.ndarray:
        """Compute difficulty for contrastive learning samples."""
        difficulties = []
        
        for i in range(len(self.dataset)):
            if self.mode == "supcon":
                anchor, positives, negatives = self.dataset[i]
                # Average difficulty across all positive and negative pairs
                pos_diffs = []
                neg_diffs = []
                
                for pos in positives[:3]:  # Take first 3 positives
                    if isinstance(anchor, str) and isinstance(pos, str):
                        overlap = len(set(anchor.lower()) & set(pos.lower()))
                        total = len(set(anchor.lower()) | set(pos.lower()))
                        sim = overlap / total if total > 0 else 0
                        pos_diffs.append(1 - sim)
                
                for neg in negatives[:3]:  # Take first 3 negatives
                    if isinstance(anchor, str) and isinstance(neg, str):
                        overlap = len(set(anchor.lower()) & set(neg.lower()))
                        total = len(set(anchor.lower()) | set(neg.lower()))
                        sim = overlap / total if total > 0 else 0
                        neg_diffs.append(sim)
                
                difficulty = np.mean(pos_diffs) + np.mean(neg_diffs) if pos_diffs and neg_diffs else 1.0
            else:  # infonce
                anchor, positive, negatives = self.dataset[i]
                # Similar to triplet but with multiple negatives
                if isinstance(anchor, str) and isinstance(positive, str):
                    anchor_pos_overlap = len(set(anchor.lower()) & set(positive.lower()))
                    anchor_pos_total = len(set(anchor.lower()) | set(positive.lower()))
                    anchor_pos_sim = anchor_pos_overlap / anchor_pos_total if anchor_pos_total > 0 else 0
                    
                    neg_sims = []
                    for neg in negatives[:3]:
                        if isinstance(neg, str):
                            overlap = len(set(anchor.lower()) & set(neg.lower()))
                            total = len(set(anchor.lower()) | set(neg.lower()))
                            sim = overlap / total if total > 0 else 0
                            neg_sims.append(sim)
                    
                    avg_neg_sim = np.mean(neg_sims) if neg_sims else 0
                    difficulty = (1 - anchor_pos_sim) + avg_neg_sim
                else:
                    difficulty = 1.0
                    
            difficulties.append(difficulty)
            
        return np.array(difficulties)


class ManualCurriculum(CurriculumBase):
    """Manual curriculum with predefined difficulty progression."""
    
    def __init__(self, dataset, mode="pair", difficulty_thresholds=None):
        super().__init__(dataset, mode)
        self.difficulty_scores = self.compute_difficulty_scores()
        
        # Default difficulty thresholds (percentiles)
        if difficulty_thresholds is None:
            self.difficulty_thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]  # Progressive inclusion
        else:
            self.difficulty_thresholds = difficulty_thresholds
            
        self.current_threshold_idx = 0
        
    def get_curriculum_batch(self, batch_size: int, epoch: int) -> Tuple[List, List]:
        """Get batch with manual difficulty progression."""
        # Update threshold based on epoch
        threshold_idx = min(epoch // 5, len(self.difficulty_thresholds) - 1)  # Change every 5 epochs
        threshold = np.percentile(self.difficulty_scores, 
                                self.difficulty_thresholds[threshold_idx] * 100)
        
        # Filter samples below threshold
        easy_indices = np.where(self.difficulty_scores <= threshold)[0]
        
        if len(easy_indices) < batch_size:
            # If not enough easy samples, include harder ones
            remaining_indices = np.where(self.difficulty_scores > threshold)[0]
            easy_indices = np.concatenate([easy_indices, remaining_indices[:batch_size - len(easy_indices)]])
        
        # Sample from easy indices
        selected_indices = np.random.choice(easy_indices, 
                                          size=min(batch_size, len(easy_indices)), 
                                          replace=False)
        
        batch_data = [self.dataset[i] for i in selected_indices]
        return batch_data, selected_indices.tolist()
    
    def update_curriculum(self, epoch: int, losses: List[float], indices: List[int]):
        """Manual curriculum doesn't update based on losses."""
        self.current_epoch = epoch


class SelfPacedCurriculum(CurriculumBase):
    """Self-paced curriculum that adapts based on model performance."""
    
    def __init__(self, dataset, mode="pair", lambda_param=1.0, alpha=0.1):
        super().__init__(dataset, mode)
        self.difficulty_scores = self.compute_difficulty_scores()
        self.lambda_param = lambda_param  # Controls curriculum pace
        self.alpha = alpha  # Learning rate for lambda
        self.sample_weights = np.ones(len(dataset))
        self.loss_history = defaultdict(list)
        
    def get_curriculum_batch(self, batch_size: int, epoch: int) -> Tuple[List, List]:
        """Get batch using self-paced sampling."""
        # Update sample weights based on difficulty and loss history
        self._update_sample_weights()
        
        # Sample based on weights
        weights = self.sample_weights / np.sum(self.sample_weights)
        selected_indices = np.random.choice(len(self.dataset), 
                                          size=batch_size, 
                                          replace=False, 
                                          p=weights)
        
        batch_data = [self.dataset[i] for i in selected_indices]
        return batch_data, selected_indices.tolist()
    
    def update_curriculum(self, epoch: int, losses: List[float], indices: List[int]):
        """Update curriculum based on training losses."""
        # Store losses for each sample
        for idx, loss in zip(indices, losses):
            self.loss_history[idx].append(loss)
        
        # Update lambda parameter
        if epoch > 0 and epoch % 5 == 0:  # Update every 5 epochs
            avg_loss = np.mean([np.mean(self.loss_history[idx]) for idx in indices if self.loss_history[idx]])
            if avg_loss < 0.1:  # If loss is low, increase difficulty
                self.lambda_param *= (1 + self.alpha)
            else:  # If loss is high, decrease difficulty
                self.lambda_param *= (1 - self.alpha)
            
            self.lambda_param = max(0.1, min(10.0, self.lambda_param))  # Clamp between 0.1 and 10
    
    def _update_sample_weights(self):
        """Update sample weights based on difficulty and lambda."""
        # Self-paced learning formula: w = 1 if loss < lambda, 0 otherwise
        for i in range(len(self.dataset)):
            if i in self.loss_history and self.loss_history[i]:
                avg_loss = np.mean(self.loss_history[i])
                if avg_loss < self.lambda_param:
                    self.sample_weights[i] = 1.0
                else:
                    self.sample_weights[i] = 0.1  # Small weight instead of 0 for exploration
            else:
                # For unseen samples, weight based on difficulty
                difficulty = self.difficulty_scores[i]
                self.sample_weights[i] = np.exp(-difficulty / self.lambda_param)


class BanditCurriculum(CurriculumBase):
    """Multi-armed bandit curriculum for adaptive sample selection."""
    
    def __init__(self, dataset, mode="pair", exploration_rate=0.1, window_size=100):
        super().__init__(dataset, mode)
        self.difficulty_scores = self.compute_difficulty_scores()
        self.exploration_rate = exploration_rate
        self.window_size = window_size
        
        # Bandit parameters
        self.arm_rewards = defaultdict(list)  # Rewards for each sample
        self.arm_counts = defaultdict(int)    # Number of times each sample was selected
        self.arm_values = defaultdict(float)  # Estimated value of each sample
        
        # Initialize arms
        for i in range(len(dataset)):
            self.arm_values[i] = 0.0
    
    def get_curriculum_batch(self, batch_size: int, epoch: int) -> Tuple[List, List]:
        """Get batch using bandit selection."""
        selected_indices = []
        
        for _ in range(batch_size):
            # Epsilon-greedy strategy
            if random.random() < self.exploration_rate:
                # Exploration: select random sample
                idx = random.randint(0, len(self.dataset) - 1)
            else:
                # Exploitation: select best sample
                idx = max(self.arm_values.keys(), key=lambda k: self.arm_values[k])
            
            selected_indices.append(idx)
            self.arm_counts[idx] += 1
        
        batch_data = [self.dataset[i] for i in selected_indices]
        return batch_data, selected_indices
    
    def update_curriculum(self, epoch: int, losses: List[float], indices: List[int]):
        """Update bandit arms based on training losses."""
        # Convert losses to rewards (lower loss = higher reward)
        max_loss = max(losses) if losses else 1.0
        rewards = [1.0 - (loss / max_loss) for loss in losses]
        
        # Update arm values using moving average
        for idx, reward in zip(indices, rewards):
            self.arm_rewards[idx].append(reward)
            
            # Keep only recent rewards
            if len(self.arm_rewards[idx]) > self.window_size:
                self.arm_rewards[idx] = self.arm_rewards[idx][-self.window_size:]
            
            # Update estimated value
            if self.arm_rewards[idx]:
                self.arm_values[idx] = np.mean(self.arm_rewards[idx])
        
        # Adjust exploration rate
        if epoch > 0 and epoch % 10 == 0:
            self.exploration_rate = max(0.01, self.exploration_rate * 0.95)  # Decay exploration


class CurriculumDataLoader:
    """DataLoader wrapper that implements curriculum learning."""
    
    def __init__(self, dataset, curriculum_strategy, batch_size, mode="pair"):
        self.dataset = dataset
        self.curriculum = curriculum_strategy
        self.batch_size = batch_size
        self.mode = mode
        self.current_epoch = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        """Get next batch using curriculum strategy."""
        batch_data, indices = self.curriculum.get_curriculum_batch(self.batch_size, self.current_epoch)
        
        # Convert to tensors based on mode
        if self.mode == "pair":
            name1_list, name2_list, label_list = zip(*batch_data)
            return name1_list, name2_list, torch.tensor(label_list, dtype=torch.float32), indices
        elif self.mode == "triplet":
            anchor_list, positive_list, negative_list = zip(*batch_data)
            return anchor_list, positive_list, negative_list, indices
        elif self.mode == "supcon":
            anchor_list, positive_list, negative_list = zip(*batch_data)
            return anchor_list, positive_list, negative_list, indices
        elif self.mode == "infonce":
            anchor_list, positive_list, negative_list = zip(*batch_data)
            return anchor_list, positive_list, negative_list, indices
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def update_curriculum(self, losses, indices):
        """Update curriculum with training losses."""
        self.curriculum.update_curriculum(self.current_epoch, losses, indices)
    
    def set_epoch(self, epoch):
        """Set current epoch for curriculum."""
        self.current_epoch = epoch
