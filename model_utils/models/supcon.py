from .base import BaseSiameseCLIP
from utils.data import SupConDataset
from torch.utils.data import DataLoader
import torch

class SiameseCLIPSupCon(BaseSiameseCLIP):
    """
    Siamese network for SupCon learning using CLIP as backbone.
    Handles multiple positive and negative examples per anchor.
    """
    def forward(self, anchor_text, positive_texts, negative_texts):
        """
        Forward pass for SupCon learning (batched).
        Args:
            anchor_text: list of anchor text strings, len=batch_size
            positive_texts: list of list of positive text strings, shape [batch_size, n_positives]
            negative_texts: list of list of negative text strings, shape [batch_size, n_negatives]
        Returns:
            tuple: (z_anchor, z_positives, z_negatives) embeddings
        """
        # Encode anchors
        z_anchor = self.encode(anchor_text)  # [batch_size, emb_dim]
        
        # Encode positives - ensure all have the same number
        batch_size = len(positive_texts)
        n_positives = len(positive_texts[0]) if positive_texts else 0
        
        # Validate that all positive lists have the same length
        for i, pos_list in enumerate(positive_texts):
            if len(pos_list) != n_positives:
                raise ValueError(f"All positive lists must have the same length. Expected {n_positives}, got {len(pos_list)} at index {i}")
        
        z_positives = torch.stack([
            self.encode(pos_list) for pos_list in positive_texts
        ], dim=0)  # [batch_size, n_positives, emb_dim]
        
        # Encode negatives - ensure all have the same number
        n_negatives = len(negative_texts[0]) if negative_texts else 0
        
        # Validate that all negative lists have the same length
        for i, neg_list in enumerate(negative_texts):
            if len(neg_list) != n_negatives:
                raise ValueError(f"All negative lists must have the same length. Expected {n_negatives}, got {len(neg_list)} at index {i}")
        
        z_negatives = torch.stack([
            self.encode(neg_list) for neg_list in negative_texts
        ], dim=0)  # [batch_size, n_negatives, emb_dim]
        
        return z_anchor, z_positives, z_negatives
