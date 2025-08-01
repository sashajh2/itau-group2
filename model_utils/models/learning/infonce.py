from ..base import BaseSiameseModel
from utils.data import InfoNCEDataset
from torch.utils.data import DataLoader
import torch

class SiameseModelInfoNCE(BaseSiameseModel):
    """
    Siamese network for InfoNCE learning using any vision-language model as backbone.
    Handles one positive and multiple negative examples per anchor.
    """
    def __init__(self, embedding_dim=512, projection_dim=128, backbone=None):
        super().__init__(embedding_dim, projection_dim, backbone)
    
    def forward(self, anchor_text, positive_text, negative_texts):
        """
        Forward pass for InfoNCE learning (batched).
        Args:
            anchor_text: list of anchor text strings, len=batch_size
            positive_text: list of positive text strings, len=batch_size
            negative_texts: tuple of lists or list of tuples of negative text strings
        Returns:
            tuple: (z_anchor, z_positives, z_negatives) embeddings
        """
        # Encode anchors
        z_anchor = self.encode(anchor_text)  # [batch_size, emb_dim]
        # Encode positives
        z_positive = self.encode(positive_text)  # [batch_size, emb_dim]
        z_positives = z_positive.unsqueeze(1)  # [batch_size, 1, emb_dim]
        # negatives: tuple of lists, shape [batch_size, n_negatives] or list of tuples [n_negatives, batch_size]
        # Transpose if needed
        if isinstance(negative_texts, tuple) and isinstance(negative_texts[0], list):
            negative_texts = list(zip(*negative_texts))
        batch_size = len(anchor_text)
        n_negatives = len(negative_texts)
        flat_negatives = []
        for neg_batch in negative_texts:
            flat_negatives.extend(neg_batch)
        z_negatives_flat = self.encode(flat_negatives)
        z_negatives = z_negatives_flat.view(n_negatives, batch_size, -1).transpose(0, 1)
        return z_anchor, z_positives, z_negatives
