import torch
from torch.utils.data import Dataset
import numpy as np

class TextPairDataset(Dataset):
    def __init__(self, dataframe):
        self.name1 = dataframe['name1'].tolist()
        self.name2 = dataframe['name2'].tolist()
        self.label = dataframe['label'].tolist()

    def __len__(self):
        return len(self.name1)

    def __getitem__(self, idx):
        return self.name1[idx], self.name2[idx], torch.tensor(self.label[idx], dtype=torch.float32)

class TripletDataset(Dataset):
    def __init__(self, dataframe):
        self.anchor_data = dataframe['fraud_name'].tolist()
        self.positive_data = dataframe['real_name'].tolist()
        self.negative_data = dataframe['negative_name'].tolist()

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        return self.anchor_data[idx], self.positive_data[idx], self.negative_data[idx]

class SupConDataset(Dataset):
    def __init__(self, dataframe):
        """
        Dataset for SupCon that handles multiple positives and negatives per anchor.
        Each anchor should have exactly 3 positives and 3 negatives.
        
        Args:
            dataframe: DataFrame with columns for anchor, positives, and negatives
        """
        # Check available columns and use appropriate ones
        available_columns = list(dataframe.columns)
        
        # Handle different data formats
        if 'fraud_name' in available_columns and 'real_name' in available_columns:
            # This is fraud_triplets.pkl format - convert to SupCon format
            
            # Group by fraud_name to get multiple positives and negatives
            grouped = dataframe.groupby('fraud_name').agg({
                'real_name': list,
                'negative_name': list
            }).reset_index()
            
            self.anchor_data = grouped['fraud_name'].tolist()
            self.positive_data = grouped['real_name'].tolist()
            self.negative_data = grouped['negative_name'].tolist()
            
        elif 'anchor_name' in available_columns and 'positive_names' in available_columns:
            # This is supcon_triplets.pkl format with correct column names
            self.anchor_data = dataframe['anchor_name'].tolist()
            self.positive_data = dataframe['positive_names'].tolist()
            self.negative_data = dataframe['negative_names'].tolist()
            
        elif 'real_name' in available_columns and 'positive_names' in available_columns:
            # This is supcon_triplets.pkl format (alternative column names)
            self.anchor_data = dataframe['real_name'].tolist()
            self.positive_data = dataframe['positive_names'].tolist()
            self.negative_data = dataframe['negative_names'].tolist()
            
        elif 'name1' in available_columns and 'name2' in available_columns:
            # This is merged_data.pkl format - convert to SupCon format
            
            # Group by name1 to get multiple positives
            grouped = dataframe.groupby('name1').agg({
                'name2': list
            }).reset_index()
            
            self.anchor_data = grouped['name1'].tolist()
            self.positive_data = grouped['name2'].tolist()
            
            # Create negatives from other names
            all_names = set()
            for names in self.positive_data:
                all_names.update(names)
            
            self.negative_data = []
            for anchor in self.anchor_data:
                negatives = list(all_names - set([anchor]))
                self.negative_data.append(negatives[:7])  # Take first 7 negatives
                
        else:
            # Try different possible column names for anchor
            anchor_col = None
            for col in ['anchor_name', 'real_name', 'anchor_name', 'fraud_name', 'name1', 'anchor']:
                if col in available_columns:
                    anchor_col = col
                    break
            
            if anchor_col is None:
                raise ValueError(f"Could not find anchor column. Available columns: {available_columns}")
            
            # Try different possible column names for positives
            positive_col = None
            for col in ['positive_names', 'real_name', 'positive_name', 'name2', 'positives']:
                if col in available_columns:
                    positive_col = col
                    break
            
            if positive_col is None:
                raise ValueError(f"Could not find positive column. Available columns: {available_columns}")
            
            # Try different possible column names for negatives
            negative_col = None
            for col in ['negative_names', 'negative_name', 'negatives']:
                if col in available_columns:
                    negative_col = col
                    break
            
            if negative_col is None:
                raise ValueError(f"Could not find negative column. Available columns: {available_columns}")
            
            self.anchor_data = dataframe[anchor_col].tolist()
            self.positive_data = dataframe[positive_col].tolist()
            self.negative_data = dataframe[negative_col].tolist()

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        """Returns anchor, exactly 3 positives and 7 negatives"""
        anchor = self.anchor_data[idx]
        positives = self.positive_data[idx]
        negatives = self.negative_data[idx]
        
        # Ensure exactly 3 positives
        if len(positives) > 3:
            positives = positives[:3]
        elif len(positives) < 3:
            # Pad with the first positive if we have fewer than 3
            if len(positives) > 0:
                positives = positives + [positives[0]] * (3 - len(positives))
            else:
                # If no positives, use the anchor as positive
                positives = [anchor] * 3
        
        # Ensure exactly 7 negatives
        if len(negatives) > 7:
            negatives = negatives[:7]
        elif len(negatives) < 7:
            # Pad with the first negative if we have fewer than 7
            if len(negatives) > 0:
                negatives = negatives + [negatives[0]] * (7 - len(negatives))
            else:
                # If no negatives, use the anchor as negative (fallback)
                negatives = [anchor] * 7
            
        return anchor, positives, negatives

class InfoNCEDataset(Dataset):
    def __init__(self, dataframe):
        """
        Dataset for InfoNCE that handles one positive and multiple negatives per anchor.
        Each anchor should have exactly 1 positive and 3 negatives.
        
        Args:
            dataframe: DataFrame with columns 'fraud_name' (anchor), 'real_name' (positive),
                      'negative_name' (list of negatives)
        """
        self.anchor_data = dataframe['anchor_name'].tolist()
        self.positive_data = dataframe['positive_name'].tolist()
        self.negative_data = dataframe['negative_names'].tolist()  # Should be list of lists

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        """Returns anchor, one positive, and exactly 3 negatives"""
        anchor = self.anchor_data[idx]
        positive = self.positive_data[idx]
        negatives = self.negative_data[idx][:3]  # Take first 3 negatives
        
        # Pad if necessary
        if len(negatives) < 3:
            negatives = negatives + [negatives[0]] * (3 - len(negatives))
            
        return anchor, positive, negatives
