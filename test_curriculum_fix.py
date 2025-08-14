#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import numpy as np

# Mock dataset classes for testing
class MockInfoNCEDataset(Dataset):
    def __init__(self, name, num_negatives):
        self.name = name
        self.num_negatives = num_negatives
        self.data = [
            (f"{name}_anchor_{i}", f"{name}_positive_{i}", 
             [f"{name}_negative_{i}_{j}" for j in range(num_negatives)])
            for i in range(10)
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def curriculum_collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    
    # Find the maximum number of negatives in this batch
    max_negatives = max(len(neg_list) for neg_list in negatives)
    
    # Pad all negative lists to the same length
    padded_negatives = []
    for neg_list in negatives:
        if len(neg_list) < max_negatives:
            # Pad by repeating the first negative
            padded_neg_list = neg_list + [neg_list[0]] * (max_negatives - len(neg_list))
        else:
            padded_neg_list = neg_list
        padded_negatives.append(padded_neg_list)
    
    return list(anchors), list(positives), padded_negatives

def test_curriculum_collate():
    print("Testing curriculum collate function...")
    
    # Create datasets with different negative counts
    easy_dataset = MockInfoNCEDataset("easy", 4)
    medium_dataset = MockInfoNCEDataset("medium", 5)
    hard_dataset = MockInfoNCEDataset("hard", 5)
    
    print(f"Easy dataset: {len(easy_dataset)} samples, {easy_dataset.num_negatives} negatives each")
    print(f"Medium dataset: {len(medium_dataset)} samples, {medium_dataset.num_negatives} negatives each")
    print(f"Hard dataset: {len(hard_dataset)} samples, {hard_dataset.num_negatives} negatives each")
    
    # Create mixed dataset
    mixed_dataset = ConcatDataset([easy_dataset, medium_dataset, hard_dataset])
    
    # Create dataloader with custom collate function
    loader = DataLoader(mixed_dataset, batch_size=4, shuffle=True, collate_fn=curriculum_collate_fn)
    
    # Test a few batches
    for i, batch in enumerate(loader):
        anchors, positives, negatives = batch
        print(f"\nBatch {i+1}:")
        print(f"  Anchors: {len(anchors)}")
        print(f"  Positives: {len(positives)}")
        print(f"  Negatives: {len(negatives)} lists")
        
        # Check that all negative lists have the same length
        neg_lengths = [len(neg_list) for neg_list in negatives]
        print(f"  Negative list lengths: {neg_lengths}")
        
        if len(set(neg_lengths)) != 1:
            print("  ERROR: Not all negative lists have the same length!")
            return False
        else:
            print(f"  SUCCESS: All negative lists have length {neg_lengths[0]}")
        
        if i >= 2:  # Test first 3 batches
            break
    
    print("\nCurriculum collate function test PASSED!")
    return True

if __name__ == "__main__":
    test_curriculum_collate()
