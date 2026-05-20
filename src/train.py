"""
RoBERTa Fine-tunning with Focal Loss

This script fine-tunes roberta-base for email classification.
Uses Focal Loss to handle class imbalance.
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from data_loader import load_dataset
from sklearn.metrics import accuracy_score, f1_score

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Formula: FL = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the true class
    
    Args:
        alpha: Class weights (tensor of shape [num_classes])
        gamma:  Focusing parameter (default=2.0)
                Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none' (default='mean')
    
    Refernce: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - Raw logits from model
            targets: (batch_size,) - True class labels (0 to num_classes-1)
            
        Returns:
            loss: Scalar tensor
        """
        # Step 1: Compute standard cross-entropy loss
        cs_loss = F.cross_entropy(inputs, targets, reduction='None')
        # ce_loss shape: (batch_size,)
        
        # Step 2: Compute p_t (probability of true class)
        p_t = torch.exp(-cs_loss) # COnvert log-prob back to prob
        # p_t shape: (batch_size,)
        
        # Step 3: Apply focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        # focal_term shape: (batch_size,)
        
        # Step 4: Multiply to get focal loss
        focal_loss = focal_term * cs_loss
        # focal_loss shape: (batch-size,)
        
        # Step 5: Apply class weights (alpha)
        if self.alpha is not None:
            # Get alpha for each sample's true class
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Step 6: Reduction (mean, sum, or none)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EmailDataset(Dataset):
    """
    PyTorch Dataset for email classification.
    
    Converts text emails into token IDs that RoBERTa can process.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Args:
            texts: List of email texts (strings)
            labels: List of category labels (integers 0-19)
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length (truncate longer emails)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Returns total number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Fetches one sample.
        
        Args:
            idx: Index of sample to fetch (0 to len-1)
        
        Returns:
            Dictionary with inputs_ids, attention_mask, labels
        """
        # Get text and label for this index
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True, # Add [CLS], [SEP]
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def compute_class_weights(train_labels, num_classes):
        """
        Compute class weights for imbalanced dataset.
        
        Formula: weight_i = total_samples / (num_classes * count_i)
        Args:
            train_labels: Array of training labels
            num_classes: Total number of classes
        
        Returns:
            class_weights: Tensor of shape(num_classes,)
            
        Example:
            Class 0: 1000 samples -> weight = 10000 / (20 * 1000) = 0.5
            Class 1: 100 samples -> weight = 10000 / (20 * 100) = 5.0
            
            Rare classes get higher weights.
        """
        # Count samples per class
        class_counts = np.bincount(train_labels, minlength=num_classes)
        # class_counts = [1000, 100, 500, ...]
        
        total_samples = len(train_labels)
        
        # Inverse frequency weighting
        class_weights = total_samples / (num_classes * class_counts)
        
        # Convert to Pytorch tensor
        class_weights = torch.FloatTensor(class_weights)
        
        print(f"Class weights computed:")
        print(f"   Min weight: {class_weights.min():.2f} (most common class)")
        print(f"   Max weight: {class_weights.max():.2f} (rarest class)")
        print(f"   Mean weight: {class_weights.mean():.2f}")
        
        return class_weights
    
    def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
        """
        Train for one epoch.
        
        Args:
            model: RoBERTa model
            dataloader: DataLoader for training data
            optimizer: AdamW optimizer
            scheduler: Learning rate scheduler
            criterion: Focal Loss
            device: 'cude' or 'cpu'
        
        Returns:
            avg_loss, f1_macro, accuracy
        """
        model.train()
        
        total_loss = 0
        predictions = []
        true_labels = []
        
        # Progress bar for visualization
        progress_bar = tqdm(dataloader, desc='Training')
        
        for batch in progress_bar:
            # Step 1: Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # Shapes: (batch_size, max_length), (batch_size, max_length), (batch_size,)
            
            # Step 2: Zero out gradients form previous batch
            optimizer.zero_grad()
            
            # Step 3: forward pass - get model predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # outputs.logits shape: (batch_size, num_classes)
            
            # Step 4: Calculate loss
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            # Step 5: Backward pass - compute gradients
            loss.backward()
            
            # Step 6: Gradient cliping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step 7: Update weights
            optimizer.step()
            
            # Step 8: Update learning rate
            scheduler.step()
            
            # Step 9: Track predictions for metrics
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item})
        
        # Calculate metrics for each epoch
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        
        return avg_loss, accuracy, f1_macro
    
    def evaluate(model, dataloader, criterion, device):
        """
        Evaluate model on validation/test set.
        
        Args:
            model: RoBERTa model
            dataloader: DataLoader for validation/test data
            criterion: Focal Loss
            device: 'cuda' or 'cpu'
        
        Retruns:
            avg_loss, accuracy, f1_macro, predictions, true_labels
        """
        model.eval()
        
        total_loss = 0
        predictions = []
        true_labels = []
        
        # torch.no_grad() disables gradient calculation
        # saves memory and speeds up evaluation
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['lebels'].to(device)
                
                # Forward pass only - no backward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            avg_loss = total_loss / len(dataloader)
            accuracy = accuracy_score(true_labels, predictions)
            f1_macro = f1_score(true_labels, predictions, average='macro')
            
            return avg_loss, accuracy, f1_macro, predictions, true_labels