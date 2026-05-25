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
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.data_loader import load_dataset
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
        cs_loss = F.cross_entropy(inputs, targets, reduction='none')
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
        progress_bar.set_postfix({'loss': loss.item()})
    
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
            labels = batch['labels'].to(device)
            
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
    
def train_roberta(train_df, val_df, num_classes, epochs=3, batch_size=16, learning_rate=2e-5, max_length=256, gamma=2.0):
    """
    Fine-tune RoBERTa with Focal Loss.
    
    Args:
        train_df, val_df: DataFrames with 'text' and 'label' columns
        num_classes: Number of categories (20)
        epochs: Training epochs (3-5 is typical)
        batch_size: Batch size (16 for GPU, 8 for smaller GPUs)
        learning_rate: Learning rate (2e-5 is standard for RoBERTa)
        max_length: Max sequence length (256 is good for emails)
        gamma: Focal loss gamma (2.0 is standard)
    
    Returns:
        model, tokenizer, training_history
    """
    print("Starting RoBERTa Fine-tuning")
    print("="*60)
    
    # Step 1: Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Step 2: Load tokenizer and model
    print("\nLoading roberta-base...")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_classes
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 3: Create Datasets and Dataloaders
    print("\nCreating Datasets and Dataloaders")
    
    train_dataset = EmailDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        max_length=max_length
    )
    val_dataset = EmailDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0 # 0 for Mac/Windows, 2-4 for Linux
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False        # Don't shuffle validation
    )
    
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Val samples: {len(val_dataset):,}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Step 4: Setup optimzer, scheduler
    print("\nSetting up optimizer...")
    
    # AdamW optimizer (Adam with weight decay)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"   Optimizer: AdamW")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: 0.01")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps:,}")
    
    # Step 5: Setup Focal Loss
    print("\nSetting up Focal Loss...")
    
    # Compute class weights
    class_weights = compute_class_weights(
        train_df['label'].values,
        num_classes
    )
    class_weights = class_weights.to(device)
    
    # Create Focal Loss
    criterion = FocalLoss(alpha=class_weights, gamma=gamma)
    
    print(f"Focal Loss gamma: {gamma}")
    
    # Step 6: Training loop
    print("\n Training Starts...")
    
    best_f1 = 0
    training_history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1}/{epochs}")
        
        # train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        
        # Evaluate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Print results
        print(f"\n Results:")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val  -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            
            # Create models directory
            Path('models').mkdir(exist_ok=True)
            
            # Save modelc checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }
            torch.save(checkpoint, 'models/roberta_best.pt')
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'train_f1_macro': float(train_f1),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'val_f1_macro': float(val_f1)
        })
    
        training_time = time.time() - start_time
        
        # Step 7: Save Final Artifacts
        print("\nSaving artifacts...")
        
        # Save tokenizer
        tokenizer.save_pretrained('models/roberta_tokenizer')
        print("\nTokenizer Saved")
        
        # Save training history
        Path('outputs').mkdir(exist_ok=True)
        with open('outputs/roberta_training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        print("\nTraining history saved")
        
        # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Models saved to: models/")
    print(f"History saved to: outputs/roberta_training_history.json")
    
    return model, tokenizer, training_history

if __name__ == '__main__':
    print("="*60)
    print(" ROBERTA FINE_TUNING FOR EMAIL CLASSIFICATION")
    print("="*60)
    
    # Load data
    print("\nLoading dataset...")
    train_df, val_df, test_df, categories = load_dataset()
    
    if train_df is None:
        print("Data not found. Run: python src/data_loader.py")
        exit(1)
    
    num_classes = len(categories)
    print(f"Loaded {num_classes} categories")
    
    # Train model
    model, tokenizer, history = train_roberta(
        train_df, 
        val_df,
        num_classes=num_classes,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    print("\nTraining complete!")
    print("\nNext steps:")
    print("   1. Check outputs/roberta_training_history.json")
    print("   2. Run: python app.py (launch demo)")
    print("   3. Compare with baseline results")

        