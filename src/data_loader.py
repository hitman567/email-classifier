"""
Data Loader for Email Classification
Downloads and prepare 20 Newsgroups dataset for training and testing
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

def download_dataset(data_dir='data'):
    """Download and prepare the 20 Newsgroups dataset"""
    
    print("Downloading 20 Newsgroups dataset...")
    
    # Download train and test datasets
    train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    # Get category names
    categories = train_data.target_names
    
    print(f"Downloaded {len(train_data.data)} training samples")
    print(f"Downloaded {len(test_data.data)} testing samples")
    print(f"Categories ({len(categories)}): {categories[:5]}...")
    
    # Combine train and test data for easier processing
    all_texts = train_data.data + test_data.data
    all_labels = list(train_data.target) + list(test_data.target)
    all_categories = [categories[label] for label in all_labels]
    
    df = pd.DataFrame({
        'text': all_texts,
        'label': all_labels,
        'category': all_categories
    })
    
    # Clean text - remove very short samples
    df = df[df['text'].str.len() > 50].reset_index(drop=True)
    
    print(f"\n Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Avg text length: {df['text'].str.len().mean():.0f} characters")
    
    # Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
    )
    
    print(f"\n📁 Split sizes:")
    print(f"   Train: {len(train_df)}")
    print(f"   Val:   {len(val_df)}")
    print(f"   Test:  {len(test_df)}")
    
    # Save to disk
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(data_path / 'processed' / 'train.csv', index=False)
    val_df.to_csv(data_path / 'processed' / 'val.csv', index=False)
    test_df.to_csv(data_path / 'processed' / 'test.csv', index=False)
    
    # Save category mapping
    category_mapping = {i: cat for i, cat in enumerate(categories)}
    with open(data_path / 'processed' / 'category_mapping.json', 'w') as f:
        json.dump(category_mapping, f, indent=2)
    
    print(f"Data saved to {data_path / 'processed'}")
    
    return train_df, val_df, test_df, categories

def load_dataset(data_dir='data'):
    """Load prepared datasets from disk"""
    data_path = Path(data_dir) / 'processed'
    
    if not data_path.exists():
        print("❌ Processed data not found. Run download_dataset() first.")
        return None, None, None, None
    
    train_df = pd.read_csv(data_path / 'train.csv')
    val_df = pd.read_csv(data_path / 'val.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    
    with open(data_path / 'category_mapping.json', 'r') as f:
        category_mapping = json.load(f)
    
    categories = [category_mapping[str(i)] for i in range(len(category_mapping))]
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    print(f"   Categories: {len(categories)}")
    
    return train_df, val_df, test_df, categories
    

if __name__ == '__main__':
    # This runs only when executing the script directly
    print("="*60)
    print("EMAIL CLASSIFIER - DATA PREPARATION")
    print("="*60)
    
    train_df, val_df, test_df, categories = download_dataset()
    
    print("\n" + "="*60)
    print("Dataset ready!")
    print("="*60)