"""
Baseline Model for Email Classification

This serves as our baseline to comapre against RoBERTa.
Expected performance: ~80-82% accuracy.
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from data_loader import load_dataset

def train_baseline(train_df, val_df, max_features=10000):
    """
    Train TF-IDF + Logistic Regression baseline.
    
    Args:
        train_df: Training dataframe with 'text' and 'label' columns
        val_df: Validation dataframe
        max_features: Maximum number of TF-IDF features (words to consider)
    
    Returns:
        vectorizer, model, metrics
    """
    print("\nTraining TF-IDF + Logistic Regression Baseline")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: TF-IDF Vectorization
    print("\nStep 1: TF-IDF Vectorization")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )
    
    # Fit on training data
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    
    print(f"Created {X_train.shape[1]:,}features from {X_train.shape[0]:,} samples")
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    
    # Step 2: Train Logistic Regression
    print(f"\nStep 2: Train Logistic Regression")
    
    model = LogisticRegression(
        max_iter=100,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Training completed.")
    
    # Step 3: Evaluation
    print("\nStep 3: Evaluation on Validation Set")
    
    # Prediction
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    train_f1 = f1_score(y_train, train_preds, average="macro")
    val_f1 = f1_score(y_val, val_preds, average="macro")
    
    training_time = time.time() - start_time
    
    # Display results
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Training Macro-F1:   {train_f1:.4f}")
    print(f"")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Validation Macro-F1: {val_f1:.4f}")
    print(f"")
    print(f"Training Time:       {training_time:.2f} seconds")
    print("="*60)
    
    # Step 4: Save model
    print("\n Saving model")
    Path('models').mkdir(exist_ok=True)
    
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(model, 'models/logistic_regression.joblib')
    
    print("\n Model saved to models/")
    
    return vectorizer, model, {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'train_f1_macro': float(train_f1),
        'val_f1_macro': float(val_f1),
        'training_time': float(training_time)
    }

if __name__ == '__main__':
    print("="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    
    # Load data
    train_df, val_df, test_df, categories = load_dataset()
    
    # Train model
    vectorizer, model, metrics = train_baseline(train_df, val_df)
    
    print("\nBaseline training complete!")
    print(f"📌 Target to beat with RoBERTa: {metrics['val_f1_macro']:.4f} macro-F1")
    print("\nNext step: python src/train.py")