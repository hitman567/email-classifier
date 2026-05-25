"""
Minimal test to verify train.py works.
Uses only 100 samples, 1 epoch - should finish in ~2 minutes on CPU.
"""

import pandas as pd
from src.train import train_roberta

# Load just 100 samples for testing
train_df = pd.read_csv('data/processed/train.csv').head(100)
val_df = pd.read_csv('data/processed/val.csv').head(50)

print("🧪 SMOKE TEST - Training on 100 samples, 1 epoch")
print("=" * 60)

# Run minimal training
model, tokenizer, history = train_roberta(
    train_df,
    val_df,
    num_classes=20,
    epochs=1,           # Just 1 epoch
    batch_size=8,       # Small batch
    learning_rate=2e-5,
    max_length=128      # Shorter sequences = faster
)

print("\n✅ SMOKE TEST PASSED!")
print("=" * 60)
print("Your train.py is working correctly!")
print("\nNext: Run full training with:")
print("  python src/train.py")