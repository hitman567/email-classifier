# Email Classification System with RoBERTa

Production-ready email classifier using fine-tuned RoBERTa for multi-class topic classification.

## 🎯 Project Overview

This project replicates a production document-intelligence pipeline that:
- Classifies emails into 20+ topic categories
- Uses RoBERTa-base fine-tuned with class-balanced focal loss
- Achieves 90%+ macro-F1 score
- Includes baseline comparison (TF-IDF + Logistic Regression)

## 📊 Performance Metrics

| Model | Accuracy | Macro-F1 | Training Time |
|-------|----------|----------|---------------|
| TF-IDF + LogReg (baseline) | TBD | TBD | TBD |
| RoBERTa-base (fine-tuned) | TBD | TBD | TBD |

## 🛠️ Tech Stack

- **Model**: RoBERTa-base (Hugging Face)
- **Framework**: PyTorch, Transformers
- **Data Processing**: Pandas, scikit-learn
- **UI**: Gradio
- **Deployment**: Hugging Face Spaces

## 📁 Project Structure

```
email-classifier/
├── data/                  # Dataset files
│   ├── raw/              # Original data
│   ├── processed/        # Cleaned & split data
│   └── README.md         # Data documentation
├── notebooks/            # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_roberta_finetuning.ipynb
├── src/                  # Source code
│   ├── data_loader.py   # Data loading utilities
│   ├── baseline.py      # TF-IDF baseline
│   ├── train.py         # RoBERTa training script
│   ├── evaluate.py      # Evaluation metrics
│   └── inference.py     # Prediction pipeline
├── models/              # Saved models
├── outputs/             # Training logs, metrics
├── app.py               # Gradio demo app
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/data_loader.py
```

### 3. Train Baseline
```bash
python src/baseline.py
```

### 4. Fine-tune RoBERTa
```bash
python src/train.py
```

### 5. Launch Demo
```bash
python app.py
```

## 📈 Training Details

- **Base Model**: `roberta-base` (125M parameters)
- **Dataset**: 20 Newsgroups (subset for email-like classification)
- **Loss Function**: Focal Loss (gamma=2.0, class weights)
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Batch Size**: 16
- **Epochs**: 3-5
- **Hardware**: Single GPU (Tesla T4 or better)

## 📝 Results

Coming soon after training completion.

## 🎓 Learning Outcomes

By building this project, you'll learn:
- ✅ Fine-tuning transformer models (RoBERTa)
- ✅ Handling class imbalance with focal loss
- ✅ Building ML baselines (TF-IDF + LogReg)
- ✅ Evaluation metrics (macro-F1, precision, recall)
- ✅ Creating interactive demos with Gradio
- ✅ Model deployment on HuggingFace Spaces

## 📚 References

- RoBERTa Paper: https://arxiv.org/abs/1907.11692
- Focal Loss: https://arxiv.org/abs/1708.02002
- Hugging Face Transformers: https://huggingface.co/docs/transformers

## 🔗 Links

- **Demo**: [Coming Soon]
- **Model Weights**: [Coming Soon]
- **Blog Post**: [Coming Soon]

---

Built by Ayush Kumar Tiwary | [LinkedIn](https://linkedin.com/in/ayushkumartiwary) | [GitHub](https://github.com/hitman567)
