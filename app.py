"""
Gradio Demo for Email Classification
Interactive web interface comparing baseline vs RoBERTa Predections
"""
import gradio as gr
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import joblib
import json
from pathlib import Path
import numpy as np

def load_models():
    """Load both trained models:"""
    print("Loading models...")
    
    # Load categories mapping
    with open ('data/processed/category_mapping.json', 'r') as f:
        categories_dict = json.load(f)
    categories = [categories_dict[str(i)] for i in range(len(categories_dict))]
    
    # Load baseline (TF-IDF + LogReg)
    try:
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        logistic_model = joblib.load('models/logistic_regression.joblib')
        baseline_loaded = True
        print("Baseline model loaded")
    except Exception as e:
        baseline_loaded = False
        tfidf_vectorizer = None
        logistic_model = None
        print(f"Error loading baseline model: {e}")
    
    # Load RoBERTa model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = RobertaTokenizer.from_pretrained('models/roberta_tokenizer')
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=len(categories)
        )
        
        # Load checkpoint
        checkpoint = torch.load('models/roberta_best.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        roberta_loaded = True
        print("RoBERTa model loaded")
        
    except Exception as e:
        roberta_loaded = False
        tokenizer = None
        model = None
        device = None
        print(f"Error loading RoBERTa model: {e}")
    
    return {
        'categories': categories,
        'baseline_loaded': baseline_loaded,
        'tfidf_vectorizer': tfidf_vectorizer,
        'logistic_model': logistic_model,
        'roberta_loaded': roberta_loaded,
        'roberta_model': model,
        'roberta_tokenizer': tokenizer,
        'device': device
    }

# Load models once at startup
models = load_models()


def predict_baseline(text):
    """Predict using TF-IDF + LogReg baseline."""
    if not models['baseline_loaded']:
        return "Baseline model not available"
    
    # Transform text
    X = models['tfidf_vectorizer'].transform([text])
    
    # Predict
    pred_label = models['logistic_model'].predict(X)[0]
    pred_proba = models['logistic_model'].predict_proba(X)[0]
    
    # Get top 5 predictions
    top5_indices = np.argsort(pred_proba)[-5:][::-1]
    
    results = {}
    for idx in top5_indices:
        category = models['categories'][idx]
        confidence = float(pred_proba[idx])
        results[category] = confidence
    
    return results

def predict_roberta(text):
    """Predict using fine-tuned RoBERTa."""
    if not models['roberta_loaded']:
        return "⚠️ RoBERTa model not available"
    
    # Tokenize
    encoding = models['roberta_tokenizer'](
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(models['device'])
    attention_mask = encoding['attention_mask'].to(models['device'])
    
    # Predict
    with torch.no_grad():
        outputs = models['roberta_model'](
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    results = {}
    for prob, idx in zip(top5_probs, top5_indices):
        category = models['categories'][idx.item()]
        confidence = float(prob.item())
        results[category] = confidence
    
    return results


# Example emails for testing
examples = [
    ["I'm having trouble with my graphics card driver. The screen keeps flickering."],
    ["NASA announced new findings from the James Webb Space Telescope about distant galaxies."],
    ["I'm looking for advice on building my first PC. What components should I prioritize?"],
    ["The new iPhone model features significant camera improvements and better battery life."],
    ["Scientists have discovered a new species of deep-sea fish near the Mariana Trench."],
]

# Build Gradio interface
with gr.Blocks(title="Email Classification Demo") as demo:
    gr.Markdown("""
    # Email Classification System
    
    ## Compare Two Approaches:
    - **Baseline**: TF-IDF + Logistic Regression
    - **RoBERTa**: Fine-tuned Transformer with Focal Loss
    
    Enter an email text below to see predictions from both models.
    """)
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Email Text",
            placeholder="Enter email content here...",
            lines=5
        )
    
    with gr.Row():
        baseline_button = gr.Button("Predict with Baseline", variant="secondary")
        roberta_button = gr.Button("Predict with RoBERTa", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline Predictions")
            baseline_output = gr.Label(label="Top 5 Categories", num_top_classes=5)
        
        with gr.Column():
            gr.Markdown("### RoBERTa Predictions")
            roberta_output = gr.Label(label="Top 5 Categories", num_top_classes=5)
    
    # Examples
    gr.Markdown("### 📝 Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=text_input
    )
    
    # Model info
    gr.Markdown(f"""
    ---
    
    ### 📊 Model Status
    - **Baseline (TF-IDF)**: {'Loaded' if models['baseline_loaded'] else 'Not found'}
    - **RoBERTa**: {'Loaded' if models['roberta_loaded'] else 'Not found'}
    
    ### 📈 Performance Metrics
    | | Accuracy | Macro-F1 |
    |---|---|---|
    | Baseline (TF-IDF) | 74.70% | 73.80% |
    | RoBERTa (fine-tuned) | 77.19% | 76.18% |
    | **Improvement** | **+2.49pp** | **+2.38pp** |
    
    ### 📚 Dataset: 20 Newsgroups
    - 20 email-like categories
    - 12,543 training samples
    - 2,688 validation samples
    
    ---
    
    Built by Ayush Kumar Tiwary | [GitHub](https://github.com/hitman567) | [LinkedIn](https://linkedin.com/in/ayushkumartiwary)
    """)
    
    # Connect buttons
    baseline_button.click(
        fn=predict_baseline,
        inputs=text_input,
        outputs=baseline_output
    )
    
    roberta_button.click(
        fn=predict_roberta,
        inputs=text_input,
        outputs=roberta_output
    )


if __name__ == '__main__':
    demo.launch(share=True)