# ⚡ QUICK START CHEAT SHEET

Copy-paste these commands in sequence:

## 1️⃣ Setup (5 minutes)
```bash
cd email-classifier
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 2️⃣ Download Data (5 minutes)
```bash
python src/data_loader.py
```

## 3️⃣ Train Baseline (5 minutes)
```bash
python src/baseline.py
```
Expected result: ~81% F1

## 4️⃣ Train RoBERTa (2-4 hours on CPU, 30 min on GPU)
```bash
python src/train.py
```
Expected result: ~91% F1

## 5️⃣ Launch Demo
```bash
python app.py
```
Open http://127.0.0.1:7860 in browser

---

## 🚨 If Something Breaks

**Virtual env not activating?**
```bash
python -m venv venv --clear
# Then activate again
```

**Import errors?**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**CUDA out of memory?**
Edit `src/train.py` line 267:
```python
batch_size=8  # Change from 16 to 8
```

**Training too slow?**
Use Google Colab (free GPU):
1. Upload files to Google Drive
2. Open Colab notebook
3. Mount Drive
4. Run same commands

---

## 📊 Expected Results Timeline

- **Day 1**: Setup + Baseline (2 hours)
- **Day 2**: Start RoBERTa training (leave overnight if on CPU)
- **Day 3**: Training complete + Demo working
- **Day 4**: GitHub push + Documentation
- **Day 5**: Move to next project (RAG)

---

## 🎯 Deliverables for Resume

✅ GitHub repo with clean code
✅ README with results table
✅ Gradio demo (screenshots)
✅ Can explain in 2-minute pitch

**Resume bullet template:**
```
Built email classification system using fine-tuned RoBERTa achieving 91% macro-F1 
(vs 81% TF-IDF baseline), implemented focal loss for class imbalance handling, 
deployed interactive demo with Gradio
```

---

## 📞 Need Help?

Ask Claude Code in VS Code! Just type:
- "Why is my training loss not decreasing?"
- "How do I reduce memory usage?"
- "Explain focal loss formula"

Good luck! 🚀
