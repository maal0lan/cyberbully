# 🧠 Cyberbullying Detection using DistilBERT

## Implementation Overview & Design Decisions

---

## 📌 1. Problem Statement

The goal of this project is to build a **binary text classification system** that detects whether a given text contains cyberbullying or not.

- Output:
    
    - `0` → Not Cyberbullying
    - `1` → Cyberbullying

---

## ⚙️ 2. Model Selection

### ✅ Implementation

- Used: **DistilBERT (distilbert-base-uncased)**

### 💡 Why?
- Pretrained transformer model (state-of-the-art NLP)
- Faster and lighter than BERT (≈40% smaller)    
- Maintains ~95% of BERT performance
- Suitable for real-time or limited GPU environments

---
## 🧹 3. Text Preprocessing
### ✅ Implementation
- Lowercasing text
- Replacing URLs → `[url]`
- Replacing mentions → `[user]`
- Removing hashtags symbol (`#`)
- Removing numbers (`\d+`)
- Removing extra whitespace
- Converting emojis → text (if `demoji` available)
### 💡 Why?
- Normalize noisy social media data
- Preserve semantic meaning (e.g., emojis → text)
- Remove irrelevant tokens (URLs, mentions)
- Simplify input for better learning

⚠️ Note:

- Numbers were removed, but can be retained for better context in some cases

---
## 📊 4. Dataset Handling

### ✅ Implementation
- Loaded dataset using pandas    
- Auto-detected text & label columns    
- Converted labels into binary format (0/1)    
- Removed null and empty values    

### 💡 Why?
- Ensure clean and consistent data    
- Handle different dataset formats flexibly    
- Avoid training errors due to missing data    

---
## ⚖️ 5. Class Imbalance Handling

### ✅ Implementation
- Used `compute_class_weight()` from sklearn
- Applied weights in `CrossEntropyLoss`
### 💡 Why?
- Dataset is imbalanced (~5:1 ratio)
- Prevent model from biasing toward majority class
- Improve recall for cyberbullying detection
---

## 🔀 6. Train / Validation / Test Split

### ✅ Implementation
- 80% → Train
- 10% → Validation
- 10% → Test
- Stratified splitting
### 💡 Why?
- Maintain class distribution across splits
- Proper evaluation of generalization
- Avoid overfitting

---
## 🔤 7. Tokenization

### ✅ Implementation
- Used `DistilBertTokenizerFast`
- Max length = 128
- Padding & truncation applied

### 💡 Why?
- Convert text → token IDs for model input
- Ensure fixed-length input
- Preserve important context within limit

---

## 🏗️ 8. Model Architecture

### ✅ Implementation

- `DistilBertForSequenceClassification`
- Output layer → 2 classes
### 💡 Why?

- Pretrained language understanding
- Fine-tuned for classification task
- Efficient and accurate

---

## 🏋️ 9. Training Strategy

### ✅ Implementation
- Optimizer: `AdamW`
- Learning rate: `2e-5`
- Scheduler: Linear warmup
- Gradient clipping
- Batch size: 16
- Epochs: 5

### 💡 Why?

- AdamW prevents overfitting via weight decay
- Warmup stabilizes early training
- Gradient clipping prevents exploding gradients
- Small LR ensures stable fine-tuning

---

## 📈 10. Evaluation Metrics

### ✅ Implementation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### 💡 Why?

- Accuracy alone is misleading (due to imbalance)
- F1-score balances precision & recall
- ROC-AUC evaluates overall performance
- Confusion matrix shows error types

---

## 🎯 11. Threshold Optimization

### ✅ Implementation

- Default threshold = 0.5
- Tuned threshold using Precision-Recall curve
- Selected threshold maximizing F1-score (~0.16)    

### 💡 Why?

- Improve balance between precision and recall
- Reduce false negatives (critical for safety task)
- Customize model behavior based on use-case

---

## 🛑 12. Early Stopping

### ✅ Implementation

- Stops training if validation F1 doesn’t improve for 3 epochs

### 💡 Why?
- Prevent overfitting
- Save training time
- Ensure best model is retained

---

## 💾 13. Model Saving

### ✅ Implementation

- Saved:
    - model weights (`.pt`)
    - tokenizer files
    - config.json (threshold, params)
### 💡 Why?

- Enable reuse without retraining
- Support deployment & inference
- Store optimal threshold

---

## 🔮 14. Inference Pipeline

### ✅ Implementation

- Clean input text
- Tokenize
- Get probability
- Apply threshold
- Output label + confidence

### 💡 Why?
- End-to-end prediction system
- Consistent with training pipeline
- Real-world usability

---

## 🚀 15. Overall Approach

### 🧠 Type:

- **Deep Learning (Transformer-based NLP)**    

### 🔥 Pipeline:

1. Data Cleaning    
2. Tokenization
3. Transformer Encoding (DistilBERT)    
4. Classification Head    
5. Threshold Optimization    

---

## 🏁 Conclusion

This project uses a **modern deep learning approach (transformers)** to detect cyberbullying with high accuracy. Key improvements such as **class weighting, threshold tuning, and early stopping** significantly enhance performance and reliability.

---

## ⭐ Key Strengths

- High performance (PR-AUC ~0.99)    
- Handles imbalance effectively    
- Optimized for real-world use    
- Lightweight yet powerful model    

---

## ⚠️ Possible Improvements

- Retain numbers instead of removing    
- Use BERT for higher accuracy    
- Hyperparameter tuning    
- Data augmentation

---