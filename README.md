##Fine-Tuned DeBERTa for Opinion Mining from Customer Reviews

**This project applies a fine-tuned DeBERTa model using LoRA (Low-Rank Adaptation) to perform **Aspect-Based Sentiment Analysis (ABSA)**. Specifically, it tackles two subtasks:**

1. **Aspect Term Extraction** — identifying relevant features, colors, brands, and aspects from customer reviews using token classification.
2. **Sentiment Classification** — determining the sentiment (positive, negative, neutral) associated with extracted aspects using sequence classification.

---

## 📌 Project Overview

- **Course**: Natural Language Understanding (NLU)
- **University**: Helwan University, Faculty of Computers & Artificial Intelligence
- **Supervisor**: Dr. Doaa Abdullah
- **Team**:
  - Esraa Ammar
  - Farida Khaled
  - Muhammed Tarek
  - Muhammed Yasser
  - Madiha Saeid
  - Hania Ruby

---

## 📂 Dataset

- **Source**: [Kaggle - Laptop ABSA Dataset](https://www.kaggle.com/datasets/benkabongo/laptop-absa)
- **Content**: Laptop reviews annotated with:
  - Aspect Terms (e.g., "battery life")
  - Categories (e.g., feature, color, brand)
  - Sentiments (positive, negative, neutral)

---

## 🔧 Preprocessing Steps

- Tokenization and NER tagging of reviews
- Mapping annotations to BIO tags:
  - B-ASP, I-ASP, B-FEATURE, I-FEATURE, B-BRAND, etc.
- Applied HuggingFace `DebertaV2ForTokenClassification` and `AutoModelForSequenceClassification`
- Addressed **class imbalance** using custom `Trainer` class with computed **class weights**

---

## 🧪 Models and Fine-Tuning

### ✅ Task 1: Aspect Term Extraction

- Model: `DeBERTaV2ForTokenClassification`
- LoRA-adapted fine-tuning with PEFT library
- Trained to detect aspects, features, colors, and brands in review text

### ✅ Task 2: Aspect Sentiment Classification

- Input: `"Aspect: {aspect}. Sentence: {review}"`
- Model: `AutoModelForSequenceClassification`
- Fine-tuned for 3-class sentiment classification

---

## 📊 Results

| Task | Metric | Base Model | Fine-Tuned with LoRA |
|------|--------|------------|-----------------------|
| **Aspect Extraction** | Accuracy | 14% | **89%** |
| | F1-score | 0.04 (macro) | **0.91** (weighted) |
| **Sentiment Analysis** | Accuracy | N/A | **88%** |
| | F1-score | N/A | **0.85** (weighted) |

> 📉 Fine-tuning resulted in a dramatic performance boost for both tasks.

---

## 📈 Evaluation Metrics

- **Precision / Recall / F1-score**
- **Confusion Matrices**
- Loss tracking during training and evaluation
- Used visual plots for result comparison (see notebook)

---
### ✅ LoRA Fine-Tuned Token Classification (Task 1)

```text```
Accuracy: 0.89
F1-score (macro): 0.40
Training Loss: 1.30
Eval Loss: 0.89 
---
| Label     | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| B-ASP     | 0.41      | 0.93   | 0.57     |
| I-ASP     | 0.26      | 0.83   | 0.39     |
| B-FEATURE | 0.10      | 0.07   | 0.08     |
| I-FEATURE | 0.00      | 0.00   | 0.00     |
| O         | 1.00      | 0.90   | 0.95     |

---

## 🧠 LoRA Fine-Tuning

- Implemented Low-Rank Adaptation (LoRA) via `peft` library
- Reduced trainable parameters while improving performance
- Allowed fine-tuning of large models with lower computational cost

---
---

## 🆚 Baseline (No Fine-Tuning)

The baseline DeBERTa model (without fine-tuning or LoRA) failed to generalize well to the ABSA task.

- **Accuracy**: ~14%
- **F1 Score (Macro)**: 0.04

This demonstrates the importance of domain-specific fine-tuning and adaptation techniques like LoRA.

---

## 🏗️ Class Imbalance Handling

The dataset exhibited **severe class imbalance**, especially in:

- Low-frequency NER labels (e.g., `B-FEATURE`, `I-FEATURE`)
- Sentiment classes (notably `neutral`)

To address this, we computed class weights based on label frequency and used them in a **custom `Trainer` class**. The `CrossEntropyLoss` function was adapted to include these weights and improve learning from underrepresented labels.

```python
loss_fct = nn.CrossEntropyLoss(weight=all_class_weights, ignore_index=-100)
loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))


## 🗃️ Repository Contents

```bash
.
├── deberta-absa-final.ipynb   # Main training and evaluation notebook
├── README.md                  # Project overview and instructions
├── sample_outputs/            # Visualizations and confusion matrices
└── utils/                     # Tokenization, preprocessing functions

## 🔧 How to Run

### Clone the repository

```bash
git clone https://github.com/your_username/Fine_Tuned_DeBERTA_for_Opinion_Mining.git
cd Fine_Tuned_DeBERTA_for_Opinion_Mining



