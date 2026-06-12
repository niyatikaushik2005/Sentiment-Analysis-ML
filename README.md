# Sentiment-Analysis-ML
# Sentiment Analysis: Classical ML vs Transformer Models

A comparative study of traditional machine learning and transformer-based approaches for Twitter sentiment classification. Built as part of an end-to-end NLP project exploring the strengths and limitations of different text classification methods.

🔗 **Live Demo:** Coming soon  
📓 **Notebook:** [sentiment_analysis_classical_ml.ipynb](./sentiment_analysis_classical_ml.ipynb)  
👤 **Author:** [niyatikaushik2005](https://github.com/niyatikaushik2005)


## Project Overview

This project classifies tweets into three sentiment categories — **Negative, Neutral, and Positive** — using two different approaches:

- **Classical ML:** TF-IDF Vectorization + Logistic Regression
- **Transformer-based:** RoBERTa transformer model (HuggingFace)

The goal is to compare accuracy, performance per class, and understand the practical limitations of each approach on real-world noisy Twitter data.


## Dataset

- **Source:** [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (HuggingFace)
- **Size:** 45,615 tweets (train split)
- **Classes:** Negative (0), Neutral (1), Positive (2)

| Class    | Count  | Percentage |
|----------|--------|------------|
| Negative | 7,093  | 16%        |
| Neutral  | 20,673 | 45%        |
| Positive | 17,849 | 39%        |


## Tech Stack

- Python
- scikit-learn
- HuggingFace Datasets & Transformers
- Pandas, Matplotlib, Seaborn
- FastAPI *(coming soon)*


## Results — Classical ML (TF-IDF + Logistic Regression)

| max_features | Accuracy |
|-------------|----------|
| 5,000       | 65.57%   |
| 10,000      | 65.86%   |
| 50,000      | 65.77%   |

**Classification Report (max_features=10,000):**

| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Negative | 0.60      | 0.34   | 0.44     |
| Neutral  | 0.63      | 0.76   | 0.69     |
| Positive | 0.71      | 0.67   | 0.69     |
| **Overall** | **0.66** | **0.66** | **0.65** |
Add these 3 things to your README today:

---
**BERT (RoBERTa) Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.71 | 0.78 | 0.74 |
| Neutral | 0.74 | 0.69 | 0.71 |
| Positive | 0.72 | 0.71 | 0.71 |
| **Overall** | **0.73** | **0.72** | **0.72** |

---
 **Comaprision results table**

| Model | Accuracy | Negative F1 | Neutral F1 | Positive F1 |
|-------|----------|-------------|------------|-------------|
| TF-IDF + Logistic Regression | 65.85% | 0.44 | 0.69 | 0.69 |
| RoBERTa (BERT) | 72.44% | 0.74 | 0.71 | 0.71 |

---

## Key Findings & Observations

**1. Class Imbalance Impact**
The dataset is significantly imbalanced — Negative tweets make up only 16% of training data compared to 45% Neutral and 39% Positive, directly impacting model performance on the minority class.

**2. Poor Negative Class Detection**
The model achieved only 0.44 F1-score on Negative sentiment vs 0.69 for both other classes, caused directly by underrepresentation of negative examples during training.

**3. Majority Class Bias**
The model frequently defaulted to Neutral predictions when uncertain, likely because Neutral was the most common class at 45% of training data. This was clearly visible in the confusion matrix.

**4. Vocabulary Size Has Minimal Impact**
Varying max_features from 5,000 to 50,000 showed less than 0.3% accuracy difference, suggesting the model's performance ceiling is determined by the algorithm rather than the feature space size.

**5. Neutral Sentiment Hardest to Learn**
TF-IDF word importance visualization revealed the model failed to learn meaningful neutral-sentiment features, associating the neutral class with generic high-frequency words rather than sentiment-specific vocabulary — suggesting neutral sentiment is inherently difficult to capture with frequency-based methods.

**6. Impact of Using BERT over Classical Model**
Switching from TF-IDF + Logistic Regression to a Twitter-pretrained RoBERTa transformer improved overall accuracy by 6.6% and nearly doubled F1-score on the minority Negative class from 0.44 to 0.74, demonstrating transformers' superior ability to handle class imbalance and contextual language.

---

## Coming Soon
- [ ] FastAPI backend
- [ ] Live demo on HuggingFace Spaces

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/niyatikaushik2005/sentiment-analysis-ml.git
cd sentiment-analysis-ml

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook sentiment_analysis_classical_ml.ipynb
```
