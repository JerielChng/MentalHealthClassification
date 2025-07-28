# Mental Health Fake News Classification

This project tackles the challenge of detecting fake vs real mental health news using both traditional machine learning and deep learning techniques. It compares a **Logistic Regression** baseline (using TF-IDF features) with a modern **BERT-based transformer model**, evaluating their effectiveness in text classification.

## Project Objectives

- Classify news articles related to mental health as **real** or **fake**
- Compare performance between traditional statistical models and modern deep learning approaches
- Analyse class-wise prediction effectiveness

## Dataset

- **Source**: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## Preprocessing Steps

- Merged and cleaned real and fake datasets
- Removed duplicates and irrelevant articles
- Tokenised and normalised text (lowercasing, punctuation removal)
- Generated TF-IDF feature vectors for traditional model
- Used Hugging Face tokeniser for BERT model input

## Models Compared

### 1. **Logistic Regression**
- Feature Extraction: TF-IDF
- Classifier: `sklearn.linear_model.LogisticRegression`
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix

### 2. **BERT (DistilBERT)**
- Model: `distilbert-base-uncased` via Hugging Face
- Training via PyTorch or Transformers pipeline
- Batched training using GPU (if available)

## Results

| Metric     | Logistic Regression | BERT Model     |
|------------|---------------------|----------------|
| Accuracy   | ~81.0%              | ~83.1%         |
| Precision  | ~81.1%              | ~83.1%         |
| Recall     | ~81.0%              | ~83.1%         |
| Macro F1   | ~81.0%              | ~83.0%         |

- BERT model shows stronger class-wise performance and generalisation
- Logistic Regression still performs decently on linearly separable features

## Technologies Used

- Python, Pandas, Scikit-learn
- Hugging Face Transformers, PyTorch
- Jupyter Notebook
