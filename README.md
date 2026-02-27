# NLP Disaster Tweets

Binary classification of disaster-related tweets using TF-IDF and machine learning, built with Python (Scikit-learn, Pandas, NLTK).

## Tools & Skills Used

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=flat&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat&logo=uv&logoColor=white)

## Quick Access

- [View Notebook](notebooks/01-disaster-tweets.ipynb)
- [View Presentation](presentation/disaster-tweets-classification.pdf)

## Setup & Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### 1. Clone the Repository

```bash
git clone https://github.com/krauseannelize/nlp-disaster-tweets.git
cd nlp-disaster-tweets
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Data Acquisition

The dataset is from the [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) Kaggle competition. Download and place the file in the `data/` folder:

- `train.csv`

### 4. Run the Project

```bash
uv run jupyter lab
```

📌 **Note:** `uv run` automatically uses the project's virtual environment, no manual activation needed

## Project Overview

This project builds an NLP pipeline to classify tweets as disaster-related or not, using the [Kaggle Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) dataset of 7,613 labeled tweets. The task is binary classification: given the text of a tweet, predict whether it refers to a real disaster (1) or not (0).

## Objectives

- Clean and preprocess raw tweet text for use in a machine learning model
- Transform text into numerical features using TF-IDF vectorization
- Train and evaluate classification models (Logistic Regression and Linear SVC as baselines)
- Assess model performance using accuracy, precision, recall, F1-score, and confusion matrix
- Identify what the model handles well and where it struggles

## Methodology

1. **Data Loading & Inspection:** Load `train.csv` with Pandas and explore class distribution, text length, and data quality.
2. **Text Preprocessing:** Lowercase text, remove URLs/mentions/special characters, remove stop words, and apply lemmatization using NLTK.
3. **Text Vectorization:** Convert cleaned text to numerical features using TF-IDF (Term Frequency–Inverse Document Frequency).
4. **Train-Test Split:** Split the data 80/20, stratified to preserve the class distribution in both sets.
5. **Model Training:** Train Logistic Regression and Linear SVC classifiers as baseline models.
6. **Model Evaluation:** Compare both models using classification reports and confusion matrices.
7. **Hyperparameter Tuning:** Use Pipeline + GridSearchCV to tune vocabulary size, ngram range, and regularisation strength (C) with 5-fold cross-validation.

## Key Findings

- **Baseline comparison:** Logistic Regression (82% accuracy) and Linear SVC (80% accuracy) perform similarly, confirming both linear models reach a comparable ceiling with TF-IDF features
- **Precision vs recall trade-off:** Logistic Regression has higher disaster precision (0.84 vs 0.78) but lower recall (0.70 vs 0.74). Linear SVC catches more disasters at the cost of more false positives
- **Tuned accuracy:** 83% after grid search over vocabulary size, ngram range, and regularisation strength
- **Best parameters:** `C=1`, `max_features=5000`, `ngram_range=(1,1)` — a capped vocabulary reduced noise, while bigrams did not improve results
- **Disaster recall gap:** The tuned model catches 72% of disaster tweets but misses 28%, largely due to figurative language (e.g. "my life is a disaster") and class imbalance
- **Error analysis:** False positives are driven by sarcasm and casual use of disaster keywords (e.g. "better than tornado!"). False negatives tend to be tweets where disaster language is subtle or indirect

## Future Improvements

- **Address class imbalance:** Apply oversampling (SMOTE) or class weighting to improve disaster recall
- **Richer features:** Experiment with word embeddings (Word2Vec, GloVe) to capture semantic meaning and word relationships
- **Alternative models:** Try Naive Bayes, Random Forest, or XGBoost for comparison
- **Context-aware models:** Use transformer-based models (e.g. BERT) that understand word order and context, which could help with sarcasm and figurative language
- **Preserve more signal:** Reconsider removing stop words like "not" and "no" that carry negation meaning in short tweet text
