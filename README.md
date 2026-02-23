# NLP Disaster Tweets

Binary classification of disaster-related tweets using TF-IDF and machine learning, built with Python (Scikit-learn, Pandas, NLTK).

## Tools & Skills Used

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=flat&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat&logo=uv&logoColor=white)

## Quick Access

- [View Notebook](notebooks/01-disaster-tweets.ipynb)

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
- Train and evaluate a classification model (Logistic Regression as baseline)
- Assess model performance using accuracy, precision, recall, F1-score, and confusion matrix
- Identify what the model handles well and where it struggles

## Methodology

1. **Data Loading & Inspection:** Load `train.csv` with Pandas and explore class distribution, text length, and data quality.
2. **Text Preprocessing:** Lowercase text, remove URLs/mentions/special characters, remove stop words, and apply lemmatization using NLTK.
3. **Text Vectorization:** Convert cleaned text to numerical features using TF-IDF (Term Frequency–Inverse Document Frequency).
4. **Train-Test Split:** Split the data into training and test sets for evaluation.
5. **Model Training:** Train a Logistic Regression classifier as the baseline model.
6. **Model Evaluation:** Evaluate using classification report and confusion matrix to understand performance across both classes.
7. **Hyperparameter Tuning (Optional):** Experiment with solver, regularization strength (C), and cross-validation.

## Key Findings

## Future Improvements
