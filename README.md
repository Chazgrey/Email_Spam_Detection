# Email Spam Detection

**Classic spam/ham classifier using Naive Bayes** – one of the most effective and lightweight approaches for text-based email spam filtering.

## ✨ Features

- Naive Bayes classifier (MultinomialNB most likely)
- Text vectorization (CountVectorizer or TfidfVectorizer)
- Simple exploratory data analysis (message length distribution, class balance)
- Model persistence with `joblib`
- Basic inference script / demo app (`app.py`)

## 🗂 Project Structure
Email_Spam_Detection/

Emailspam.ipynb        -        Main notebook: EDA → preprocessing → model training & evaluation

app.py                -         Simple deployment / prediction interface (Flask/Streamlit/CLI)

 spam.csv                       Original dataset (likely SMS Spam Collection or similar)

spam_model.joblib         -     Trained Naive Bayes model

vectorizer.joblib           -  Fitted vectorizer

requirements.txt          -     Dependencies

message_length_distribution.png -  # EDA visualization

dataset_split_pie_chart.png    -   # Train/val/test or ham/spam distribution



## ⚙️ Technologies

- Python 3.8+
- scikit-learn
- pandas & numpy
- matplotlib / seaborn (for EDA plots)
- joblib (model & vectorizer serialization)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Chazgrey/Email_Spam_Detection.git
cd Email_Spam_Detection
pip install -r requirements.txt
```
### 2. Explore the model in the notebook
```bash
jupyter notebook Emailspam.ipynb
```
### 3. Run App Locally 
```bash
streamlit run app.py
```
### Live Demo 

Try the deployed version right now — no installation needed!![Click Here](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

→ Live Email Spam Detector ←

Just paste any email/SMS text and see if it's classified as Spam or Ham (with probability). 

### Dataset
- spam.csv 
- Columns: label (ham/spam)

### Typical Performance (Naive Bayes on this data)
- Accuracy: ~97–99%
- Spam Precision: very high (low false positives – crucial for spam filters)
- Spam Recall: strong

### Easy Improvements
- Switch to TF-IDF vectorizer
- Compare with Logistic Regression, SVM, LightGBM, or small BERT
- Add better text cleaning (URLs, numbers, signatures…)
- Show word importance / top spam indicators
- Add cross-validation + confusion matrix visualization



