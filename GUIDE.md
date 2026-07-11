# 📱 SMS/Email Spam Classifier — Complete Guide

A machine learning web app that detects spam messages using **Logistic Regression** with **98.26% accuracy**. Built with **Streamlit**, **scikit-learn**, and **NLTK**.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Running the App](#running-the-app)
5. [Project Structure](#project-structure)
6. [How It Works](#how-it-works)
7. [Retraining the Model](#retraining-the-model)
8. [Deployment](#deployment)
9. [Testing Examples](#testing-examples)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This app classifies SMS/email messages as **Spam** or **Not Spam (Ham)** using a machine learning model trained on 5,169 SMS messages. The pipeline:

1. **Preprocesses** text (lowercasing, tokenizing, stemming)
2. **Vectorizes** with TF-IDF (up to 8,000 features with trigrams)
3. **Classifies** using Logistic Regression with balanced class weights

**Accuracy:** 98.26% on the test set.

---

## ✅ Prerequisites

- **Python 3.9+** installed
- **pip** (Python package manager)
- **Chrome/Chromium** (for local app viewing)
- Internet connection (for downloading NLTK data)

---

## 🚀 Setup Instructions

### Step 1: Navigate to the project directory

```bash
cd sms-spam-classifier-main
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**
```
streamlit
nltk
scikit-learn
```

> **Note:** The original `sklearn` package is deprecated. We use `scikit-learn` instead.

### Step 3: Download required NLTK data

The app requires NLTK tokenizers and stopwords. Download them:

```bash
python -m nltk.downloader stopwords
python -m nltk.downloader punkt_tab
```

> **Note:** Modern NLTK uses `punkt_tab`, but older versions use `punkt`. The `nltk.txt` file lists both for compatibility across NLTK versions.

These are also listed in `nltk.txt` for automated deployment (e.g., on Heroku).

---

## ▶️ Running the App

### Start the Streamlit server:

```bash
streamlit run app.py
```

You'll see output like:
```
Local URL: http://localhost:8501
Network URL: http://192.168.3.135:8501
```

Open **http://localhost:8501** in your browser.

### To stop the app:
Press **`Ctrl + C`** in the terminal.

---

## 📁 Project Structure

```
sms-spam-classifier-main/
├── app.py              # Streamlit web application
├── model.pkl           # Trained Logistic Regression model
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── spam.csv            # Original training dataset
├── requirements.txt    # Python dependencies
├── nltk.txt            # NLTK data required (for deployment)
├── setup.sh            # Streamlit config for deployment
├── Procfile            # Heroku deployment config
├── sms-spam-detection.ipynb  # Jupyter notebook (model training)
├── GUIDE.md            # This file
└── README.md           # Short project description
```

---

## ⚙️ How It Works

### 1. Text Preprocessing (`app.py` → `transform_text()`)

```
Input:  "FREE FREE FREE! Get 1000 dollars"
Steps:
  1. Lowercase: "free free free! get 1000 dollars"
  2. Tokenize:  ["free", "free", "free", "!", "get", "1000", "dollars"]
  3. Filter:    Keep only alphanumeric tokens
  4. Stem:      ["free", "free", "free", "get", "1000", "dollar"]
  5. Join:      "free free free get 1000 dollar"
```

> 🔑 **Key detail:** Stopwords are **NOT removed** — they provide useful context for spam detection.

> 📄 **Data note:** The dataset `spam.csv` uses `latin-1` encoding (not UTF-8). When loading with pandas, use `pd.read_csv('spam.csv', encoding='latin-1')`.

### 2. TF-IDF Vectorization

Converts preprocessed text into numerical features. Configured with:
- **max_features=8000** — uses the 8,000 most important words/phrases
- **ngram_range=(1,3)** — captures single words, bigrams, and trigrams
  - Example: "free", "free free", "free free free", "free 1000", "1000 dollar"
- **sublinear_tf=True** — applies log scaling to term frequencies
- **min_df=2** — ignores words that appear in fewer than 2 documents

### 3. Classification

**Logistic Regression** with:
- **C=5.0** — lower regularization (allows more complex decision boundaries)
- **class_weight='balanced'** — automatically adjusts for imbalanced classes (ham vs spam ratio ~7:1)
- **max_iter=1000** — ensures convergence

### 4. Output

- **Spam** — marked as risky (likely promotional, fraudulent, or unsolicited)
- **Not Spam** — regular personal messages

---

## 🔄 Retraining the Model

If you want to retrain the model (e.g., with new data or different parameters):

### Option A: Run the Jupyter Notebook
```bash
jupyter notebook sms-spam-detection.ipynb
```

### Option B: Use Python directly

```python
python -c "
import pickle, pandas as pd, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Preprocessing function (MUST match app.py)
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [ps.stem(t) for t in tokens if len(t) > 1]
    return ' '.join(tokens)

# Load and prepare data
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['target'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.drop_duplicates(keep='first')
df['transformed_text'] = df['text'].apply(transform_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['transformed_text'], df['target'], test_size=0.2, random_state=42
)

# Train
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,3), 
                         sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)

model = LogisticRegression(C=5.0, max_iter=1000, random_state=42, 
                            class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Evaluate
accuracy = model.score(tfidf.transform(X_test), y_test)
print(f'Accuracy: {accuracy:.4f}')

# Save
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print('Models saved!')
"
```

---

## 🚢 Deployment

### Heroku
The project includes deployment files:
- **`Procfile`** — tells Heroku how to run the app
- **`setup.sh`** — configures Streamlit for Heroku
- **`nltk.txt`** — specifies NLTK data to download

To deploy on Heroku:
```bash
# Login and create app
heroku login
heroku create your-app-name

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1
```

### Other platforms (Streamlit Cloud, Railway, etc.)
- Set the start command to: `streamlit run app.py`
- Ensure NLTK data is downloaded (use the `nltk.txt` or a `setup.sh` script)

---

## 🧪 Testing Examples

Open the app and try these messages:

### Should be classified as **Spam** 🚫
| Message | Why it's spam |
|---|---|
| `FREE FREE FREE! Get 1000 dollars` | Repetitive CAPS, money offer |
| `Congratulations you have won a free iPhone` | Fake prize, urgency |
| `URGENT: Your account has been compromised. Verify immediately at fake-bank.com` | Phishing, urgency, fake URL |
| `You are a winner! Claim your �5000 prize now! Call 09061701939` | Prize scam, premium number |
| `Want 2 get laid tonight? Text HOT to 69888` | Adult content spam |

### Should be classified as **Not Spam** ✅
| Message | Why it's ham |
|---|---|
| `Hey, are we still meeting for lunch at 12?` | Normal casual conversation |
| `Dont forget to pick up milk and bread on your way home.` | Reminder, personal |
| `Hi babe, thinking of you. Cant wait to see you tonight.` | Personal affectionate message |
| `I'll be late for the meeting, stuck in traffic` | Work-related notification |
| `Thanks for the birthday wishes! Had a great day` | Social gratitude |

---

## 🔧 Troubleshooting

### ❌ `LookupError: Resource 'punkt_tab' not found`

**Fix:** Download the missing NLTK data:
```bash
python -m nltk.downloader punkt_tab
```

### ❌ `ModuleNotFoundError: No module named 'sklearn'`

**Fix:** The `sklearn` package is deprecated. Use `scikit-learn`:
```bash
pip install scikit-learn
```
Then update `requirements.txt`:
```
scikit-learn
```

### ❌ `NotFittedError: This TfidfTransformer instance is not fitted yet`

**Fix:** The model files (`model.pkl`, `vectorizer.pkl`) are incompatible with your scikit-learn version. Retrain the model (see [Retraining the Model](#retraining-the-model) above).

### ❌ Model predictions are wrong

If the app misclassifies messages:
1. **Retrain** the model (scikit-learn version changes can break pickle files)
2. Ensure the `transform_text()` function in `app.py` **matches exactly** the preprocessing used during training

### ❌ Streamlit doesn't launch

If `streamlit` is not found, the Scripts directory may not be in your PATH:
```bash
# Find where streamlit is installed
python -m streamlit run app.py
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **Accuracy** | **98.26%** |
| **Algorithm** | Logistic Regression |
| **Features** | 8,000 TF-IDF (with trigrams) |
| **Training Data** | 4,135 SMS messages |
| **Test Data** | 1,034 SMS messages |
| **Class Balance** | `class_weight='balanced'` |

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **ML:** [scikit-learn](https://scikit-learn.org/) (Logistic Regression, TF-IDF)
- **NLP:** [NLTK](https://www.nltk.org/) (tokenization, stemming)
- **Language:** Python 3

---

*Happy spam detecting! 🚀*
