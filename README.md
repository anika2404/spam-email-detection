# spam-email-detection
A Spam Detection System built using Python and Scikit-learn.   This project classifies messages as **Spam** or **Not Spam** using the Multinomial Naive Bayes algorithm.

--- 

##  Project Overview

This project implements a complete Machine Learning pipeline:

- Data Loading & Preprocessing
- Train-Test Split with Stratification
- Text Vectorization using Bag-of-Words
- Model Training using Naive Bayes
- Model Evaluation
- Real-time User Message Prediction

The model achieves high accuracy on a labeled spam dataset.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- CountVectorizer (Bag of Words)
- Multinomial Naive Bayes

---

## Dataset

The dataset contains:

- `label` → 0 (Not Spam), 1 (Spam)
- `text` → Email/SMS content

Example:

| label | text |
|-------|------|
| 1 | Win money now |
| 0 | Let’s meet tomorrow |

---

##  Machine Learning Workflow

1. Load dataset
2. Clean and prepare data
3. Split into training and testing sets
4. Convert text to numerical vectors using CountVectorizer
5. Train model using Multinomial Naive Bayes
6. Evaluate using accuracy score
7. Predict custom user messages

---

##  Model Performance

The model achieves approximately:

Accuracy: ~95–98% (depending on dataset split)

Stratified splitting ensures balanced spam/ham distribution.

---

##  How to Run the Project

### 1️ Clone the repository

```bash
git clone https://github.com/your-username/spam-email-classifier.git
