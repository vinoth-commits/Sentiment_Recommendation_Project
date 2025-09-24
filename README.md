# Sentiment-Based Product Recommendation System
GenAI based project for review system

## Project Overview
This project implements a sentiment-based product recommendation system that:

1. Predicts the sentiment (Positive/Negative/Neutral) of user reviews using a Logistic Regression model.
2. Builds an item-based collaborative filtering recommendation system.
3. Recommends **top 20 products** to a user and filters **top 5 products** based on positive sentiment.

---

## Folder Structure

Sentiment_Recommendation_Project/
│
├── app.py # Flask file connecting backend ML model and frontend
├── model.py # Contains Logistic Regression model + recommendation system
├── Jupyter notebooks/
│ └── Sentiment_Based_Product_Recommendation_System.ipynb # End-to-end notebook
├── models/ # Pickled models and preprocessed matrices
│ ├── sentiment_model.joblib
│ ├── tfidf_vectorizer.joblib
│ ├── label_encoder.joblib
│ ├── rating_matrix_filled.csv
│ └── item_similarity_df.csv
└── templates/
└── index.html # HTML file for user interface


---

## How to Run

### 1. Jupyter Notebook
- Open `Jupyter notebooks/Sentiment_Based_Product_Recommendation_System.ipynb`.
- Contains all steps:
  - Data cleaning and preprocessing
  - Feature extraction
  - Model building and evaluation
  - Building the recommendation system
  - Flask deployment
- The deployment link via ngrok is included in the notebook for easy access.

### 2. Flask App Deployment
- Ensure the following files exist in the project root:
  - `app.py`
  - `model.py`
  - `templates/index.html`
  - All files in the `models/` folder
- Run locally:
```bash
python app.py
Access the web app via the ngrok URL printed in the console.

### 3. How It Works

Enter a valid username in the input box.
Click Submit.
The app returns the top 5 recommended products for that user based on ratings and sentiment of reviews.

### Assumptions

No new users or products will be introduced beyond the provided dataset.
Models are trained only on the provided dataset.
All preprocessing and feature extraction steps are consistent between training and deployment.
