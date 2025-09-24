# Sentiment-Based Product Recommendation System

This project implements an **end-to-end product recommendation system** that combines **user-item collaborative filtering** with **sentiment analysis** to provide personalized product suggestions. It uses user reviews to fine-tune recommendations and ensures that only products with highly positive feedback are recommended.

# Workflow Summary

1. Libraries & Setup
2. NLP: nltk (stopwords, lemmatizer)
3. ML: sklearn, xgboost
4. Web: Flask, pyngrok for public URL

## Data Loading

Loads dataset and inspects shape & types.

## Data Cleaning & Preprocessing

1. Drops irrelevant columns.
2. Handles missing values in usernames, titles, and manufacturer.
3. Maps ratings to sentiments (Positive, Neutral, Negative).
4. Encodes sentiments to numeric labels.

## Text Preprocessing

Lowercase, remove non-alphabetic, tokenize, remove stopwords, lemmatize.

## Feature Extraction

TF-IDF vectorization (1-2 grams, max 5000 features).

## Model Training

1. Logistic Regression, Random Forest, Naive Bayes, XGBoost.
2. Prints classification reports.
3. Logistic Regression chosen as the best model.

## Recommendation System

1. Item-based CF: Cosine similarity of product ratings.
2. User-based CF: Cosine similarity of user ratings.
3. Evaluates both systems to pick the best.

## Sentiment Fine-Tuning

1. Top 20 products from CF system.
2. Select top 5 products based on positive sentiment score.

## Flask Deployment

1. Form to enter username.
2. Displays top 5 recommended products.
3. Hosted with ngrok for public URL.

## Folder Structure

Sentiment_Recommendation_Project/
├── app.py
├── model.py
├── models/
│ ├── sentiment_model.joblib
│ ├── tfidf_vectorizer.joblib
│ ├── label_encoder.joblib
│ ├── rating_matrix_filled.csv
│ └── item_similarity_df.csv
├── templates/
│ └── index.html
└── Jupyter notebooks/
└── Sentiment_Based_Product_Recommendation_System.ipynb


## How to Run Locally

1. Clone the repository:

```bash
git clone <repository_url>
cd Sentiment_Recommendation_Project

2. Install dependencies:
pip install -r requirements.txt

3. Run the Flask app:
python app.py

4. Open the URL provided by ngrok or Flask to access the application.

## Deployment

The project can be deployed on Heroku.
model.py contains the ML model and recommendation system for deployment.
app.py connects the backend ML model with the frontend HTML (index.html).

## Author
Vinoth Kumar
