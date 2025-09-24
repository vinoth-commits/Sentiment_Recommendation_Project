# Sentiment-Based Product Recommendation System

This project implements an **end-to-end product recommendation system** that combines **user-item collaborative filtering** with **sentiment analysis** to provide personalized product suggestions. It uses user reviews to fine-tune recommendations and ensures that only products with highly positive feedback are recommended.

## Features

- **Data Cleaning & Preprocessing**
  - Handles missing values in reviews, usernames, and other columns.
  - Combines review titles and texts for complete review text.
  - Performs text preprocessing: lowercasing, punctuation removal, stopword removal, and lemmatization.

- **Sentiment Analysis**
  - Maps review ratings to sentiment labels (Positive, Negative, Neutral).
  - Trains multiple ML models (Logistic Regression, Random Forest, Naive Bayes, XGBoost) for sentiment prediction.
  - Selects the best-performing model based on balanced accuracy and recall for negative reviews.

- **Recommendation System**
  - Builds a **user-item rating matrix** and computes **item-item similarity** using cosine similarity.
  - Recommends top 20 products for a given user.
  - Fine-tunes recommendations by analyzing positive review percentages, filtering **top 5 products** with highest positive sentiment.

- **Web Deployment**
  - User-friendly interface to input username.
  - Displays the top 5 recommended products.
  - Fully deployable via **Flask** and **Heroku**.

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
