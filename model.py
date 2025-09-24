import pandas as pd
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pickled artifacts
model = joblib.load("models/sentiment_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# Load your rating matrix and item similarity (already computed)
rating_matrix_filled = pd.read_csv("models/rating_matrix_filled.csv", index_col=0)
item_similarity_df = pd.read_csv("models/item_similarity_df.csv", index_col=0)

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Recommendation Functions
def filter_top5_products(username, top_products, df):
    product_sentiment_score = {}
    positive_label = list(label_encoder.transform(['Positive']))[0]

    for product in top_products:
        reviews = df[df['id'] == product]['full_review']
        if reviews.empty:
            continue
        reviews_clean = reviews.apply(preprocess)
        X_reviews = vectorizer.transform(reviews_clean)
        preds = model.predict(X_reviews)
        product_sentiment_score[product] = np.mean(preds == positive_label)

    top5 = sorted(product_sentiment_score, key=product_sentiment_score.get, reverse=True)[:5]
    return top5

def recommend_products(username, df):
    if username not in rating_matrix_filled.index:
        return ["No recommendations available for this user."]

    user_ratings = rating_matrix_filled.loc[username].fillna(0)
    scores = item_similarity_df.dot(user_ratings)
    scores = scores / item_similarity_df.abs().sum(axis=1).replace(0, 1e-9)
    top_20_products = scores.sort_values(ascending=False).head(20).index.tolist()
    top_5_products = filter_top5_products(username, top_20_products, df)
    return top_5_products
