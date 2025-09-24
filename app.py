from flask import Flask, render_template, request
import pandas as pd
from model import recommend_products

# Load full dataset
df = pd.read_csv("models/sample30.csv")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    if request.method == "POST":
        username = request.form["username"]
        recommendations = recommend_products(username, df)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
