# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
 
# Load trained model (saved by train_model.py)
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        age = int(request.form["age"])
        garage = int(request.form["garage"])
        pool = int(request.form["pool"])
        gym = int(request.form["gym"])

        # Keep the feature order EXACTLY as training: ['area','bedrooms','age','garage','pool','gym']
        features = np.array([[area, bedrooms, age, garage, pool, gym]])
        predicted_price = model.predict(features)[0]

        return render_template(
            "result.html",
            prediction=f"{predicted_price:,.2f}",
            area=int(area),
            bedrooms=bedrooms,
            age=age,
            garage=garage,
            pool=pool,
            gym=gym
        )
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
