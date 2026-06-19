# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Load your CSV (area, bedrooms, age, garage, pool, gym, price)
data = pd.read_csv("house_data.csv")

# Use ONLY columns that exist in your CSV
X = data[['area', 'bedrooms', 'age', 'garage', 'pool', 'gym']]
y = data['price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate RMSE in a version-independent way
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("✅ Model trained. RMSE:", rmse)

# Save trained model
joblib.dump(model, "house_price_model.pkl")
print("✅ Model saved as house_price_model.pkl")
