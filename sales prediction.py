# =========================================================
# SALES PREDICTION USING PYTHON AND DATASET
# MACHINE LEARNING PROJECT
# =========================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# LOAD DATASET
# =========================================================

# Replace with your dataset file name
df = pd.read_csv("sales_dataset.csv")

# =========================================================
# DISPLAY DATASET
# =========================================================

print("FIRST 5 ROWS OF DATASET")
print(df.head())

# =========================================================
# DISPLAY COLUMN NAMES
# =========================================================

print("\nCOLUMN NAMES")
print(df.columns)

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

df = df.dropna()

# =========================================================
# FEATURES AND TARGET
# =========================================================

# Example Columns:
# TV, Radio, Newspaper = Input Features
# Sales = Output Target

X = df[['TV', 'Radio', 'Newspaper']]

y = df['Sales']

# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# CREATE MACHINE LEARNING MODEL
# =========================================================

model = LinearRegression()

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED")

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")

print("Mean Absolute Error :", mae)

print("Mean Squared Error :", mse)

print("Root Mean Squared Error :", rmse)

print("R2 Score :", r2)

# =========================================================
# ACTUAL VS PREDICTED VALUES
# =========================================================

result = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': y_pred
})

print("\nACTUAL VS PREDICTED SALES")
print(result)

# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# =========================================================
# PREDICT NEW SALES VALUE
# =========================================================

# Example Input
new_data = pd.DataFrame({
    'TV': [150],
    'Radio': [25],
    'Newspaper': [30]
})

predicted_sales = model.predict(new_data)

print("\nPREDICTED SALES VALUE")
print(predicted_sales[0])

# =========================================================
# MODEL COEFFICIENTS
# =========================================================

print("\nMODEL COEFFICIENTS")

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coefficients)

print("\nINTERCEPT")
print(model.intercept_)

# =========================================================
# REAL WORLD APPLICATIONS
# =========================================================

print("\n===== REAL WORLD APPLICATIONS =====")

print("1. Sales forecasting")

print("2. Advertisement analysis")

print("3. Business growth prediction")

print("4. Product demand estimation")

print("5. Marketing strategy optimization")

print("6. Retail and e-commerce analytics")

# =========================================================
# END OF PROGRAM
# =========================================================