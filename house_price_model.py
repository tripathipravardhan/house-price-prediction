import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
# This builds the path from this project's folder, so it works on other computers too.
project_folder = Path(__file__).parent
data_path = project_folder / "house-prices-advanced-regression-techniques" / "train.csv"
data = pd.read_csv(data_path)

# 2. Select input features and the value to predict
# GrLivArea = above-ground living area in square feet
# BedroomAbvGr = bedrooms above ground
# FullBath = full bathrooms
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

X = data[features]
y = data[target]

# 3. Split data: 80% for learning, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict prices for the test houses
predictions = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("Model evaluation")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R² Score: {r2:.3f}")

# 7. Show a few actual vs predicted prices
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": predictions
})

print("\nSample predictions:")
print(results.head(10))

# 8. Visualize actual vs predicted prices
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], color="red")

plt.show()

# 9. Predict a new house
new_house = pd.DataFrame(
    [[2000, 3, 2]],
    columns=features
)

predicted_price = model.predict(new_house)[0]
print(f"\nPredicted price for a 2000 sq ft, 3-bedroom, 2-bathroom house: ${predicted_price:,.2f}")
