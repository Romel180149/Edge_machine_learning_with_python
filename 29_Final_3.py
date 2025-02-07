# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
# Replace the path with the location where your CSV file is stored
data = pd.read_csv("B43_3_Final.csv")
# print(data)
# Check the first few rows of the dataset
print(data.head())

# Step 2: Split the data into features and target variable
# Assuming the last column is the target
X = data.iloc[:, :-1]  # Select all columns except the last (features)
y = data.iloc[:, -1]   # Select the last column (target)

# Step 3: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply Linear Regression
# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
r2 = r2_score(y_test, y_pred)              # Calculate R² score

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
