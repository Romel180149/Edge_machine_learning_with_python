import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
file_path = "B43_4_Final.csv"  # Ensure the correct file path
df = pd.read_csv(file_path)

# Print actual column names
print("Column Names:", df.columns.tolist())

# Ensure the correct target column name
target_column = "Class"  # Update based on the dataset

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Splitting features and target
X = df.drop(columns=[target_column])  # Drop the correct column name
y = df[target_column]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM with RBF kernel
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Train MLP (Neural Network)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluate models correctly
models = {"SVM": y_pred_svm, "MLP": y_pred_mlp, "Decision Tree": y_pred_dt}
results = {}
for model_name, y_pred in models.items():
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),  # Fixed: No 'average' for accuracy
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted')
    }

# Convert results to DataFrame for visualization
results_df = pd.DataFrame(results)

# Plot the comparison
plt.figure(figsize=(10, 5))
results_df.plot(kind='bar', figsize=(10, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.show()
