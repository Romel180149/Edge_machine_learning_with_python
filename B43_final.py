import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "B43_2_Final.csv"  # Update this if needed
df = pd.read_csv(file_path)

# Display first few rows to confirm column names
print(df.head())

# Extract relevant features (based on correct column names)
X = df[['V', 'H']]  # Use 'V' and 'H' instead of 'voltage' and 'high'

# Standardizing the data (DBScan is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBScan Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Tune these parameters if needed
df['cluster'] = dbscan.fit_predict(X_scaled)

# Visualize the clustered dataset
plt.figure(figsize=(8,6))
plt.scatter(df['V'], df['H'], c=df['cluster'], cmap='viridis', edgecolors='k')
plt.xlabel('Voltage (V)')
plt.ylabel('High (H)')
plt.title('DBScan Clustering Visualization')
plt.colorbar(label="Cluster Label")
plt.show()
