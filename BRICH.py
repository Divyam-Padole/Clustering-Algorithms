import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset_path = input("Enter the path to your CSV file: ")
dataset = pd.read_csv(dataset_path)

# Display available features for user selection
print("Available features:")
print(", ".join(dataset.columns))
user_features = input("Enter the features you want to use for clustering (comma-separated): ").split(',')

# Select the user-specified features
features = dataset[user_features]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# User input for BIRCH clustering parameters
threshold = float(input("Enter the threshold for BIRCH clustering: "))
n_clusters = int(input("Enter the number of clusters: "))

# Perform BIRCH Clustering
birch = Birch(threshold=threshold, n_clusters=n_clusters)
labels = birch.fit_predict(features_scaled)

# Visualize the clusters
plt.scatter(features[user_features[0]], features[user_features[1]], c=labels, cmap='viridis', marker='o', edgecolors='k', s=50)
plt.title('BIRCH Clustering')
plt.xlabel(user_features[0])
plt.ylabel(user_features[1])
plt.show()
