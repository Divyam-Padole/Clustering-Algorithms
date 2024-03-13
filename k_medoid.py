import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn_extra.cluster import KMedoids
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

# User input for K-Medoids clustering parameters
n_clusters = int(input("Enter the number of clusters: "))

# Perform K-Medoids Clustering
kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
labels = kmedoids.fit_predict(features_scaled)

# Visualize the clusters
plt.scatter(features[user_features[0]], features[user_features[1]], c=labels, cmap='viridis', marker='o', edgecolors='k', s=50)
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Medoids')
plt.title('K-Medoids Clustering')
plt.xlabel(user_features[0])
plt.ylabel(user_features[1])
plt.legend()
plt.show()
