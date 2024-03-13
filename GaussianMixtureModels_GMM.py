import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd

# Get the path to the CSV file from the user
dataset_path = input("Enter the path to your CSV file: ")

# Load the dataset
dataset = pd.read_csv(dataset_path)

# Display available features for user selection
print("Available features:")
print(", ".join(dataset.columns))
user_features = input("Enter the features you want to use for clustering (comma-separated): ").split(',')

# Select the user-specified features
features = dataset[user_features]

# Standardize the features (optional, but often recommended for GMMs)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# User input for Gaussian Mixture Model parameters
n_components = int(input("Enter the number of components (clusters): "))

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(features_scaled)

# Predict clusters
labels = gmm.predict(features_scaled)

# Plot the results
plt.scatter(features[user_features[0]], features[user_features[1]], c=labels, cmap='viridis', s=50, edgecolors='k')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=200, c='red', marker='X', label='Cluster Centers')
plt.title('Gaussian Mixture Model')
plt.xlabel(user_features[0])
plt.ylabel(user_features[1])
plt.legend()
plt.show()
