#Learning material
"""
https://www.displayr.com/what-is-hierarchical-clustering/#:~:text=Hierarchical%20clustering%20starts%20by%20treating,the%20clusters%20are%20merged%20together.

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
dataset_path = 'dataset\prac6 - prac6.csv'
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

# User input for hierarchical clustering parameters
n_clusters = int(input("Enter the number of clusters: "))

# Perform Hierarchical Clustering
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = clustering.fit_predict(features_scaled)

# Visualize the clusters
plt.scatter(features[user_features[0]], features[user_features[1]], c=labels, cmap='viridis', marker='o', edgecolors='k', s=50)
plt.title('Hierarchical Clustering')
plt.xlabel(user_features[0])
plt.ylabel(user_features[1])
plt.show()

# Plot the dendrogram
linkage_matrix = linkage(features_scaled, 'ward')
dendrogram(linkage_matrix, orientation='top', labels=labels)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
