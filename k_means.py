# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def perform_kmeans_clustering(dataset_path, num_clusters=5):
    # Importing the dataset
    dataset = pd.read_csv(dataset_path)

    # Display the available columns for user reference
    print("Available columns in the dataset:")
    print(dataset.columns)

    # Get user input for feature columns
    features_input = input("Enter the feature columns use only the numeric data features(comma-separated): ")
    features_columns = [col.strip() for col in features_input.split(',')]

    # Selecting the features from the dataset
    X = dataset[features_columns].values

    '''
    Using the elbow method to find the optimal number of clusters
    wcss = within cluster sum of squares
    it is the method to determine the optimal number of clusters
    wcss is the sum of squares of the distances of each data point in all clusters to their respective centroids
    The idea is to minimize the wcss for finding the optimal number of clusters

    '''



    # Using the elbow method to find the optimal number of clusters
    wcss = []
    max_clusters = min(11, len(X))  # Limit the number of clusters to the number of samples or 10, whichever is smaller
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plotting the elbow method
    plt.plot(range(1, max_clusters), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Fitting K-Means to the dataset
    num_clusters = min(num_clusters, len(X))  # Limit the number of clusters to the number of samples
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    # Visualizing the clusters
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=100, cmap='viridis', label='Clusters')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    plt.title('Clusters')
    plt.xlabel(features_columns[0])  # Use the user-provided feature names
    plt.ylabel(features_columns[1])
    plt.legend()
    plt.show()

# Example usage
dataset_path = 'dataset\prac6 - prac6.csv'  
# Replace with the path to your dataset
perform_kmeans_clustering(dataset_path)
