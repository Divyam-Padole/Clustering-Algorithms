import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the dataset use the path to the dataset on your system here

dataset_path = input('Enter the path:  ')
dataset = pd.read_csv(dataset_path)


# Display available features for user selection
print("Available features:")
print(", ".join(dataset.columns))

#enter the features you want to use for clustering 
user_features = input("Enter the features you want to use for clustering (comma-separated): ").split(',')


# Select the user-specified features
features = dataset[user_features]


# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


"""
User input for DBSCAN parameters
here eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other
min_samples is the number of samples in a neighborhood for a point to be considered as a core point
eps and min_samples can be determined using the elbow method means you can optimize the values of eps and min_samples
using different method for optimation 
here the optimaization is not done 

"""

# Enter the values for eps and min_samples
# eps = 0.5
# min_samples = 5

eps = float(input("Enter the value for eps: "))
min_samples = int(input("Enter the value for min_samples: "))

# Perform DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(features_scaled)

# Visualize the clusters
plt.scatter(features[user_features[0]], features[user_features[1]], c=labels, cmap='viridis', marker='o', edgecolors='k', s=50)
plt.title('DBSCAN Clustering')
plt.xlabel(user_features[0])
plt.ylabel(user_features[1])
plt.show()
