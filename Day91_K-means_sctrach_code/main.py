import pandas as pd
import matplotlib.pyplot as plt
from kmeans import KMeans


# =====================================
# Load Dataset
# =====================================

df = pd.read_csv('student_clustering.csv')

print(df.head())

# Convert to numpy array
X = df.iloc[:, :].values

# =====================================
# Elbow Method
# =====================================

wcss_values = []

K = range(1, 11)

for k in K:

    km = KMeans(n_clusters=k, max_iter=500)

    clusters = km.fit_predict(X)

    wcss = km.calculate_wcss(X, clusters)

    wcss_values.append(wcss)

# Plot Elbow Graph
plt.figure(figsize=(8,5))

plt.plot(K, wcss_values, marker='o')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')

plt.grid(True)

plt.show()

# =====================================
# Final KMeans Clustering
# =====================================

optimal_k = 4

km = KMeans(n_clusters=optimal_k, max_iter=500)

y_means = km.fit_predict(X)

# =====================================
# Plot Clusters
# =====================================

colors = ['red', 'blue', 'green', 'yellow']

plt.figure(figsize=(8,6))

for i in range(optimal_k):

    plt.scatter(
        X[y_means == i, 0],
        X[y_means == i, 1],
        color=colors[i],
        label=f'Cluster {i}'
    )

# Plot Centroids
plt.scatter(
    km.centroids[:, 0],
    km.centroids[:, 1],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title('K-Means Clustering')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend()

plt.grid(True)

plt.show()