import random
import numpy as np

class KMeans:

    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    # Train Model
    def fit_predict(self, X):

        # Random centroids
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)

        self.centroids = X[random_index]

        # Iterations
        for i in range(self.max_iter):

            # Assign clusters
            cluster_group = self.assign_clusters(X)

            old_centroids = self.centroids

            # Move centroids
            self.centroids = self.move_centroids(X, cluster_group)

            # Stop if centroids don't change
            if np.allclose(old_centroids, self.centroids):
                break

        return cluster_group

    # Assign nearest centroid
    def assign_clusters(self, X):

        cluster_group = []

        for row in X:

            distances = []

            for centroid in self.centroids:

                distance = np.sqrt(np.dot(row - centroid, row - centroid))

                distances.append(distance)

            min_distance = min(distances)

            index_pos = distances.index(min_distance)

            cluster_group.append(index_pos)

        return np.array(cluster_group)

    # Move centroid
    def move_centroids(self, X, cluster_group):

        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:

            new_centroid = X[cluster_group == type].mean(axis=0)

            new_centroids.append(new_centroid)

        return np.array(new_centroids)

    # Calculate WCSS
    def calculate_wcss(self, X, cluster_group):

        wcss = 0

        for i, centroid in enumerate(self.centroids):

            cluster_points = X[cluster_group == i]

            for point in cluster_points:

                wcss += np.sum((point - centroid) ** 2)

        return wcss