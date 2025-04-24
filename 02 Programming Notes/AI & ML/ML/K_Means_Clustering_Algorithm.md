### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of K Means Clustering Algorithm::K-Means Clustering is an unsupervised machine learning algorithm that partitions data into K distinct, non-overlapping clusters based on feature similarity. It works by iteratively assigning data points to the nearest cluster centroid and then updating those centroids based on the mean of all points assigned to each cluster. The algorithm aims to minimize the within-cluster sum of squares (inertia), making clusters as compact and separated as possible.

---
### Context  
Where and when is it used? Why is it important?

In what context is K Means Clustering Algorithm typically applied::K-Means Clustering is applied in customer segmentation (grouping customers by purchasing behavior), document classification, image compression, anomaly detection, and preprocessing for other algorithms. It's used whenever we need to discover underlying patterns or groups in unlabeled data. The algorithm is important because it's conceptually simple, computationally efficient for large datasets, easy to implement, produces tight clusters when they exist in the data, and serves as a building block for more complex clustering methods.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Principal_Component_Analysis]]
- [[Hyper_parameter_Tuning]]
- [[Training_and_Testing_Data]]
- [[Bias_vs_Variance]]
- [[K_nearest_neighbors_classification]]

What concepts are connected to K Means Clustering Algorithm::[[What_is_Machine_Learning]], [[Principal_Component_Analysis]], [[Hyper_parameter_Tuning]], [[K_nearest_neighbors_classification]], [[Bias_vs_Variance]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# K-Means Clustering example
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial.distance import cdist

# Set a random seed for reproducibility
np.random.seed(42)

# Part 1: Basic K-Means on synthetic data
print("=== K-Means on Synthetic Data ===")

# Generate synthetic data with clear clusters
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit the K-Means model
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

# Get cluster centers and inertia (sum of squared distances to centroids)
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Number of iterations: {kmeans.n_iter_}")
print(f"Inertia: {inertia:.2f}")

# Calculate silhouette score (ranges from -1 to 1, higher is better)
silhouette = silhouette_score(X_scaled, y_pred)
print(f"Silhouette Score: {silhouette:.2f}")

# Calculate Calinski-Harabasz index (higher is better)
ch_score = calinski_harabasz_score(X_scaled, y_pred)
print(f"Calinski-Harabasz Index: {ch_score:.2f}")

# Part 2: Determining the optimal number of clusters with the Elbow Method
print("\n=== Elbow Method for Optimal K ===")

max_k = 10
inertias = []
silhouette_scores = []
ch_scores = []

for k in range(1, max_k+1):
    if k > 1:  # Silhouette requires at least 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        y_pred = kmeans.labels_
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, y_pred))
        ch_scores.append(calinski_harabasz_score(X_scaled, y_pred))
    else:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(0)  # Placeholder for k=1
        ch_scores.append(0)  # Placeholder for k=1

# Print the optimal k based on different metrics
print(f"Suggested optimal k based on inertia elbow: visual inspection needed")
print(f"Suggested optimal k based on silhouette score: {np.argmax(silhouette_scores) + 1}")
print(f"Suggested optimal k based on Calinski-Harabasz index: {np.argmax(ch_scores) + 1}")

# Part 3: K-Means on Iris dataset (with PCA visualization)
print("\n=== K-Means on Iris Dataset ===")

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris_true = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Standardize the data
X_iris_scaled = StandardScaler().fit_transform(X_iris)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_iris_pca = pca.fit_transform(X_iris_scaled)
explained_variance = pca.explained_variance_ratio_
print(f"PCA explained variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
print(f"Total explained variance: {sum(explained_variance):.2f}")

# Run K-Means with k=3 (matching the true number of classes)
kmeans_iris = KMeans(n_clusters=3, random_state=42)
y_iris_pred = kmeans_iris.fit_predict(X_iris_scaled)

# Compute accuracy (this is just for illustration, not a standard clustering metric)
# We need to map cluster labels to true labels first
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    # Create a confusion matrix
    contingency_matrix = np.zeros((np.max(y_true) + 1, np.max(y_pred) + 1))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    
    # Find the best assignment using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    # Create a mapped prediction array
    y_pred_mapped = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_pred_mapped[i] = col_ind[y_pred[i]]
    
    # Return accuracy
    return accuracy_score(y_true, y_pred_mapped)

acc = cluster_accuracy(y_iris_true, y_iris_pred)
print(f"Cluster to Class Accuracy: {acc:.2f}")

# Part 4: From scratch K-Means implementation (for educational purposes)
print("\n=== K-Means Implementation from Scratch ===")

def kmeans_from_scratch(X, k, max_iters=100, tol=1e-4):
    # Randomly initialize centroids
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    # Previous centroids to check for convergence
    prev_centroids = np.zeros_like(centroids)
    
    # Cluster assignments
    clusters = np.zeros(n_samples)
    
    # Track iterations
    n_iter = 0
    
    # Main loop
    for i in range(max_iters):
        # Store previous centroids
        prev_centroids = centroids.copy()
        
        # Assign each point to nearest centroid
        distances = cdist(X, centroids, 'euclidean')
        clusters = np.argmin(distances, axis=1)
        
        # Update centroids
        for j in range(k):
            if np.sum(clusters == j) > 0:  # Avoid empty clusters
                centroids[j] = np.mean(X[clusters == j], axis=0)
        
        # Check for convergence
        n_iter = i + 1
        if np.allclose(prev_centroids, centroids, rtol=tol):
            break
    
    # Calculate inertia
    distances = cdist(X, centroids, 'euclidean')
    inertia = np.sum(np.min(distances, axis=1) ** 2)
    
    return clusters, centroids, inertia, n_iter

# Apply from-scratch implementation to smaller synthetic dataset
small_X, small_y = make_blobs(n_samples=100, centers=3, random_state=42)
small_X_scaled = StandardScaler().fit_transform(small_X)

# Run our implementation
scratch_clusters, scratch_centroids, scratch_inertia, scratch_iters = kmeans_from_scratch(small_X_scaled, k=3)

# Run scikit-learn implementation for comparison
sklearn_kmeans = KMeans(n_clusters=3, random_state=42, max_iter=100, tol=1e-4)
sklearn_kmeans.fit(small_X_scaled)
sklearn_inertia = sklearn_kmeans.inertia_
sklearn_iters = sklearn_kmeans.n_iter_

print(f"From scratch - Inertia: {scratch_inertia:.2f}, Iterations: {scratch_iters}")
print(f"Scikit-learn - Inertia: {sklearn_inertia:.2f}, Iterations: {sklearn_iters}")

# Part 5: Visualizations
plt.figure(figsize=(20, 15))

# Plot 1: Original data with K-Means clusters (synthetic data)
plt.subplot(2, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering Results (k=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Plot 2: Elbow Method visualization
plt.subplot(2, 3, 2)
plt.plot(range(1, max_k+1), inertias, 'o-', color='blue')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Plot 3: Silhouette and CH Index
plt.subplot(2, 3, 3)
plt.plot(range(1, max_k+1), silhouette_scores, 'o-', color='blue', label='Silhouette Score')
plt.plot(range(1, max_k+1), np.array(ch_scores) / max(ch_scores), 'o-', color='red', 
         label='Normalized CH Index')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.title('Clustering Evaluation Metrics')
plt.legend()
plt.grid(True)

# Plot 4: Iris clustering with PCA
plt.subplot(2, 3, 4)
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris_pred, cmap='viridis', alpha=0.7)
centroids_pca = pca.transform(kmeans_iris.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering of Iris Dataset (PCA)')
plt.xlabel(f'PC1 ({explained_variance[0]:.2f} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f} variance)')
plt.legend()
plt.grid(True)

# Plot 5: True classes of Iris dataset with PCA
plt.subplot(2, 3, 5)
for i, target_name in enumerate(target_names):
    plt.scatter(X_iris_pca[y_iris_true == i, 0], X_iris_pca[y_iris_true == i, 1], 
                label=target_name, alpha=0.7)
plt.title('True Classes of Iris Dataset (PCA)')
plt.xlabel(f'PC1 ({explained_variance[0]:.2f} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2f} variance)')
plt.legend()
plt.grid(True)

# Plot 6: K-Means vs From Scratch comparison
plt.subplot(2, 3, 6)
plt.scatter(small_X_scaled[:, 0], small_X_scaled[:, 1], c=scratch_clusters, cmap='viridis', alpha=0.5, label='From Scratch')
plt.scatter(scratch_centroids[:, 0], scratch_centroids[:, 1], c='red', marker='X', s=200, label='Scratch Centroids')
plt.scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], 
            c='blue', marker='o', s=200, alpha=0.5, label='Sklearn Centroids')
plt.title('K-Means: From Scratch vs Scikit-learn')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Part 6: K-Means limitations demonstration (non-globular clusters)
from sklearn.datasets import make_moons

# Generate non-globular data
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply K-Means with k=2
kmeans_moons = KMeans(n_clusters=2, random_state=42)
y_moons_pred = kmeans_moons.fit_predict(X_moons)

# Visualize the issue
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.7)
plt.title('True Structure of Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons_pred, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nK-Means Limitations:")
print("1. Assumes clusters are spherical and equally sized")
print("2. Sensitive to initial centroids")
print("3. Requires specifying k in advance")
print("4. Not effective for non-globular clusters")
print("5. Sensitive to outliers")
```

**K-Means Algorithm Steps:**

1. **Initialization**: Randomly select k data points as initial centroids (or use more sophisticated initialization like k-means++).

2. **Assignment Step**: Assign each data point to the nearest centroid, forming k clusters.

3. **Update Step**: Recalculate the centroids as the mean of all points assigned to each cluster.

4. **Iteration**: Repeat steps 2 and 3 until convergence (centroids no longer change significantly or maximum iterations reached).

5. **Final Result**: Return the final clusters and centroids.

**Objective Function**: Minimize the sum of squared distances between points and their cluster centroids (inertia):
J = Σ Σ ||x_i - c_j||², where x_i is a data point and c_j is its cluster centroid.

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to effectively handle data with clusters of varying densities and sizes, where the standard K-Means assumption of equal variance in all directions breaks down?

How can I apply this to a real project or problem?
I could use K-Means to segment customers for an e-commerce platform based on purchase behavior, engagement metrics, and demographics, enabling personalized marketing strategies for each segment. The clusters would help identify high-value customer groups and their distinctive characteristics.

What's a common misconception or edge case?
A common misconception is that K-Means will always find the globally optimal clustering. In reality, it's highly sensitive to the initial placement of centroids and may converge to local optima, which is why it's often run multiple times with different initializations. Another misconception is that K-Means can identify clusters of any shape; it struggles with non-convex clusters and works best with spherical, equally-sized clusters.

The key idea behind K Means Clustering Algorithm is {{partitioning data into homogeneous groups by iteratively assigning points to the nearest cluster center and updating those centers}}.

---
##### Tags

#ai/K_Means_Clustering_Algorithm #ai #python #flashcard 