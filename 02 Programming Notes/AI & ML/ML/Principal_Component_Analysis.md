### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Principal Component Analysis::Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It works by identifying the orthogonal axes (principal components) along which the data varies the most, then projecting the data onto these axes. Each principal component is a linear combination of the original features, with the first component capturing the most variance, the second component capturing the second most variance, and so on. PCA helps simplify data analysis, visualization, and machine learning model training by reducing noise and redundancy in the original data.

---
### Context  
Where and when is it used? Why is it important?

In what context is Principal Component Analysis typically applied::PCA is applied in a wide range of domains including image processing, genomics, finance, and machine learning. It's used for data visualization (reducing dimensions to 2D or 3D for plotting), noise reduction (dropping lower-variance components that may represent noise), feature extraction (creating new meaningful features from combinations of original ones), preprocessing for machine learning (reducing multicollinearity and dimensionality), and compression (representing data with fewer dimensions). PCA is important because it improves computational efficiency of downstream algorithms, helps avoid the curse of dimensionality, handles multicollinearity, facilitates data visualization, and can uncover hidden patterns in complex datasets.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Linear_Regression_Multiple_Variables]]
- [[Support_Vector_Machine]]
- [[Bias_vs_Variance]]
- [[L1_and_L2_Regularization]]
- [[Training_and_Testing_Data]]
- [[Dummy_Variables_One_Hot_Encoding]]

What concepts are connected to Principal Component Analysis::[[What_is_Machine_Learning]], [[Linear_Regression_Multiple_Variables]], [[Support_Vector_Machine]], [[Bias_vs_Variance]], [[L1_and_L2_Regularization]], [[Training_and_Testing_Data]], [[Dummy_Variables_One_Hot_Encoding]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Principal Component Analysis Implementation and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Part 1: Basic PCA implementation and visualization
print("=== Basic PCA Implementation and Visualization ===")

# Generate synthetic data with clear structure
X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.5)

# Add some random features to make the dataset higher dimensional
X_random = np.random.randn(300, 8)
X_combined = np.hstack((X, X_random))
n_features = X_combined.shape[1]

print(f"Original data shape: {X_combined.shape}")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained variance ratio by component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, n_features+1), explained_variance)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')
plt.xticks(range(1, n_features+1))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, n_features+1), cumulative_variance, 'o-')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.xticks(range(1, n_features+1))
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Visualize the first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolors='w')
plt.colorbar(scatter)
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title('PCA: First Two Principal Components')
plt.grid(True)
plt.show()

# Part 2: Determining optimal number of components
print("\n=== Determining Optimal Number of Components ===")

# Function to find number of components to reach a variance threshold
def find_n_components(explained_variance_ratio, threshold=0.9):
    cumulative = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative >= threshold) + 1
    return n_components

optimal_n = find_n_components(explained_variance, threshold=0.9)
print(f"Number of components needed to preserve 90% of variance: {optimal_n}")

# Part 3: PCA for Dimensionality Reduction in a Machine Learning Pipeline
print("\n=== PCA in a Machine Learning Pipeline ===")

# Load a real-world dataset: Breast Cancer
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target
feature_names = cancer.feature_names

print(f"Dataset shape: {X_cancer.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Target classes: {cancer.target_names}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_cancer, y_cancer, test_size=0.3, random_state=42
)

# Create a pipeline without PCA (baseline)
pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Create a pipeline with PCA
pipeline_with_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Train and evaluate both pipelines
pipeline_no_pca.fit(X_train, y_train)
pipeline_with_pca.fit(X_train, y_train)

# Make predictions
y_pred_no_pca = pipeline_no_pca.predict(X_test)
y_pred_with_pca = pipeline_with_pca.predict(X_test)

# Calculate accuracy
accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)
accuracy_with_pca = accuracy_score(y_test, y_pred_with_pca)

print(f"Accuracy without PCA: {accuracy_no_pca:.4f}")
print(f"Accuracy with PCA: {accuracy_with_pca:.4f}")

# Get the number of components used
n_components_used = pipeline_with_pca.named_steps['pca'].n_components_
print(f"Number of components used (95% variance): {n_components_used}")

# Part 4: PCA for Data Visualization (with a multi-class dataset)
print("\n=== PCA for Data Visualization ===")

# Load the Wine dataset
wine = load_wine()
X_wine = wine.data
y_wine = wine.target
wine_feature_names = wine.feature_names
wine_target_names = wine.target_names

print(f"Wine dataset shape: {X_wine.shape}")
print(f"Number of features: {len(wine_feature_names)}")
print(f"Target classes: {wine_target_names}")

# Scale the data
X_wine_scaled = StandardScaler().fit_transform(X_wine)

# Apply PCA
wine_pca = PCA(n_components=2)
X_wine_pca = wine_pca.fit_transform(X_wine_scaled)

# Plot the results
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
for i, color, target_name in zip(range(len(wine_target_names)), colors, wine_target_names):
    plt.scatter(X_wine_pca[y_wine == i, 0], X_wine_pca[y_wine == i, 1], 
                color=color, alpha=0.8, lw=2, label=target_name)
plt.xlabel(f'PC1 ({wine_pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({wine_pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Wine Dataset')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Create a 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Apply PCA with 3 components
wine_pca_3d = PCA(n_components=3)
X_wine_pca_3d = wine_pca_3d.fit_transform(X_wine_scaled)

# Plot
for i, color, target_name in zip(range(len(wine_target_names)), colors, wine_target_names):
    ax.scatter(X_wine_pca_3d[y_wine == i, 0], X_wine_pca_3d[y_wine == i, 1], X_wine_pca_3d[y_wine == i, 2],
               color=color, alpha=0.8, label=target_name)
ax.set_xlabel(f'PC1 ({wine_pca_3d.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({wine_pca_3d.explained_variance_ratio_[1]:.2%})')
ax.set_zlabel(f'PC3 ({wine_pca_3d.explained_variance_ratio_[2]:.2%})')
ax.set_title('3D PCA of Wine Dataset')
plt.legend()
plt.show()

# Part 5: Understanding the Principal Components
print("\n=== Understanding Principal Components ===")

# Get the components (loadings)
pca_wine = PCA().fit(X_wine_scaled)
components = pca_wine.components_

# Create a DataFrame for visualization
loadings = pd.DataFrame(
    components.T,  # Transpose to get features as rows
    columns=[f'PC{i+1}' for i in range(components.shape[0])],
    index=wine_feature_names
)

print("Principal component loadings (first 3 components):")
print(loadings.iloc[:, :3])

# Visualize the loadings of the first two components
plt.figure(figsize=(12, 10))
loadings_plot = loadings.iloc[:, :2].copy()
loadings_plot['feature'] = loadings_plot.index

plt.figure(figsize=(12, 10))
for i, feature in enumerate(wine_feature_names):
    plt.arrow(0, 0, loadings.iloc[i, 0]*5, loadings.iloc[i, 1]*5, head_width=0.1, head_length=0.1, fc='r', ec='r')
    plt.text(loadings.iloc[i, 0]*5.2, loadings.iloc[i, 1]*5.2, feature, fontsize=9)

plt.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=y_wine, alpha=0.3, cmap='viridis')
plt.circle = plt.Circle((0, 0), 5, fill=False, color='r', linestyle='--')
plt.gca().add_patch(plt.circle)
plt.grid(True)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel(f'PC1 ({wine_pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({wine_pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Loadings: Contribution of Each Feature to Principal Components')
plt.tight_layout()
plt.show()

# Part 6: PCA for Noise Reduction
print("\n=== PCA for Noise Reduction ===")

# Create data with added noise
np.random.seed(42)
X_original = X_wine_scaled.copy()

# Add random noise
noise_factor = 0.5
X_noisy = X_original + noise_factor * np.random.normal(0, 1, X_original.shape)

# Apply PCA for denoising
pca_denoise = PCA(n_components=0.9)  # Keep 90% of variance
X_reduced = pca_denoise.fit_transform(X_noisy)
X_reconstructed = pca_denoise.inverse_transform(X_reduced)

# Calculate reconstruction error
reconstruction_error = np.mean((X_original - X_reconstructed) ** 2)
noise_error = np.mean((X_original - X_noisy) ** 2)

print(f"Number of components used for denoising: {pca_denoise.n_components_}")
print(f"Original noise level (MSE): {noise_error:.4f}")
print(f"Reconstruction error after PCA denoising (MSE): {reconstruction_error:.4f}")
print(f"Error reduction: {(1 - reconstruction_error/noise_error):.2%}")

# Part 7: Mathematical Foundation of PCA
print("\n=== Mathematical Foundation of PCA ===")
print("PCA can be derived in multiple ways:")
print("\n1. From Maximizing Variance:")
print("   - Find directions (eigenvectors) along which data varies most")
print("   - These directions are orthogonal to each other")
print("   - They correspond to the eigenvectors of the covariance matrix")
print("\n2. From Minimizing Reconstruction Error:")
print("   - Find a lower-dimensional subspace that minimizes projection error")
print("   - Equivalent to minimizing the squared distance between original and projected points")
print("\n3. Using Singular Value Decomposition (SVD):")
print("   - The principal components are the right singular vectors")
print("   - The explained variances are related to the singular values")
print("\nKey Mathematical Steps:")
print("1. Standardize the data (zero mean, unit variance)")
print("2. Compute the covariance matrix: Σ = (1/n) * X^T * X")
print("3. Find eigenvalues (λ) and eigenvectors (v) of Σ")
print("4. Sort eigenvectors by decreasing eigenvalues")
print("5. Select top k eigenvectors to form the projection matrix W")
print("6. Transform data: X' = X * W")

# Part 8: PCA Implementation from Scratch
print("\n=== PCA Implementation from Scratch ===")

def pca_from_scratch(X, n_components=None):
    # 1. Standardize the data
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # 2. Compute covariance matrix
    cov_matrix = np.cov(X_std, rowvar=False)
    
    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvectors by decreasing eigenvalues
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # 5. Calculate explained variance ratio
    total_var = np.sum(eigenvalues)
    explained_var_ratio = eigenvalues / total_var
    
    # 6. Select number of components
    if n_components is None:
        n_components = X.shape[1]
    elif n_components < 1:  # Interpret as proportion of variance
        n_components = np.argmax(np.cumsum(explained_var_ratio) >= n_components) + 1
    
    # 7. Select top eigenvectors and transform data
    W = eigenvectors[:, :n_components]
    X_pca = X_std.dot(W)
    
    return X_pca, explained_var_ratio, W

# Apply our scratch implementation to wine dataset
X_wine_pca_scratch, explained_var_scratch, W_scratch = pca_from_scratch(X_wine, n_components=2)

# Compare with sklearn's implementation
explained_var_sklearn = wine_pca.explained_variance_ratio_

print("Explained variance from scratch implementation:")
print(explained_var_scratch[:2])
print("\nExplained variance from sklearn:")
print(explained_var_sklearn)

# Visualize both implementations
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for i, color, target_name in zip(range(len(wine_target_names)), colors, wine_target_names):
    plt.scatter(X_wine_pca_scratch[y_wine == i, 0], X_wine_pca_scratch[y_wine == i, 1], 
                color=color, alpha=0.8, label=target_name)
plt.title('PCA from Scratch')
plt.xlabel(f'PC1 ({explained_var_scratch[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_var_scratch[1]:.2%} variance)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for i, color, target_name in zip(range(len(wine_target_names)), colors, wine_target_names):
    plt.scatter(X_wine_pca[y_wine == i, 0], X_wine_pca[y_wine == i, 1], 
                color=color, alpha=0.8, label=target_name)
plt.title('PCA from sklearn')
plt.xlabel(f'PC1 ({wine_pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({wine_pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Part 9: Summary of PCA Best Practices and Limitations
print("\n=== Summary of PCA Best Practices and Limitations ===")
print("Best Practices:")
print("1. Always standardize your data before applying PCA")
print("2. Use explained variance to select the number of components")
print("3. Consider PCA as a preprocessing step in machine learning pipelines")
print("4. Interpret the loadings to understand feature contributions")
print("5. Use PCA for visualization and exploratory data analysis")

print("\nLimitations:")
print("1. PCA assumes linear relationships between features")
print("2. It is sensitive to outliers")
print("3. Interpretability of principal components can be difficult")
print("4. It may not capture complex patterns that require nonlinear transformations")
print("5. It works best when features are correlated")

print("\nAlternatives to PCA:")
print("- Kernel PCA: For nonlinear dimensionality reduction")
print("- Factor Analysis: When interpretability of components is important")
print("- t-SNE/UMAP: For visualization of high-dimensional data")
print("- Independent Component Analysis (ICA): When assuming statistical independence")
print("- Linear Discriminant Analysis (LDA): For supervised dimensionality reduction")
```

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to effectively determine which principal components are most meaningful from a domain perspective, rather than just relying on explained variance? Sometimes components with lower variance might still capture important patterns relevant to the specific problem.

How can I apply this to a real project or problem?
I could use PCA to analyze customer purchase data with hundreds of product categories, reducing it to a smaller set of "shopping patterns" that represent different types of consumer behavior. This would allow for more targeted marketing strategies and simplified customer segmentation, reducing the complexity of working with high-dimensional purchase data.

What's a common misconception or edge case?
A common misconception is that PCA always improves machine learning model performance. In reality, while PCA can reduce noise and simplify models, it might sometimes remove subtle but important signals in the data that are relevant to the target variable. This is particularly true when the variance in the data doesn't align well with the prediction task. Another misconception is interpreting principal components as having inherent meaning - they're mathematical constructs that need careful interpretation in the context of the original features.

The key idea behind Principal Component Analysis is {{transforming high-dimensional data into a lower-dimensional space by projecting along axes of maximum variance, enabling simplified analysis while preserving essential information}}.

---
##### Tags

#ai/Principal_Component_Analysis #ai #python #flashcard 