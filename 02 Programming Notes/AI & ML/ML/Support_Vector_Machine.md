### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Support Vector Machine::A Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression that finds the optimal hyperplane to separate data points of different classes. The algorithm maximizes the margin—the distance between the hyperplane and the nearest data points from each class (called support vectors). SVMs can handle linearly separable data directly, but they also excel at non-linear classification through the "kernel trick," which implicitly maps input features to higher-dimensional spaces without explicitly computing the transformation. This makes SVMs powerful for complex classification tasks while maintaining computational efficiency.

---
### Context  
Where and when is it used? Why is it important?

In what context is Support Vector Machine typically applied::Support Vector Machines are applied in diverse domains including text classification (spam detection, sentiment analysis), image recognition, bioinformatics (protein classification, gene expression analysis), and financial forecasting. They're particularly valuable when dealing with high-dimensional data, when the number of features exceeds the number of samples, or when clear margin of separation is needed. SVMs are important because they're effective in high-dimensional spaces, memory efficient (only support vectors matter), versatile through different kernel functions, and robust against overfitting, especially in text classification and bioinformatics where feature spaces are large but not all features are relevant.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Training_and_Testing_Data]]
- [[Hyper_parameter_Tuning]]
- [[K_Fold_Cross_Validation]]
- [[Bias_vs_Variance]]
- [[Principal_Component_Analysis]]
- [[Dummy_Variables_One_Hot_Encoding]]

What concepts are connected to Support Vector Machine::[[What_is_Machine_Learning]], [[Training_and_Testing_Data]], [[Hyper_parameter_Tuning]], [[K_Fold_Cross_Validation]], [[Bias_vs_Variance]], [[Principal_Component_Analysis]], [[Dummy_Variables_One_Hot_Encoding]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Support Vector Machine Implementation and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.datasets import make_classification, make_circles, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Linear SVM on Linearly Separable Data
print("=== Linear SVM on Linearly Separable Data ===")

# Generate synthetic linearly separable data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           random_state=42, n_clusters_per_class=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear SVM
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = linear_svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Linear SVM Accuracy: {accuracy:.4f}")

# Visualize decision boundary
plt.figure(figsize=(12, 5))

# Plot training data and decision boundary
plt.subplot(1, 2, 1)
plot_decision_regions(X_train_scaled, y_train, clf=linear_svm, legend=2)
plt.title('Linear SVM Decision Boundary (Training Data)')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')

# Plot test data and decision boundary
plt.subplot(1, 2, 2)
plot_decision_regions(X_test_scaled, y_test, clf=linear_svm, legend=2)
plt.title('Linear SVM Decision Boundary (Test Data)')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')

plt.tight_layout()
plt.show()

# 2. SVM with Non-Linear Kernels on Complex Data
print("\n=== SVM with Non-Linear Kernels on Complex Data ===")

# Generate complex non-linear data (circles)
X_circles, y_circles = make_circles(n_samples=100, noise=0.1, factor=0.2, random_state=42)

# Split and scale data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_circles, y_circles, test_size=0.3, random_state=42)
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# Train SVMs with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_models = {}
accuracies = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, gamma='scale', random_state=42)
    svm.fit(X_train_c_scaled, y_train_c)
    svm_models[kernel] = svm
    
    y_pred_c = svm.predict(X_test_c_scaled)
    accuracies[kernel] = accuracy_score(y_test_c, y_pred_c)
    print(f"SVM with {kernel} kernel - Accuracy: {accuracies[kernel]:.4f}")

# Visualize decision boundaries for different kernels
plt.figure(figsize=(15, 10))

for i, kernel in enumerate(kernels):
    plt.subplot(2, 2, i+1)
    plot_decision_regions(X_circles, y_circles, clf=svm_models[kernel], legend=2)
    plt.title(f'SVM with {kernel} kernel (Accuracy: {accuracies[kernel]:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 3. SVM on Real-World Data: Breast Cancer Dataset
print("\n=== SVM on Breast Cancer Dataset ===")

# Load breast cancer dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

print(f"Dataset shape: {X_cancer.shape}")
print(f"Features: {feature_names[:5]}...")
print(f"Target classes: {target_names}")

# Split and scale data
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.3, random_state=42)

scaler_cancer = StandardScaler()
X_train_cancer_scaled = scaler_cancer.fit_transform(X_train_cancer)
X_test_cancer_scaled = scaler_cancer.transform(X_test_cancer)

# Train SVM with RBF kernel
svm_cancer = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_cancer.fit(X_train_cancer_scaled, y_train_cancer)

# Predict and evaluate
y_pred_cancer = svm_cancer.predict(X_test_cancer_scaled)
accuracy_cancer = accuracy_score(y_test_cancer, y_pred_cancer)
print(f"SVM Accuracy on Breast Cancer: {accuracy_cancer:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test_cancer, y_pred_cancer)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_cancer, y_pred_cancer, target_names=target_names))

# 4. SVM Hyperparameter Tuning
print("\n=== SVM Hyperparameter Tuning ===")

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}

# Perform grid search
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_cancer_scaled, y_train_cancer)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate best model
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_cancer_scaled)
accuracy_best = accuracy_score(y_test_cancer, y_pred_best)
print(f"Best SVM Accuracy on Test Set: {accuracy_best:.4f}")

# 5. SVM and Feature Importance Using Weights (for Linear SVM)
print("\n=== Feature Importance in Linear SVM ===")

# Train a Linear SVM
linear_svm_cancer = LinearSVC(C=1.0, dual=False, random_state=42)
linear_svm_cancer.fit(X_train_cancer_scaled, y_train_cancer)

# Feature importance from coefficients
importance = np.abs(linear_svm_cancer.coef_[0])
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Features for Linear SVM')
plt.tight_layout()
plt.show()

# 6. SVM with Dimensionality Reduction using PCA
print("\n=== SVM with PCA ===")

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_cancer_scaled)
X_test_pca = pca.transform(X_test_cancer_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Train SVM on reduced features
svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_pca.fit(X_train_pca, y_train_cancer)

# Predict and evaluate
y_pred_pca = svm_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_cancer, y_pred_pca)
print(f"SVM Accuracy after PCA: {accuracy_pca:.4f}")

# Visualize decision boundary in 2D PCA space
plt.figure(figsize=(10, 8))
plot_decision_regions(X_test_pca, y_test_cancer, clf=svm_pca, legend=2)
plt.title('SVM Decision Boundary in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 7. SVM for Probability Estimation
print("\n=== SVM Probability Estimation ===")

# Predict probabilities with the best SVM model
y_prob = best_svm.predict_proba(X_test_cancer_scaled)

# Plot probability distribution for each class
plt.figure(figsize=(10, 6))
plt.hist(y_prob[:, 1], bins=20, alpha=0.6, label='Malignant')
plt.hist(1 - y_prob[:, 0], bins=20, alpha=0.6, label='Benign')
plt.title('SVM Class Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 8. SVM with Unbalanced Classes
print("\n=== SVM with Unbalanced Classes ===")

# Create an artificially unbalanced dataset
# We'll keep only 20% of class 0 (benign)
indices_class0 = np.where(y_train_cancer == 0)[0]
indices_class1 = np.where(y_train_cancer == 1)[0]
random_indices_class0 = np.random.choice(indices_class0, size=int(len(indices_class0) * 0.2), replace=False)
indices_unbalanced = np.concatenate([random_indices_class0, indices_class1])

X_train_unbalanced = X_train_cancer_scaled[indices_unbalanced]
y_train_unbalanced = y_train_cancer[indices_unbalanced]

print(f"Class distribution in unbalanced training set:")
print(f"Class 0 (Benign): {np.sum(y_train_unbalanced == 0)}")
print(f"Class 1 (Malignant): {np.sum(y_train_unbalanced == 1)}")

# Train SVM without class weights
svm_unbalanced = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
y_pred_unbalanced = svm_unbalanced.predict(X_test_cancer_scaled)

# Train SVM with class weights
svm_weighted = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
svm_weighted.fit(X_train_unbalanced, y_train_unbalanced)
y_pred_weighted = svm_weighted.predict(X_test_cancer_scaled)

# Compare results
print("\nSVM without Class Weights:")
print(classification_report(y_test_cancer, y_pred_unbalanced, target_names=target_names))

print("\nSVM with Class Weights:")
print(classification_report(y_test_cancer, y_pred_weighted, target_names=target_names))

# 9. SVM Kernels Comparison
print("\n=== SVM Kernels Mathematical Explanation ===")
print("1. Linear Kernel: K(x, y) = x^T y")
print("   - Simple dot product of vectors")
print("   - Used when data is linearly separable")
print("   - Fastest to compute, but limited expressiveness")
print("\n2. Polynomial Kernel: K(x, y) = (γx^T y + r)^d")
print("   - Creates polynomial combinations of features")
print("   - Parameters: degree d, gamma γ, and coefficient r")
print("   - Good for problems with all features normalized")
print("\n3. RBF Kernel: K(x, y) = exp(-γ||x - y||^2)")
print("   - Gaussian radial basis function")
print("   - Maps to infinite dimensional space")
print("   - Very versatile, but sensitive to gamma parameter")
print("\n4. Sigmoid Kernel: K(x, y) = tanh(γx^T y + r)")
print("   - Based on the hyperbolic tangent")
print("   - Similar to the activation function in neural networks")
print("   - Parameters: gamma γ and coefficient r")

# 10. The Kernel Trick Explained
print("\n=== The Kernel Trick Visualization ===")

# Generate simple non-linear data
X_nonlinear = np.random.randn(300, 2)
y_nonlinear = np.logical_xor(X_nonlinear[:, 0] > 0, X_nonlinear[:, 1] > 0).astype(int)

# Plot original data
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_nonlinear, cmap=ListedColormap(['#FF9999', '#9999FF']))
plt.title('Original 2D Data (Not Linearly Separable)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Linear SVM in original space
linear_svm_nl = SVC(kernel='linear')
linear_svm_nl.fit(X_nonlinear, y_nonlinear)
y_pred_linear = linear_svm_nl.predict(X_nonlinear)
accuracy_linear = accuracy_score(y_nonlinear, y_pred_linear)

plt.subplot(1, 3, 2)
plot_decision_regions(X_nonlinear, y_nonlinear, clf=linear_svm_nl, legend=2)
plt.title(f'Linear SVM (Acc: {accuracy_linear:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# RBF SVM (implicit mapping to higher dimension)
rbf_svm_nl = SVC(kernel='rbf')
rbf_svm_nl.fit(X_nonlinear, y_nonlinear)
y_pred_rbf = rbf_svm_nl.predict(X_nonlinear)
accuracy_rbf = accuracy_score(y_nonlinear, y_pred_rbf)

plt.subplot(1, 3, 3)
plot_decision_regions(X_nonlinear, y_nonlinear, clf=rbf_svm_nl, legend=2)
plt.title(f'RBF Kernel SVM (Acc: {accuracy_rbf:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Conclusion
print("\n=== Support Vector Machine Summary ===")
print("Key Advantages of SVM:")
print("1. Effective in high-dimensional spaces")
print("2. Memory efficient as it uses only a subset of training points (support vectors)")
print("3. Versatile through different kernel functions for various decision boundaries")
print("4. Robust against overfitting when parameters are properly tuned")
print("\nKey Disadvantages of SVM:")
print("1. Not suitable for very large datasets due to quadratic optimization problem")
print("2. Sensitive to noise, especially with overlapping classes")
print("3. Doesn't directly provide probability estimates (requires additional computation)")
print("4. Requires careful tuning of hyperparameters like C and gamma")
print("\nBest Use Cases:")
print("1. Text and document classification")
print("2. Image recognition with moderate-sized datasets")
print("3. Biological data analysis with high dimensionality")
print("4. Classification tasks where the decision boundary needs to be precise")
```

**Mathematical Foundation of SVM:**

The primary goal of SVM is to find the hyperplane that maximizes the margin between two classes. For binary classification with linearly separable data:

1. **Decision Boundary (Hyperplane)**: w·x + b = 0
   - w: normal vector to the hyperplane
   - b: bias term
   - x: feature vector

2. **Classification Rule**: 
   - Class +1 if w·x + b ≥ 1
   - Class -1 if w·x + b ≤ -1

3. **Margin Optimization**:
   - Maximize the margin 2/||w||
   - Equivalent to minimizing ||w||²/2
   - Subject to y_i(w·x_i + b) ≥ 1 for all training points (x_i, y_i)

4. **Soft Margin SVM** (for non-separable data):
   - Minimize ||w||²/2 + C·Σξ_i
   - Subject to y_i(w·x_i + b) ≥ 1 - ξ_i and ξ_i ≥ 0
   - C: regularization parameter (controls trade-off between margin and classification errors)
   - ξ_i: slack variables allowing for misclassifications

5. **Kernel Trick** (for non-linear boundaries):
   - Maps data to higher-dimensional space: Φ(x)
   - Decision function becomes: f(x) = Σα_i y_i K(x, x_i) + b
   - K(x, x_i) = Φ(x)·Φ(x_i) is the kernel function
   - Common kernels:
     - Linear: K(x, y) = x·y
     - Polynomial: K(x, y) = (γx·y + r)^d
     - RBF/Gaussian: K(x, y) = exp(-γ||x-y||²)
     - Sigmoid: K(x, y) = tanh(γx·y + r)

The beauty of the kernel trick is that we never need to explicitly compute the high-dimensional Φ(x), only the kernel values, making SVMs computationally efficient even for complex decision boundaries.

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to effectively choose between different kernels and their parameters for a given dataset? While RBF kernels work well in many cases, other kernels might be better in specific scenarios, but the selection process often feels more like an art than a science.

How can I apply this to a real project or problem?
I could build a medical diagnosis support system that uses SVM with an RBF kernel to classify medical images (like X-rays or MRIs) as showing signs of disease or not. The ability of SVMs to handle high-dimensional data would be valuable for extracting and using complex patterns from medical images, while the probabilistic outputs could help physicians assess prediction confidence.

What's a common misconception or edge case?
A common misconception about SVMs is that they inherently perform well on imbalanced datasets. In reality, SVMs can be heavily biased toward the majority class if not properly configured. This is because the standard optimization objective tries to maximize accuracy, which can be achieved by simply predicting the majority class. For imbalanced data, using class weights or adjusting the class_weight parameter is crucial.

The key idea behind Support Vector Machine is {{finding the optimal hyperplane that maximizes the margin between classes, using support vectors to define the boundary, and leveraging kernel functions to transform non-linearly separable data into higher-dimensional spaces where they become linearly separable}}.

---
##### Tags

#ai/Support_Vector_Machine #ai #python #flashcard 