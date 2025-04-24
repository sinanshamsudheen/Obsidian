### created: 18-04-2025
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Naive Bayes Classifier Algorithm Part 1::Naive Bayes is a probabilistic classifier based on Bayes' theorem with a "naive" assumption of conditional independence between features. It calculates the probability of a data point belonging to each class and assigns it to the class with the highest probability. The algorithm learns the prior probabilities of classes and the conditional probabilities of features given each class from the training data. Part 1 focuses on the algorithm's foundations, mathematical principles, and implementation of the Gaussian and Multinomial variants.

---
### Context  
Where and when is it used? Why is it important?

In what context is Naive Bayes Classifier Algorithm Part 1 typically applied::Naive Bayes is applied in text classification (spam filtering, sentiment analysis, document categorization), medical diagnosis, recommendation systems, and real-time prediction scenarios. It's important because it's simple yet effective, computationally efficient (scales linearly with data size), requires minimal training data, handles high-dimensional data well, is relatively robust to irrelevant features, performs well with categorical input, and provides a good baseline for more complex models. Its probabilistic nature also makes it well-suited for incremental learning and multi-class problems.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Training_and_Testing_Data]]
- [[Dummy_Variables_One_Hot_Encoding]]
- [[Logistic_Regression_Binary_Classification]]
- [[K_Fold_Cross_Validation]]
- [[Naive_Bayes_Classifier_Part_2]]

What concepts are connected to Naive Bayes Classifier Algorithm Part 1::[[What_is_Machine_Learning]], [[Training_and_Testing_Data]], [[Dummy_Variables_One_Hot_Encoding]], [[Logistic_Regression_Binary_Classification]], [[K_Fold_Cross_Validation]], [[Naive_Bayes_Classifier_Part_2]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Naive Bayes Classifier implementation and examples
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set random seed for reproducibility
np.random.seed(42)

# Part 1: Gaussian Naive Bayes for continuous features
print("=== Gaussian Naive Bayes on Iris Dataset ===")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create and train a Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Cross-validation
cv_scores = cross_val_score(gnb, X, y, cv=5)
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Calculate prior probabilities and feature means/variances
print("\nPrior Probabilities (Class Distribution):")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {gnb.class_prior_[i]:.4f}")

print("\nFeature Means for Each Class:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}:")
    for j, feature in enumerate(feature_names):
        print(f"  {feature}: {gnb.theta_[i, j]:.4f}")

print("\nFeature Variances for Each Class:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}:")
    for j, feature in enumerate(feature_names):
        print(f"  {feature}: {gnb.var_[i, j]:.4f}")

# Implement Gaussian Naive Bayes from scratch
print("\n=== Gaussian Naive Bayes from Scratch ===")

class GaussianNaiveBayesFromScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        # Calculate prior probabilities
        self.priors = np.zeros(self.n_classes)
        for i, c in enumerate(self.classes):
            self.priors[i] = np.mean(y == c)
        
        # Calculate means and variances for each feature in each class
        self.means = np.zeros((self.n_classes, self.n_features))
        self.variances = np.zeros((self.n_classes, self.n_features))
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i] = X_c.mean(axis=0)
            self.variances[i] = X_c.var(axis=0) + 1e-9  # Add small value to avoid zero variance
    
    def _calculate_likelihood(self, x, mean, var):
        # Gaussian PDF
        exponent = -0.5 * ((x - mean) ** 2) / var
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(exponent)
    
    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], self.n_classes))
        
        for i, x in enumerate(X):
            for c in range(self.n_classes):
                # Prior probability
                class_prior = np.log(self.priors[c])
                
                # Conditional probabilities (in log space to avoid underflow)
                conditional = np.sum(np.log(self._calculate_likelihood(x, self.means[c], self.variances[c])))
                
                # Posterior probability (unnormalized in log space)
                probas[i, c] = class_prior + conditional
        
        # Convert from log space and normalize
        probas = np.exp(probas)
        probas = probas / probas.sum(axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]

# Train and test our implementation
gnb_scratch = GaussianNaiveBayesFromScratch()
gnb_scratch.fit(X_train, y_train)
y_pred_scratch = gnb_scratch.predict(X_test)
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)

print(f"From scratch implementation accuracy: {accuracy_scratch:.4f}")
print(f"Sklearn implementation accuracy: {accuracy:.4f}")

# Part 2: Multinomial Naive Bayes for document classification
print("\n=== Multinomial Naive Bayes for Text Classification ===")

# Load a subset of the 20 Newsgroups dataset (just 3 categories for simplicity)
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics']
news = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# Extract text features using bag-of-words
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_text = vectorizer.fit_transform(news.data)
y_text = news.target

# Split dataset
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text, y_text, test_size=0.3, random_state=42, stratify=y_text
)

# Train a Multinomial Naive Bayes classifier
mnb = MultinomialNB()
mnb.fit(X_train_text, y_train_text)

# Make predictions
y_pred_text = mnb.predict(X_test_text)

# Evaluate the model
accuracy_text = accuracy_score(y_test_text, y_pred_text)
print(f"Accuracy: {accuracy_text:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_text, y_pred_text, target_names=categories))

# Extract most informative features
def show_top_features(vectorizer, clf, class_names, n=10):
    feature_names = vectorizer.get_feature_names_out()
    for i, category in enumerate(class_names):
        top_indices = np.argsort(clf.feature_log_prob_[i])[-n:]
        top_features = [feature_names[j] for j in top_indices]
        top_weights = [clf.feature_log_prob_[i, j] for j in top_indices]
        print(f"\nTop features for category '{category}':")
        for feature, weight in zip(top_features, top_weights):
            print(f"  {feature}: {weight:.4f}")

show_top_features(vectorizer, mnb, categories)

# Part 3: Visualizations
plt.figure(figsize=(18, 12))

# Plot 1: Feature Distributions by Class (Iris)
plt.subplot(2, 3, 1)
for feature_idx in range(4):
    plt.subplot(2, 2, feature_idx + 1)
    for class_idx in range(3):
        plt.hist(X[y == class_idx, feature_idx], alpha=0.5, label=class_names[class_idx])
    plt.title(f'Feature Distribution: {feature_names[feature_idx]}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Plot 2: Confusion Matrix for Gaussian NB
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Plot 3: Decision Boundary Visualization (for 2 features)
plt.figure(figsize=(10, 8))
# Select 2 features for visualization
feature_1, feature_2 = 2, 3  # petal length and width

# Create a mesh grid
h = 0.02
x_min, x_max = X[:, feature_1].min() - 1, X[:, feature_1].max() + 1
y_min, y_max = X[:, feature_2].min() - 1, X[:, feature_2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Train a model on just these two features
gnb_2d = GaussianNB()
gnb_2d.fit(X_train[:, [feature_1, feature_2]], y_train)

# Create the decision boundary
Z = gnb_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

# Plot the training points
scatter = plt.scatter(X[:, feature_1], X[:, feature_2], c=y, 
                     edgecolors='k', cmap=plt.cm.RdYlBu)
plt.xlabel(feature_names[feature_1])
plt.ylabel(feature_names[feature_2])
plt.title('Decision Boundary - Gaussian Naive Bayes')
plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
plt.tight_layout()
plt.show()

# Part 4: Compare different types of Naive Bayes classifiers
print("\n=== Comparing Naive Bayes Variants ===")

# Generate synthetic dataset with binary features
np.random.seed(42)
n_samples = 1000
n_features = 20

# Create binary features
X_binary = np.random.randint(0, 2, size=(n_samples, n_features))
# Create target: sum of certain features > threshold indicates class 1
y_binary = (X_binary[:, 0] + X_binary[:, 1] + X_binary[:, 2] > 1).astype(int)

# Split dataset
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# Train different Naive Bayes variants
gnb_bin = GaussianNB()
bnb = BernoulliNB()

gnb_bin.fit(X_train_bin, y_train_bin)
bnb.fit(X_train_bin, y_train_bin)

# Make predictions
y_pred_gnb = gnb_bin.predict(X_test_bin)
y_pred_bnb = bnb.predict(X_test_bin)

# Evaluate models
print(f"Gaussian NB Accuracy on Binary Data: {accuracy_score(y_test_bin, y_pred_gnb):.4f}")
print(f"Bernoulli NB Accuracy on Binary Data: {accuracy_score(y_test_bin, y_pred_bnb):.4f}")

# Compare types of Naive Bayes on Iris dataset
nb_types = {
    'Gaussian NB': GaussianNB(),
    'Multinomial NB': MultinomialNB(),
    'Bernoulli NB': BernoulliNB()
}

# Need to ensure data is non-negative for Multinomial NB
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_shifted = X_scaled - X_scaled.min()  # Shift to ensure non-negativity

# Cross-validation for each type
for name, nb in nb_types.items():
    scores = cross_val_score(nb, X_scaled_shifted, y, cv=5)
    print(f"{name} on Iris: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Naive Bayes Mathematical Foundation:**

1. **Bayes' Theorem**:
   P(y|X) = P(X|y) × P(y) / P(X)

   Where:
   - P(y|X) is the posterior probability of class y given features X
   - P(X|y) is the likelihood of features X given class y
   - P(y) is the prior probability of class y
   - P(X) is the evidence (feature probability)

2. **Naive Conditional Independence Assumption**:
   P(X|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)

3. **Classification Decision**:
   y_pred = argmax_y [P(y) × ∏ P(xᵢ|y)]

**Naive Bayes Variants**:

1. **Gaussian Naive Bayes**:
   - For continuous features
   - Assumes features follow normal distribution
   - P(xᵢ|y) = (1/√(2πσ²ᵧᵢ)) × exp(-(xᵢ-μᵧᵢ)²/(2σ²ᵧᵢ))

2. **Multinomial Naive Bayes**:
   - For discrete features (e.g., word counts)
   - P(xᵢ|y) = θᵧᵢ^xᵢ
   - Where θᵧᵢ is the probability of feature i appearing in class y

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to effectively handle the "zero frequency" problem in Naive Bayes, especially with sparse data where certain feature-class combinations might not appear in the training set?

How can I apply this to a real project or problem?
I could use Naive Bayes to build a spam filter for emails by treating each word as a feature, using Multinomial Naive Bayes to classify messages based on word frequencies, and potentially improving it by incorporating feature selection to focus on the most discriminative words.

What's a common misconception or edge case?
A common misconception is that the "naive" independence assumption makes Naive Bayes ineffective for real problems. In practice, despite this oversimplification, it often performs surprisingly well, especially for text classification. However, it can struggle with highly correlated features, where the independence assumption is strongly violated, potentially leading to biased probability estimates.

The key idea behind Naive Bayes Classifier Algorithm Part 1 is {{using Bayes' theorem with a simplifying independence assumption to calculate class probabilities based on feature distributions}}.

---
##### Tags

#ai/Naive_Bayes_Classifier_Part_1 #ai #python #flashcard 