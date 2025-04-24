### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Naive Bayes Classifier Algorithm Part 2::Naive Bayes Classifier Part 2 focuses on advanced implementation aspects, optimization techniques, and handling specific challenges of the algorithm. It covers Bernoulli Naive Bayes for binary features, solving the zero probability problem through Laplace smoothing, handling continuous data with kernel density estimation, dealing with class imbalance, feature selection methods, and implementing Naive Bayes with real-world considerations such as computational efficiency and online learning.

---
### Context  
Where and when is it used? Why is it important?

In what context is Naive Bayes Classifier Algorithm Part 2 typically applied::The advanced Naive Bayes concepts are applied in complex text classification scenarios like sentiment analysis with imbalanced classes, large-scale spam filtering systems that need online updating, cyber threat detection requiring real-time classification, natural language processing with high-dimensional sparse data, and multimodal classification problems combining different feature types. These advanced techniques are important because they help overcome practical limitations of basic Naive Bayes implementations, improve model robustness in real-world scenarios, enable better handling of diverse data characteristics, and provide fine-tuning capabilities for specific applications.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Naive_Bayes_Classifier_Part_1]]
- [[Dummy_Variables_One_Hot_Encoding]]
- [[K_Fold_Cross_Validation]]
- [[Hyper_parameter_Tuning]]
- [[Training_and_Testing_Data]]

What concepts are connected to Naive Bayes Classifier Algorithm Part 2::[[What_is_Machine_Learning]], [[Naive_Bayes_Classifier_Part_1]], [[Dummy_Variables_One_Hot_Encoding]], [[K_Fold_Cross_Validation]], [[Hyper_parameter_Tuning]], [[Training_and_Testing_Data]]
<!--SR:!2025-04-22,4,270-->

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Advanced Naive Bayes techniques and optimizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Part 1: Bernoulli Naive Bayes - When features are binary
print("=== Bernoulli Naive Bayes for Binary Features ===")

# Generate a synthetic binary dataset
X_binary, y_binary = make_classification(
    n_samples=1000, n_features=20, n_informative=10, 
    n_redundant=5, n_classes=2, random_state=42
)

# Convert features to binary (0 or 1)
X_binary = (X_binary > 0).astype(int)

# Split dataset
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# Train a Bernoulli Naive Bayes classifier
bnb = BernoulliNB()
bnb.fit(X_train_bin, y_train_bin)

# Make predictions
y_pred_bin = bnb.predict(X_test_bin)
y_prob_bin = bnb.predict_proba(X_test_bin)[:, 1]

# Evaluate the model
accuracy_bin = accuracy_score(y_test_bin, y_pred_bin)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_bin, y_pred_bin, average='binary')

print(f"Accuracy: {accuracy_bin:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Part 2: Handling the Zero Probability Problem with Laplace Smoothing
print("\n=== Laplace Smoothing (Additive Smoothing) ===")

# Example text data with sparse features
texts = [
    "I love machine learning",
    "Machine learning is fascinating",
    "Python makes machine learning easy",
    "Deep learning is a subset of machine learning",
    "Natural language processing is fun"
]
labels = [1, 1, 1, 0, 0]  # Binary labels

# Split into training and test sets
train_texts = texts[:3]
train_labels = labels[:3]
test_texts = texts[3:]
test_labels = labels[3:]

# Create vectorizers with different smoothing parameters
vectorizer_no_smooth = CountVectorizer()
X_train_no_smooth = vectorizer_no_smooth.fit_transform(train_texts)
X_test_no_smooth = vectorizer_no_smooth.transform(test_texts)

# Train without smoothing (alpha=0)
mnb_no_smooth = MultinomialNB(alpha=0.0)
mnb_no_smooth.fit(X_train_no_smooth, train_labels)

# Problem: Words like "deep" and "subset" don't appear in training data
# This would lead to zero probabilities without smoothing

# Train with Laplace smoothing (default alpha=1.0)
mnb_smooth = MultinomialNB(alpha=1.0)
mnb_smooth.fit(X_train_no_smooth, train_labels)

# Demonstrate the effect on feature log probabilities
feature_names = vectorizer_no_smooth.get_feature_names_out()
print("Selected feature log probabilities for class 1:")

# For demonstration, find a feature that doesn't appear in class 1
zero_prob_feature = None
for i, feature in enumerate(feature_names):
    if feature not in " ".join(train_texts[:2]).lower():  # Not in class 1 documents
        zero_prob_feature = feature
        break

if zero_prob_feature:
    feature_idx = np.where(feature_names == zero_prob_feature)[0][0]
    
    try:
        print(f"Feature '{zero_prob_feature}':")
        print(f"  Without smoothing: {mnb_no_smooth.feature_log_prob_[1, feature_idx]}")
    except:
        print(f"  Without smoothing: -inf (zero probability)")
    
    print(f"  With smoothing: {mnb_smooth.feature_log_prob_[1, feature_idx]}")

# Predict with smoothing
try:
    y_pred_no_smooth = mnb_no_smooth.predict(X_test_no_smooth)
    print(f"Predictions without smoothing: {y_pred_no_smooth}")
except:
    print("Prediction failed without smoothing due to zero probabilities")

y_pred_smooth = mnb_smooth.predict(X_test_no_smooth)
print(f"Predictions with smoothing: {y_pred_smooth}")
print(f"True labels: {test_labels}")

# Part 3: Comparing Different Smoothing Parameters
print("\n=== Effect of Different Smoothing Parameters ===")

# Load a subset of the 20 Newsgroups dataset
categories = ['comp.graphics', 'sci.med', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(
    subset='all', categories=categories, 
    shuffle=True, random_state=42,
    remove=('headers', 'footers', 'quotes')  # Remove metadata to avoid overfitting
)

# Split dataset
X_news = newsgroups.data
y_news = newsgroups.target

X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(
    X_news, y_news, test_size=0.3, random_state=42, stratify=y_news
)

# Create a pipeline with vectorization and Naive Bayes
alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
results = []

for alpha in alphas:
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_features=1000, stop_words='english')),
        ('classifier', MultinomialNB(alpha=alpha))
    ])
    
    pipeline.fit(X_train_news, y_train_news)
    score = pipeline.score(X_test_news, y_test_news)
    results.append((alpha, score))
    print(f"Alpha = {alpha:.4f}, Accuracy = {score:.4f}")

# Demonstrate the impact of smoothing on feature importance
print("\nFeature importance changes with smoothing:")
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train_news)
X_test_vec = vectorizer.transform(X_test_news)

# Get top 5 features for class 0 with different alphas
mnb_low_alpha = MultinomialNB(alpha=0.001)
mnb_high_alpha = MultinomialNB(alpha=10.0)

mnb_low_alpha.fit(X_train_vec, y_train_news)
mnb_high_alpha.fit(X_train_vec, y_train_news)

feature_names = vectorizer.get_feature_names_out()

# Print top features for first class with different smoothing
for name, model in [("Low Alpha (0.001)", mnb_low_alpha), ("High Alpha (10.0)", mnb_high_alpha)]:
    top_indices = np.argsort(model.feature_log_prob_[0])[-5:]
    top_features = [feature_names[i] for i in top_indices]
    print(f"\nTop 5 features for class '{categories[0]}' with {name}:")
    for feature in top_features:
        feature_idx = np.where(feature_names == feature)[0][0]
        print(f"  {feature}: {model.feature_log_prob_[0, feature_idx]:.4f}")

# Part 4: Complement Naive Bayes for Imbalanced Data
print("\n=== Complement Naive Bayes for Imbalanced Data ===")

# Create an imbalanced dataset by subsampling
category_counts = np.bincount(y_news)
min_count = category_counts.min()
max_count = category_counts.max()
imbalance_ratio = max_count / min_count
print(f"Original class distribution: {category_counts}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

# Create imbalanced dataset
majority_class = np.argmax(category_counts)
minority_indices = np.where(y_news != majority_class)[0]
majority_indices = np.where(y_news == majority_class)[0]

# Take all majority and just a fraction of others
majority_subsample = majority_indices
minority_subsample = np.random.choice(minority_indices, size=len(majority_indices)//5, replace=False)
imbalanced_indices = np.concatenate([majority_subsample, minority_subsample])

X_imbalanced = [X_news[i] for i in imbalanced_indices]
y_imbalanced = y_news[imbalanced_indices]

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imbalanced, y_imbalanced, test_size=0.3, random_state=42
)

# Check imbalance
imb_train_counts = np.bincount(y_train_imb)
print(f"Imbalanced training set class distribution: {imb_train_counts}")
print(f"New imbalance ratio: {imb_train_counts.max() / imb_train_counts.min():.2f}")

# Compare standard Multinomial NB vs Complement NB
vectorizer_imb = CountVectorizer(max_features=1000, stop_words='english')
X_train_imb_vec = vectorizer_imb.fit_transform(X_train_imb)
X_test_imb_vec = vectorizer_imb.transform(X_test_imb)

# Standard Multinomial NB
mnb_standard = MultinomialNB(alpha=1.0)
mnb_standard.fit(X_train_imb_vec, y_train_imb)
y_pred_std = mnb_standard.predict(X_test_imb_vec)

# Complement NB (better for imbalanced data)
cnb = ComplementNB(alpha=1.0)
cnb.fit(X_train_imb_vec, y_train_imb)
y_pred_cnb = cnb.predict(X_test_imb_vec)

# Evaluate per class
for name, y_pred in [("Standard Multinomial NB", y_pred_std), ("Complement NB", y_pred_cnb)]:
    precisions, recalls, f1s, _ = precision_recall_fscore_support(y_test_imb, y_pred)
    accuracies = []
    for class_idx in range(len(categories)):
        class_accuracy = accuracy_score(
            y_test_imb == class_idx, 
            y_pred == class_idx
        )
        accuracies.append(class_accuracy)
    
    print(f"\n{name} per-class metrics:")
    for i, category in enumerate(categories):
        print(f"  Class '{category}':")
        print(f"    Precision: {precisions[i]:.4f}")
        print(f"    Recall: {recalls[i]:.4f}")
        print(f"    F1 Score: {f1s[i]:.4f}")
        print(f"    Accuracy: {accuracies[i]:.4f}")
    
    overall_accuracy = accuracy_score(y_test_imb, y_pred)
    print(f"  Overall Accuracy: {overall_accuracy:.4f}")

# Part 5: Feature Selection for Naive Bayes
print("\n=== Feature Selection for Naive Bayes ===")

# Create a pipeline with feature selection
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=5000, stop_words='english')),
    ('feature_selection', SelectKBest(chi2, k=1000)),  # Select top 1000 features
    ('classifier', MultinomialNB())
])

# Compare performance with and without feature selection
pipeline_no_selection = Pipeline([
    ('vectorizer', CountVectorizer(max_features=5000, stop_words='english')),
    ('classifier', MultinomialNB())
])

# Time and accuracy comparison
start_time = time.time()
pipeline_no_selection.fit(X_train_news, y_train_news)
no_selection_time = time.time() - start_time
no_selection_score = pipeline_no_selection.score(X_test_news, y_test_news)

start_time = time.time()
pipeline.fit(X_train_news, y_train_news)
selection_time = time.time() - start_time
selection_score = pipeline.score(X_test_news, y_test_news)

print(f"Without feature selection:")
print(f"  Training time: {no_selection_time:.4f} seconds")
print(f"  Accuracy: {no_selection_score:.4f}")

print(f"\nWith feature selection (k=1000):")
print(f"  Training time: {selection_time:.4f} seconds")
print(f"  Accuracy: {selection_score:.4f}")

# Part 6: Probability Calibration for Naive Bayes
print("\n=== Probability Calibration for Naive Bayes ===")

# Create a vectorized dataset
vectorizer_cal = CountVectorizer(max_features=1000, stop_words='english')
X_train_cal = vectorizer_cal.fit_transform(X_train_news)
X_test_cal = vectorizer_cal.transform(X_test_news)

# Create an uncalibrated classifier
uncal_nb = MultinomialNB(alpha=1.0)
uncal_nb.fit(X_train_cal, y_train_news)

# Create a calibrated classifier using isotonic regression
cal_nb = CalibratedClassifierCV(uncal_nb, method='isotonic', cv=5)
cal_nb.fit(X_train_cal, y_train_news)

# Get probabilities
uncal_probs = uncal_nb.predict_proba(X_test_cal)
cal_probs = cal_nb.predict_proba(X_test_cal)

# Plot calibration curves for one class (class 0)
class_idx = 0  # First class
plt.figure(figsize=(10, 6))

for i, (clf, name) in enumerate([(uncal_nb, 'Uncalibrated'), (cal_nb, 'Calibrated')]):
    y_prob = clf.predict_proba(X_test_cal)[:, class_idx]
    prob_true, prob_pred = calibration_curve(y_test_news == class_idx, y_prob, n_bins=10)
    
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', label=name)

plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title(f'Calibration curve for class: {categories[class_idx]}')
plt.legend()
plt.grid(True)
plt.show()

# Part 7: Visualizations
# 1. Feature importance across classes
vectorizer_final = CountVectorizer(max_features=1000, stop_words='english')
X_final = vectorizer_final.fit_transform(X_news)
feature_names_final = vectorizer_final.get_feature_names_out()

mnb_final = MultinomialNB()
mnb_final.fit(X_final, y_news)

# Get top features for each class
plt.figure(figsize=(15, 10))
n_top_features = 10

for class_idx, category in enumerate(categories):
    plt.subplot(len(categories), 1, class_idx + 1)
    top_indices = np.argsort(mnb_final.feature_log_prob_[class_idx])[-n_top_features:]
    top_features = [feature_names_final[i] for i in top_indices]
    top_weights = [mnb_final.feature_log_prob_[class_idx, i] for i in top_indices]
    
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_weights, align='center')
    plt.yticks(y_pos, top_features)
    plt.xlabel('Log Probability')
    plt.title(f'Top {n_top_features} features for class: {category}')

plt.tight_layout()
plt.show()

# 2. ROC curves for different variants of Naive Bayes
plt.figure(figsize=(10, 8))

# Prepare a common dataset for various Naive Bayes variants
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ensure data is appropriate for MultinomialNB and BernoulliNB
X_train_pos = X_train - X_train.min()  # Make features non-negative for Multinomial
X_test_pos = X_test - X_test.min()
X_train_bin = (X_train > 0).astype(np.int64)  # Binarize for Bernoulli
X_test_bin = (X_test > 0).astype(np.int64)

# Train different Naive Bayes variants
classifiers = {
    'Gaussian NB': GaussianNB().fit(X_train, y_train),
    'Multinomial NB': MultinomialNB().fit(X_train_pos, y_train),
    'Bernoulli NB': BernoulliNB().fit(X_train_bin, y_train),
    'Complement NB': ComplementNB().fit(X_train_pos, y_train)
}

# Plot ROC curve for each classifier
for name, clf in classifiers.items():
    if name == 'Multinomial NB' or name == 'Complement NB':
        y_prob = clf.predict_proba(X_test_pos)[:, 1]
    elif name == 'Bernoulli NB':
        y_prob = clf.predict_proba(X_test_bin)[:, 1]
    else:
        y_prob = clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Naive Bayes Variants')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 3. Comparison of Precision-Recall curves for imbalanced data
plt.figure(figsize=(10, 8))

# Get probabilities for standard MNB and CNB on imbalanced data
minority_class = np.argmin(imb_train_counts)

mnb_probs = mnb_standard.predict_proba(X_test_imb_vec)[:, minority_class]
cnb_probs = cnb.predict_proba(X_test_imb_vec)[:, minority_class]

# Calculate precision-recall curves
mnb_precision, mnb_recall, _ = precision_recall_curve(y_test_imb == minority_class, mnb_probs)
cnb_precision, cnb_recall, _ = precision_recall_curve(y_test_imb == minority_class, cnb_probs)

# Calculate area under PR curve
mnb_auc_pr = auc(mnb_recall, mnb_precision)
cnb_auc_pr = auc(cnb_recall, cnb_precision)

# Plot PR curves
plt.plot(mnb_recall, mnb_precision, lw=2, 
        label=f'Multinomial NB (AUC = {mnb_auc_pr:.3f})')
plt.plot(cnb_recall, cnb_precision, lw=2, 
        label=f'Complement NB (AUC = {cnb_auc_pr:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve for Minority Class: {categories[minority_class]}')
plt.legend(loc="best")
plt.grid(True)
plt.show()
```

**Advanced Naive Bayes Concepts:**

1. **Variants and Their Applications:**
   - **Bernoulli NB**: For binary features (presence/absence)
   - **Multinomial NB**: For discrete count features (word frequencies)
   - **Gaussian NB**: For continuous features with normal distribution
   - **Complement NB**: Specifically for imbalanced datasets

2. **Laplace Smoothing (Additive Smoothing):**
   - Formula: P(x|y) = (count(x,y) + α) / (count(y) + α × |V|)
   - Where α is the smoothing parameter and |V| is the feature vocabulary size
   - Prevents zero probabilities for unseen feature-class combinations

3. **Log Space Computation:**
   - Using log probabilities: log(P(y|X)) = log(P(y)) + Σ log(P(xᵢ|y))
   - Prevents numerical underflow with many features
   - Converts multiplication to addition for computational efficiency

4. **Feature Selection Metrics:**
   - Chi-squared (χ²): Measures dependence between feature and class
   - Mutual Information: Quantifies information gain about the class from the feature
   - Information Gain Ratio: Normalized version of mutual information

5. **Handling Class Imbalance:**
   - Complement Naive Bayes: Uses complement of class data for parameter estimation
   - Weighting classes: Adjusting prior probabilities based on desired class weights

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to determine the optimal smoothing parameter (alpha) for different types of datasets and Naive Bayes variants, since it can significantly impact model performance particularly with sparse features?

How can I apply this to a real project or problem?
I could implement an email classification system that not only identifies spam but classifies legitimate emails into different categories (work, personal, promotions), using Complement Naive Bayes to handle class imbalance, feature selection to improve efficiency, and Laplace smoothing with optimized alpha values to handle the sparse nature of text data.

What's a common misconception or edge case?
A common misconception is that probability calibration isn't necessary for Naive Bayes. In reality, while Naive Bayes provides probabilistic outputs, they're often poorly calibrated due to the independence assumption, especially for Gaussian Naive Bayes. The probabilities tend to be pushed toward 0 or 1, making them unreliable for decision-making when accurate probability estimates are required. Calibration techniques like Platt scaling or isotonic regression can significantly improve the reliability of these probability estimates.

The key idea behind Naive Bayes Classifier Algorithm Part 2 is {{enhancing basic Naive Bayes with techniques like smoothing, feature selection, and variant selection to overcome practical limitations and optimize performance for specific data characteristics}}.

---
##### Tags

#ai/Naive_Bayes_Classifier_Part_2 #ai #python #flashcard 