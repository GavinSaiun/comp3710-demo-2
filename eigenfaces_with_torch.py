import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
# Download the data and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Split into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Center the data by subtracting the mean
mean = torch.mean(X_train, dim=0)
X_train -= mean
X_test -= mean

# Compute SVD
U, S, V = torch.linalg.svd(X_train, full_matrices=False)

# Select the top components
n_components = 150
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))

# Project the data into the PCA subspace
X_transformed = torch.matmul(X_train, components.T)
X_test_transformed = torch.matmul(X_test, components.T)

print(X_transformed.shape)
print(X_test_transformed.shape)

# Plot the eigenfaces
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces.detach().numpy(), eigenface_titles, h, w)
plt.show()

# Plot explained variance
explained_variance = (S ** 2) / (n_samples - 1)
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = torch.cumsum(explained_variance_ratio, dim=0)
eigenvalueCount = torch.arange(n_components)

plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()

# Build Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Convert the transformed data to numpy arrays for RandomForest
X_transformed_np = X_transformed.detach().numpy()
X_test_transformed_np = X_test_transformed.detach().numpy()

estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed_np, y_train)

# Predict on the test data
predictions = estimator.predict(X_test_transformed_np)
correct = predictions == y_test
total_test = len(X_test_transformed_np)

print("Total Testing:", total_test)
print("Predictions:", predictions)
print("Which Correct:", correct)
print("Total Correct:", np.sum(correct))
print("Accuracy:", np.sum(correct) / total_test)
print(classification_report(y_test, predictions, target_names=target_names))
