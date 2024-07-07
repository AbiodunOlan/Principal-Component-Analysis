# Principal-Component-Analysis

# Breast Cancer Diagnosis with PCA and Logistic Regression

This project aims to predict whether a breast cancer diagnosis is malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset. The dataset is reduced to 2 principal components using PCA, and logistic regression is applied for classification.

## Dataset

The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, available from the UCI Machine Learning Repository. It consists of features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

- **Number of instances**: 569
- **Number of features**: 30
- **Target variable**: Diagnosis (M = malignant, B = benign)
- **No missing values**

### Features

The features represent various characteristics of the cell nuclei present in the image:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- matplotlib
- seaborn
- ucimlrepo

## Installation

Install the required libraries using pip:

```sh
pip install pandas scikit-learn matplotlib seaborn ucimlrepo




Usage
Step 1: Fetch the Dataset
The dataset is fetched using the ucimlrepo library.

python
Copy code
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
Step 2: Preprocess the Data
Standardize the features and apply PCA to reduce the dimensionality to 2 components.

python
Copy code
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Access the data (features and targets)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# Convert the target variable 'Diagnosis' to numerical values
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y['Diagnosis'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
Step 3: Split the Data
Split the dataset into training and testing sets.

python
Copy code
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_numeric, test_size=0.3, random_state=42)
Step 4: Train the Logistic Regression Model
Train a logistic regression model on the training set.

python
Copy code
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
Step 5: Evaluate the Model
Make predictions on the test set and evaluate the model's performance.

python
Copy code
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions
y_pred = logistic_regression.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
Results
The logistic regression model's performance is evaluated using accuracy, classification report, and confusion matrix. The PCA reduces the dataset to 2 dimensions, making it easier to visualize and understand.
