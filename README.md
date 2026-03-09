# Vehicle Classification using PCA and Machine Learning

## Overview

This project explores vehicle classification using machine learning and Principal Component Analysis (PCA). The dataset contains numerical measurements describing vehicle shape and geometry, with the goal of classifying each sample into one of three classes:

- Bus
- Car
- Van

The workflow includes data preprocessing, label encoding, dimensionality reduction with PCA, 3D visualisation of the transformed feature space, and classification using Gaussian Naive Bayes and Logistic Regression.

---

## Key Results

- PCA reduced the original 18-dimensional feature space to **5 principal components** while retaining approximately **90% of the variance**
- **Logistic Regression** achieved the best validation accuracy at **71%**
- **Gaussian Naive Bayes** generalised slightly better on the held-out test set with **65% accuracy**
- The **Bus** class was the most difficult to classify consistently
- Feature overlap between vehicle types limited class separation and led to misclassifications

---

## Dataset

The dataset contains **18 numerical input features** and one target column named `class`.

After preprocessing and removal of missing values, the final dataset used in the analysis contained:

- **813 samples**
- **18 input features**
- **3 target classes**

The target classes were label encoded as:

- `0` = Bus
- `1` = Car
- `2` = Van

---

## Project Workflow

### 1. Data Preprocessing
The dataset was loaded into a pandas DataFrame and prepared for modelling by:

- checking data structure and feature types
- identifying missing values
- removing rows containing missing data
- encoding the target variable using `LabelEncoder`

### 2. Train / Validation / Test Split
The cleaned dataset was split into:

- **60% training**
- **20% validation**
- **20% testing**

This allowed model development on the training data, model comparison on the validation data, and final evaluation on the unseen test data.

### 3. Standardisation
Before PCA, the feature values were standardised using `StandardScaler` so that all variables contributed on a comparable scale.

### 4. Principal Component Analysis
PCA was applied to the standardised feature space to reduce dimensionality while preserving as much variance as possible.

The analysis retained **5 principal components**, which together explained approximately **90% of the total variance**.

### 5. Visualisation
The transformed feature space was visualised in 3D using the first three principal components. Additional PCA loading analysis was used to interpret how the original features contributed to class separation.

### 6. Classification
Two machine learning models were trained on the PCA-transformed data:

- Gaussian Naive Bayes
- Logistic Regression

These models were then evaluated on both validation and test sets.

---

## PCA Results

PCA retained **5 principal components** to explain approximately **90% of the variance** in the dataset.

Explained variance ratios for the first five principal components:

- **PC1**: 0.5209
- **PC2**: 0.1727
- **PC3**: 0.0993
- **PC4**: 0.0655
- **PC5**: 0.0537

### Scree Plot

![Scree Plot](outputs/scree_plot.png)

---

## 3D PCA Visualisations

### Training Set Projection
![3D PCA Training Plot](outputs/pca_3d_train.png)

### Test Set Projection
![3D PCA Test Plot](outputs/pca_3d_test.png)

### PCA Loadings Plot
![3D Loadings Plot](outputs/loadings_3d.png)

---

## Model Performance and Insights

Two classification models were trained on the PCA-transformed dataset:

- Gaussian Naive Bayes
- Logistic Regression

### Model Performance

| Model | Validation Accuracy | Test Accuracy |
|------|------:|------:|
| Gaussian Naive Bayes | 69.33% | 65.03% |
| Logistic Regression | 71.17% | 61.35% |

Both models performed slightly better on the validation set than on the held-out test set, indicating some degree of overfitting.

The **Bus** class showed the largest performance drop between datasets, achieving **61% accuracy on the validation set but only 33% on the test set**, suggesting that the model struggled to generalise this class effectively.

---

## Feature Insights from PCA

Analysis of the PCA loading vectors provided insight into which features contributed most strongly to each vehicle class.

### Van
The **Van** class was primarily influenced by **Elongatedness**, suggesting that vehicle length relative to width was an important distinguishing characteristic.

### Bus
The **Bus** class was influenced by features associated with **negative PC2 and PC3 values**, particularly:

- Circularity
- Scaled Radius of Gyration

These features contributed to bus clustering in regions where **PC1 values were greater than 4**.

### Car
The **Car** class was influenced by features with high absolute values along **PC1 and PC2**, including:

- Compactness
- Circularity

These features helped distinguish cars from the other vehicle classes in PCA space.

---

## Class Separation

The PCA projections showed **significant overlap between the three classes**. This is likely due to shared structural characteristics between vehicle types, especially features such as:

- Elongatedness
- Scaled Radius of Gyration

This overlap made classification more difficult and contributed to several misclassifications, particularly for the **Bus** class.

Despite this, the **Car** class was classified relatively well, suggesting that its feature distribution was more distinct within the PCA-transformed feature space.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Jupyter Notebook

---

## How to Run

### 1. Clone the repository

```
git clone https://github.com/benjmoh/vehicle-classification-pca.git
cd vehicle-classification-pca
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```
jupyter notebook
```

### 4. Open the notebook

Open:

```
vehicle-classification-pca.ipynb
```

Run the notebook cells from top to bottom to reproduce the full analysis, including:

- data preprocessing  
- label encoding  
- train / validation / test splitting  
- feature standardisation  
- principal component analysis (PCA)  
- 3D PCA visualisations  
- model training  
- model evaluation


