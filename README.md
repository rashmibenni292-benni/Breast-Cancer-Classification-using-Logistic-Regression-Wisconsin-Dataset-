

# 🩺 Breast Cancer Classification

## 📋 Table of Contents

* [Project Overview](https://www.google.com/search?q=%23project-overview)
* [Project Objective](https://www.google.com/search?q=%23project-objective)
* [Why Logistic Regression?](https://www.google.com/search?q=%23why-logistic-regression)


* [Project Structure](https://www.google.com/search?q=%23project-structure)
* [Folder Explanation](https://www.google.com/search?q=%23folder-explanation)


* [Dataset Information](https://www.google.com/search?q=%23dataset-information)
* [Exploratory Data Analysis (EDA)](https://www.google.com/search?q=%23exploratory-data-analysis-eda)
* [Data Preprocessing](https://www.google.com/search?q=%23data-preprocessing)
* [Logistic Regression — Theory & Intuition](https://www.google.com/search?q=%23logistic-regression--theory--intuition)
* [Model Implementations](https://www.google.com/search?q=%23model-implementations)
* [Model Evaluation](https://www.google.com/search?q=%23model-evaluation)
* [Interpretation](https://www.google.com/search?q=%23interpretation)
* [Residual Analysis & Assumptions](https://www.google.com/search?q=%23residual-analysis--assumptions)
* [Model Limitations](https://www.google.com/search?q=%23model-limitations)
* [Possible Improvements](https://www.google.com/search?q=%23possible-improvements)

---

## 📌 Project Overview

### Project Objective

The goal is to develop a predictive model that classifies breast mass tumors as **Malignant** or **Benign**. By analyzing nuclear features from Fine Needle Aspirate (FNA) images, this project aims to provide a data-driven approach to assist in clinical diagnostics.

### Why Logistic Regression?

Unlike Linear Regression, which predicts continuous values, **Logistic Regression** is the industry standard for binary classification. It is computationally efficient, highly interpretable (essential in medicine), and outputs probabilities that allow for adjustable risk thresholds.

---

## 📂 Project Structure

```text
├── data/
│   └── breast_cancer.csv       
├── notebooks/
│   └── logistic_Regression.ipynb 
├── src/
│   ├── logistic_Regression.py                     
├── README.md                 
└── requirements.txt          
```

### Folder Explanation

* **`data/`**: Contains the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
* **`notebooks/`**: Interactive walkthrough of the entire data science pipeline.
* **`src/`**: Modularized code for reusability and production-like environment.

---

## 🧬 Dataset Information

The dataset consists of **569 instances** with **30 numeric features**.

* **Target:** `Diagnosis` (M = Malignant, B = Benign).
* **Features:** Ten real-valued features are computed for each cell nucleus, including Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry, and Fractal dimension.

---

## 📊 Exploratory Data Analysis (EDA)

### Key EDA Steps

1. **Class Balance:** Checking the ratio of Malignant vs. Benign cases using `plotly.express` histograms.
2. **Correlation Mapping:** Identifying multicollinearity between features (e.g., Radius vs. Area).
3. **Feature Distribution:** Visualizing how specific features like `mean_concave_points` differ between classes.

---

## ⚙️ Data Preprocessing

* **Label Encoding:** Mapping 'M' to 1 and 'B' to 0.
* **Handling Missing Values:** Dataset verification shows zero null values.
* **Feature Scaling:** Implementing standard scaling to ensure the gradient descent converges faster.

---

## 🧠 Logistic Regression — Theory & Intuition

### Model Equation

The model applies the **Sigmoid function** to a linear combination of inputs:



Where 

### Objective Function (Log Loss)

We use the **Binary Cross-Entropy** loss function to penalize incorrect classifications:


### Optimization

Weights are optimized using **Gradient Descent**, iteratively moving toward the global minimum of the loss curve.

---

## 🛠 Model Implementations

### Linear Regression From Scratch

* Manual initialization of weights.
* Custom `sigmoid` and `cost_function` methods.
* Manual gradient updates.

### Linear Regression using scikit-learn

* Utilizing `LogisticRegression()` from `sklearn.linear_model`.
* Implementing `train_test_split` for validation.

---

## 📈 Model Evaluation

| Metric | Result |
| --- | --- |
| **Accuracy** | 98.23% |
| **Precision** | 100.00% |
| **Recall** | 95.24% |
| **F1-Score** | 97.56% |

---

## 🔍 Interpretation

The model shows an exceptional **Precision of 100%**, meaning it never falsely identified a benign tumor as malignant in the test set. However, a **Recall of 95%** suggests that while rare, some malignant cases could be missed, which is the primary focus of medical model tuning.

---

## 📉 Residual Analysis & Assumptions

* **Linearity:** Assumes a linear relationship between the log-odds of the dependent variable and the independent variables.
* **Independence:** Observations are independent of each other.
* **No Multicollinearity:** Features like 'Area' and 'Perimeter' are highly correlated, which can sometimes inflate the variance of coefficient estimates.

---

## ⚠️ Model Limitations

* **Linear Decision Boundary:** If the data were non-linearly separable, a simple Logistic Regression might underperform.
* **Outlier Sensitivity:** Can be swayed by extreme values in the clinical data.

## 🚀 Possible Improvements

* **L2 Regularization (Ridge):** To handle the high multicollinearity found in cellular measurements.
* **Hyperparameter Tuning:** Using `GridSearchCV` to find the optimal solver.
* **Polynomial Features:** Introducing non-linear terms to capture more complex biological relationships.
