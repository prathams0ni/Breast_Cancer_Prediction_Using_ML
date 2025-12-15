# 🧬 Breast Cancer Prediction using Machine Learning (Classification Project)

## 📖 Project Description

Breast cancer is one of the most common and life‑threatening diseases among women worldwide. Early and accurate detection plays a crucial role in improving survival rates and treatment outcomes.

This project focuses on building a **machine learning classification model** that predicts whether a breast cell tumor is **Malignant (Cancerous)** or **Benign (Non‑cancerous)** using various numerical features extracted from cell nucleus measurements. The dataset used is a well‑known open dataset frequently applied in healthcare analytics and machine learning research.

The complete workflow includes **data preprocessing, exploratory data analysis (EDA), feature selection, model training, and performance evaluation**. The trained model can be integrated into real‑world applications such as hospital systems, diagnostic tools, or web‑based medical interfaces.

---

## 🎯 Problem Statement

To develop a machine learning model that can accurately classify breast cancer tumors as **Malignant** or **Benign** based on diagnostic features.

---

## 🗂 Dataset Information

* **Dataset Name:** Breast Cancer Diagnostic Dataset
* **Source:** Open Dataset (CSV format)
* **Target Variable:** `diagnosis`

  * `M` → Malignant
  * `B` → Benign

### 🔢 Features

The dataset contains multiple numerical features derived from cell nuclei measurements such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

---

## ⚙️ Technologies & Libraries Used

```bash
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

---

## 🧪 Machine Learning Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

## 🔄 Project Workflow

```text
1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Performance Comparison
```

---

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

These metrics help ensure that the model is not just accurate but also reliable for medical decision‑making.

---

## 🏆 Best Model Selection

Among all the trained models, **Random Forest Classifier** performed the best due to:

* Higher accuracy
* Better generalization
* Lower overfitting compared to Decision Tree

---

## 🚀 How to Run the Project

```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/breast-cancer-prediction-ml.git

# Step 2: Navigate to the project directory
cd breast-cancer-prediction-ml

# Step 3: Install required libraries
pip install -r requirements.txt

# Step 4: Open Jupyter Notebook
jupyter notebook

# Step 5: Run the notebook
Cancer Prediction.ipynb
```

---

## 📌 Project Structure

```text
📦 Breast-Cancer-Prediction
 ┣ 📜 Cancer Prediction.ipynb
 ┣ 📜 data.csv
 ┣ 📜 README.md
```

---

## 🧠 Key Insights

* Malignant tumors show higher mean radius and perimeter values
* Feature scaling significantly improves model performance
* Ensemble models outperform individual classifiers
* Random Forest provides stable and robust predictions

---

## 🏥 Real‑World Applications

* Medical diagnosis support systems
* Hospital decision‑making tools
* Web‑based cancer screening applications
* AI‑powered healthcare analytics

---

## 📌 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Model deployment using Flask or FastAPI
* Integration with real‑time medical devices
* Explainable AI (SHAP, LIME) for medical transparency

---

## ⭐ Acknowledgment

This project is created for learning and academic purposes using publicly available healthcare datasets.

If you find this project useful, please ⭐ the repository!
