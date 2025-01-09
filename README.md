
# **Credit Card Fraud Detection**

> **A machine learning-based system designed to accurately detect fraudulent transactions using advanced algorithms and data preprocessing techniques.**

---

## **Table of Contents**

1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Future Scope](#future-scope)
10. [License](#license)

---

## **Introduction**

Credit card fraud poses a major challenge in the digital financial world. This project leverages machine learning algorithms to develop an efficient fraud detection system capable of identifying fraudulent transactions in real-time. The system addresses key challenges such as data imbalance and scalability.

---

## **Features**

- Handles highly imbalanced datasets using **SMOTE (Synthetic Minority Oversampling Technique)**.
- Implements advanced machine learning algorithms:
  - **XGBoost**, **CatBoost**, **LightGBM**, **Random Forest**, and **Artificial Neural Networks (ANNs)**.
- Supports **real-time fraud detection**.
- Provides detailed **performance evaluation** metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Visualizes data insights with heatmaps, confusion matrices, and ROC curves.

---

## **Technologies Used**

- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `imbalanced-learn`, `XGBoost`, `LightGBM`, `CatBoost`
- **Tools**:
  - Jupyter Notebook
  - GitHub for version control

---

## **Project Structure**

```

Credit-Card-Fraud-Detection/
│
├── data/                     # (Optional) Include a sample dataset or a README pointing to the source (like Kaggle)
├── images/                   # Graphs, confusion matrices, and model comparison visualizations
├── notebooks/                # Jupyter notebooks for exploration and model training
├── src/                      # Python scripts for data preprocessing, training, and evaluation
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License file (optional)

```

---

## **Dataset**

The dataset was sourced from [Kaggle](https://www.kaggle.com) and contains anonymized credit card transaction data. Key characteristics:
- **Transactions**: 284,807
- **Fraudulent Transactions**: 492 (~0.17%)
- **Features**: 30 anonymized features, including `Time` and `Amount`.

**Note**: Sensitive data is excluded for privacy.

---

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

1. **Preprocess the Data**:
   - Clean, scale, and balance the dataset using `src/preprocess.py`.

2. **Train the Models**:
   - Run `src/train_models.py` to train multiple machine learning models.
   - Example:
     ```bash
     python src/train_models.py --model xgboost
     ```

3. **Evaluate Models**:
   - Evaluate models with `src/evaluate.py` to generate performance metrics and visualizations.
 

---

## **Results**

### **Model Performance**

| Model       | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------|----------|-----------|--------|----------|---------|
| XGBoost     | 99.96%   | 95.0      | 82.0   | 88.0     | 0.999   |
| CatBoost    | 99.96%   | 93.0      | 82.0   | 87.0     | 0.998   |
| RandomForest| 99.95%   | 89.0      | 81.0   | 85.0     | 0.997   |
| ANN         | 99.95%   | 85.0      | 81.0   | 83.0     | 0.996   |

### **Visualizations**
- **Confusion Matrix**:
  ![Confusion Matrix](images/confusion_matrix.png)
- **ROC Curve**:
  ![ROC Curve](images/roc_curve.png)

---

## **Future Scope**

- **Real-Time Processing**: Implement streaming capabilities for live transaction data.
- **Deep Learning**: Explore architectures like **LSTMs** for sequential fraud detection.
- **Global Deployment**: Extend support for multi-currency and multi-lingual data.

---


## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
