
# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using models like XGBoost, LightGBM, and ANNs, with SMOTE for handling imbalanced datasets. The system emphasizes accuracy, precision, and user trust through advanced evaluation and visualization tools.

A machine learning project focused on detecting fraudulent credit card transactions using advanced models like XGBoost, LightGBM, and Artificial Neural Networks (ANNs). The system leverages imbalanced data handling techniques (SMOTE) and evaluates performance with metrics like F1-score and precision.

## Features

- **Imbalanced Data Handling**: Uses SMOTE to balance the dataset.
- **Multi-Model Comparison**: Compares models such as XGBoost, LightGBM, Random Forest, and ANN.
- **Comprehensive Evaluation**: Evaluates accuracy, precision, recall, and F1-score.
- **Visualization Tools**: Provides insights into model performance and data distribution.

## Tools & Technologies

- **Programming Language**: Python
- **Libraries**: TensorFlow, Scikit-learn, XGBoost, LightGBM, Matplotlib, Pandas, NumPy

## Project Structure

```
credit-card-fraud-detection/
├── README.md              # Project overview and instructions
├── LICENSE                # Project license
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored files and folders
├── data/                  # Dataset folder (sample or linked)
│   ├── sample_data.csv    # Example dataset (if sharable)
├── models/                # Model-specific scripts
│   ├── xgboost_model.py
│   ├── ann_model.py
├── results/               # Results and visualizations
│   ├── performance_metrics.csv
│   ├── confusion_matrix.png
├── report/                # Project report
│   ├── final_report.pdf
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd credit-card-fraud-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Project**:
   Execute the main Python script to start the system:
   ```bash
   python main.py
   ```

4. **View Results**:
   Check the `results/` folder for metrics and visualizations.

## Results

- **Best Model**: XGBoost achieved the highest F1-score (88%) with balanced precision and recall.
- **Performance Visualizations**: ROC curves and confusion matrices demonstrate model performance.

## Future Enhancements

- **Real-Time Processing**: Implement streaming data handling.
- **Deep Learning Models**: Explore architectures like LSTM for sequential data.
- **Global Adaptation**: Enable multi-currency and multi-lingual support.


---

## LICENSE

MIT License


