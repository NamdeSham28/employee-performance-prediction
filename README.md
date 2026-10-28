Machine Learning-Based Employee Performance Prediction

1. Overview

Employee performance affects the productivity and success of an organization. Traditional evaluation relies on manual reviews and manager observations, which are slow and subjective. This project uses machine learning to classify employees into three performance categories: Low, Medium and High, based on employee-related attributes.

2. Objectives
Collect and prepare employee-related data
Preprocess and clean the data
Analyze performance patterns using visualizations
Identify the most important performance-related features
Train a Random Forest classifier
Predict employee performance categories
Evaluate the model (accuracy, precision, recall, F1-score, confusion matrix)
3. Dataset

A synthetic dataset of 1,000 employee records and 13 columns, generated in the notebook (seed = 42) and saved as employee_performance_dataset.csv.

Column	Description
Employee_ID	Unique ID (EMP0001 to EMP1000)
Age	21 to 60 years
Experience_Years	0 to 20 years
Attendance_Percent	70 to 100 %
Training_Hours	0 to 50 hours
Working_Hours	6 to 12 hours per day
Job_Satisfaction	Rating 1 to 5
Overtime_Hours	0 to 20 hours
Projects_Completed	1 to 15
Previous_Performance	Score 50 to 100
Department	IT, HR, Finance, Sales, Operations
Performance_Score	Weighted score (0 to 100) computed from the factors above plus noise
Performance_Category	Low / Medium / High (based on the 33rd and 67th percentiles of the score)
4. Methodology
Employee Data -> Preprocessing -> Train/Test Split (80:20) -> Random Forest -> Evaluation -> Visualization -> Prediction
Dataset generation: create 1,000 records and compute the performance score.
Preprocessing: drop ID/score columns, one-hot encode Department.
Split: 800 training and 200 testing records (stratified).
Model: RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced").
Evaluation: accuracy, classification report, confusion matrix.
Visualization: performance distribution, feature importance, scatter plots.
Prediction: predict the category and confidence for a new employee.
5. Tech Stack
Python 3
Pandas, NumPy
Scikit-learn
Matplotlib
Google Colab
6. Project Structure
employee-performance-prediction/
├── README.md
├── requirements.txt
├── employee_performance_prediction.ipynb
├── employee_performance_dataset.csv
├── images/
└── docs/
7. How to Run

Google Colab (recommended)

Open employee_performance_prediction.ipynb in Google Colab.
Click Runtime -> Run all.

Locally

bash
git clone https://github.com/<your-username>/employee-performance-prediction.git
cd employee-performance-prediction
pip install -r requirements.txt
jupyter notebook employee_performance_prediction.ipynb
8. Results
Random Forest accuracy: XX.XX % (replace with your Colab output)
Confusion Matrix	Feature Importance
Confusion Matrix	Feature Importance

Other plots (see images/): performance distribution, Experience vs Score, Training Hours vs Score, Attendance vs Score.

9. Limitations
The dataset is synthetic, so accuracy will not directly carry over to real HR data.
Prediction quality depends on data quality; biased data leads to biased predictions.
Employee data must be protected, and the model needs retraining when patterns change.
Predictions are decision support only and should not drive promotion or termination decisions on their own.
10. Future Scope
Use authorized real-world employee data
Compare with Decision Tree, Logistic Regression and SVM
Hyperparameter tuning and feature engineering
Privacy and fairness evaluation
Test on larger, more diverse datasets
11. References
Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.
Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.
McKinney, W. (2010). Data Structures for Statistical Computing in Python. SciPy Conference, 56-61.
Harris, C. R., et al. (2020). Array Programming with NumPy. Nature, 585, 357-362.
Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95.
