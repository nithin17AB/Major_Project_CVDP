❤️ Heart Disease Prediction using Machine Learning

📌 Project Overview

Heart disease (cardiovascular disease) is one of the leading causes of death worldwide. This project aims to build a machine learning model that can predict the presence of heart disease based on various health parameters.

The system analyzes medical attributes such as age, blood pressure, cholesterol levels, and lifestyle factors to provide accurate predictions.

---

🎯 Objectives

- Perform data preprocessing and cleaning
- Analyze dataset using visualization techniques
- Identify relationships between features using correlation matrix
- Apply multiple machine learning algorithms
- Compare model performance
- Build an accurate prediction model

---

📂 Dataset

- Dataset used: Cardio Dataset
- Features include:
  - Age
  - Gender
  - Height & Weight
  - Blood Pressure (ap_hi, ap_lo)
  - Cholesterol
  - Glucose
  - Smoking, Alcohol, Physical Activity
- Target variable: cardio (0 = No Disease, 1 = Disease)

---

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

🔍 Data Preprocessing

- Handled dataset formatting issues (semicolon-separated values)
- Checked for missing values
- Cleaned and structured data
- Feature selection and separation (X and y)

---

📊 Data Visualization

- Count plots for disease distribution
- Age distribution histogram
- Gender vs heart disease comparison
- Multiple plots to extract meaningful insights

---

🔗 Correlation Matrix

- Heatmap used to visualize relationships between features
- Helps in identifying important variables affecting heart disease

---

🤖 Machine Learning Models Used

Model| Description
Logistic Regression| Baseline classification model
K-Nearest Neighbors| Distance-based classification
Support Vector Machine| Margin-based classifier
Decision Tree| Tree-based model
Random Forest| Ensemble learning model

---

📈 Model Performance

Model| Accuracy
Logistic Regression| ~71%
KNN| ~55%
SVM| ~59%
Decision Tree| ~63%
Random Forest| ~72% ✅

---

🏆 Final Model

- Random Forest Classifier was selected as the final model
- Achieved highest accuracy among all models
- Handles complex data effectively and reduces overfitting

---

💡 Future Improvements

- Hyperparameter tuning for better accuracy
- Deploy as a web application
- Add real-time user input prediction system
- Use deep learning models for advanced analysis

---

📌 Conclusion

This project demonstrates how machine learning techniques can be applied to predict heart disease effectively. Among all models, Random Forest provided the best performance and can be used for reliable predictions.

---
