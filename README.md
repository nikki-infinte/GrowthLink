# Titanic Survival Prediction - Logistic Regression & Random Forest

## Project Overview

This project builds machine learning models to predict passenger survival on the Titanic based on various features like age, sex, fare, and class. The dataset used is the Titanic dataset available in `titanic_dataset.csv`. Two models are used: **Logistic Regression** and **Random Forest Regression**.

## Dataset Description

The dataset consists of the following relevant features:

- **Survived**: Target variable (0 = No, 1 = Yes)
- Machine learning models are used to predict passenger survival on the Titanic based on various features.

  **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Preprocessing Steps

1. **Drop Irrelevant Columns**: `PassengerId`, `Name`, `Ticket`, and `Cabin` are removed.
2. **Handle Missing Values**:
   - `Age`: Filled with the median value.
   - `Fare`: Filled with the median value.
   - `Embarked`: Filled with the most frequent value (mode).
3. **One-Hot Encoding**:
   - `Sex` and `Embarked` are converted into numerical features.
4. **Feature Scaling**:
   - `Age` and `Fare` are standardized using `StandardScaler()`.

## Model Training

### Logistic Regression

- A **Logistic Regression** model is used with a preprocessing pipeline.
- The dataset is split into **80% training** and **20% testing**.
- The model is trained using `LogisticRegression(solver='liblinear')`.

### Random Forest Classification

- A **Random Forest Classifier** is also used for comparison.
- It consists of multiple decision trees and predicts survival probability.
- The model is trained using `RandomForestClassifier(n_estimators=100, random_state=42)`.

## Preprocessing Pipeline

To streamline data preparation, a **pipeline** is used for preprocessing:

- **Numeric Features**: `StandardScaler()` is applied to `Age` and `Fare`.
- **Categorical Features**: `OneHotEncoder()` is used for `Sex` and `Embarked`.
- The `ColumnTransformer` ensures all transformations are applied in one step before feeding data into models.

## Evaluation Metrics

Both models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **Cross-Validation Scores**

## Model Performance

### Logistic Regression Performance:

**Accuracy**: 0.8101

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.83      0.86      0.84       105
           1       0.79      0.74      0.76        74

    accuracy                           0.81       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179
```

**Confusion Matrix:**

```
[[90 15]
 [19 55]]
```

### Random Forest Performance:

**Accuracy**: 0.8212

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.83      0.87      0.85       105
           1       0.80      0.76      0.78        74

    accuracy                           0.82       179
   macro avg       0.82      0.81      0.81       179
weighted avg       0.82      0.82      0.82       179
```

**Confusion Matrix:**

```
[[91 14]
 [18 56]]
```

### Cross-Validation Scores:

- **Logistic Regression**: CV Accuracy: **0.7912 ± 0.0185**
- **Random Forest**: CV Accuracy: **0.8137 ± 0.0191**

## How to Run the Project

1. Install dependencies:
   ```sh
   pip install pandas numpy seaborn scikit-learn joblib
   ```
2. Place `titanic_dataset.csv` in the project directory.
3. Run the Python script:
   ```sh
   python titanic_model.py
   ```
4. The models will be saved as `logistic_regression_titanic.pkl` and `random_forest_titanic.pkl`.

## Future Improvements

- Experiment with additional classification algorithms.
- Tune hyperparameters for better performance.
- Add feature engineering techniques to improve predictions.
- Use cross-validation for more robust model evaluation.


