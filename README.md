# Internship Task

This repository contains tasks completed as part of an internship program. Each task involves data analysis, preprocessing, and machine learning techniques. Below is a detailed explanation of each task:

## Task 2: Sentiment Analysis on IMDB Dataset
- **Objective**: Perform sentiment analysis on movie reviews.
- **Details**:
  1. **Imports and Data Loading**:
     - Imported libraries: `nltk`, `pandas`, `sklearn`, and `re`.
     - Downloaded necessary NLTK resources (`stopwords`, `punkt`, `wordnet`).
     - Loaded the IMDB dataset using `sklearn.datasets.load_files`.
  2. **Text Preprocessing**:
     - Defined a `preprocess` function to:
       - Convert text to lowercase.
       - Remove special characters using regex.
       - Tokenize text using NLTK.
       - Remove stopwords and lemmatize tokens.
     - Applied preprocessing to all reviews.
  3. **Feature Engineering**:
     - Used `TfidfVectorizer` to extract features from the cleaned text.
  4. **Model Training**:
     - Split the dataset into training and testing sets.
     - Trained a `LogisticRegression` model on the training data.
  5. **Model Evaluation**:
     - Predicted sentiments on the test set.
     - Evaluated the model using a classification report.

## Task 3: Credit Card Fraud Detection
- **Objective**: Detect fraudulent transactions in a credit card dataset.
- **Details**:
  1. **Data Loading**:
     - Loaded the `creditcard_reduced.csv` dataset using `pandas`.
     - Displayed the first few rows and class distribution.
  2. **Handling Class Imbalance**:
     - Used `SMOTE` to oversample the minority class.
     - Split the dataset into training and testing sets.
  3. **Model Training**:
     - Trained a `RandomForestClassifier` with 20 estimators on the resampled data.
  4. **Model Evaluation**:
     - Predicted classes on the test set.
     - Evaluated the model using a classification report.
  5. **Test Transaction Function**:
     - Created a function to simulate a random transaction and predict whether it is fraudulent or legitimate.

## Task 4: Housing Price Prediction
- **Objective**: Predict housing prices using regression models.
- **Details**:
  1. **Data Loading and Preprocessing**:
     - Loaded the `HousingData.csv` dataset.
     - Normalized numerical features and one-hot encoded categorical columns.
     - Dropped missing values.
  2. **Model Training**:
     - Implemented a custom `SimpleXGBoostRegressor` class.
     - Trained a `RandomForestRegressor` and the custom XGBoost model.
  3. **Feature Importance**:
     - Calculated feature importance for both models.
     - Visualized feature importance using bar plots.
  4. **Model Evaluation**:
     - Evaluated models using RMSE and R² metrics.
     - Visualized residuals and actual vs. predicted values.


## Titanic Dataset Analysis
- **Objective**: Perform exploratory data analysis (EDA) on the Titanic dataset.
- **Details**:
  1. **Data Loading**:
     - Loaded the Titanic dataset from a CSV file.
     - Displayed the first few rows and dataset information.
  2. **Data Cleaning**:
     - Handled missing values in the `Age` and `Embarked` columns.
     - Removed duplicate rows.
  3. **Outlier Detection and Removal**:
     - Identified outliers in the `Fare` and `Age` columns using the IQR method.
     - Filtered out rows with outliers.
  4. **Data Visualization**:
     - Created boxplots to visualize outliers.
     - Plotted distributions of `Sex`, `Pclass`, and `Age`.
     - Generated a heatmap to show correlations between numerical features.
  5. **Survival Analysis**:
     - Analyzed survival rates by gender and passenger class.
     - Visualized survival rates using count plots.

## Datasets
- `datasets/creditcard_reduced.csv`: Used for credit card fraud detection.
- `datasets/HousingData.csv`: Reserved for future tasks.

## How to Run
1. Clone the repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebooks and execute the cells sequentially.

## Requirements
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, imbalanced-learn

## Notes
- Ensure the datasets are placed in the `datasets/` folder before running the notebooks.