import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ========================================
# Function to load medical dataset
# ========================================
def load_medical_data(file_path):
    """
    Loads the medical dataset from the specified file path.
    :param file_path: Path to the dataset file (CSV format).
    :return: Pandas DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)

# ========================================
# Function to perform Exploratory Data Analysis (EDA)
# ========================================
def perform_medical_eda(data):
    """
    Performs EDA on the medical dataset.
    :param data: DataFrame containing the dataset.
    """
    print(data.info())
    print(data.describe())
    print(data.isnull().sum())
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()

# ========================================
# Function to preprocess the medical data
# ========================================
def preprocess_medical_data(data, target_column):
    """
    Handles missing values, encodes categorical variables, and scales numerical features.
    :param data: DataFrame containing the dataset.
    :param target_column: Target column for prediction.
    :return: Preprocessed DataFrame.
    """
    data.fillna(data.mean(), inplace=True)
    categorical_features = data.select_dtypes(include=['object']).columns
    for col in categorical_features:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data.drop(target_column, axis=1))
    data_scaled = pd.DataFrame(scaled_features, columns=data.columns[:-1])
    data_scaled[target_column] = data[target_column]
    return data_scaled

# ========================================
# Function to split the data
# ========================================
def split_medical_data(data, target_column, test_size=0.2):
    """
    Splits the dataset into training and testing sets.
    :param data: Preprocessed DataFrame.
    :param target_column: Target column for prediction.
    :param test_size: Proportion of test data.
    :return: Training and testing sets for features and target.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)

# ========================================
# Function to train the disease prediction model
# ========================================
def train_disease_model(X_train, y_train):
    """
    Trains a Random Forest model using GridSearchCV.
    :param X_train: Training features.
    :param y_train: Training target.
    :return: Trained model and best parameters.
    """
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# ========================================
# Function to evaluate the disease prediction model
# ========================================
def evaluate_disease_model(model, X_test, y_test):
    """
    Evaluates the trained model using test data.
    :param model: Trained model.
    :param X_test: Test features.
    :param y_test: Test target.
    """
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))
    print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print("\nClassification Report:\n", classification_report(y_test, predictions))

# ========================================
# Function to save the trained model
# ========================================
def save_disease_model(model, file_name):
    """
    Saves the trained disease prediction model to a file.
    :param model: Trained model.
    :param file_name: File name for saving the model.
    """
    joblib.dump(model, file_name)

# ========================================
# Main function to execute the disease prediction pipeline
# ========================================
def main():
    """
    Executes the disease prediction pipeline.
    """
    file_path = 'medical_data.csv'  # Replace with actual dataset path
    target_column = 'disease'      # Replace with actual target column

    # Load the dataset
    data = load_medical_data(file_path)

    # Perform EDA
    perform_medical_eda(data)

    # Preprocess the data
    data_scaled = preprocess_medical_data(data, target_column)

    # Split the data
    X_train, X_test, y_train, y_test = split_medical_data(data_scaled, target_column)

    # Train the model
    best_model, best_params = train_disease_model(X_train, y_train)
    print("Best Parameters:", best_params)

    # Evaluate the model
    evaluate_disease_model(best_model, X_test, y_test)

    # Save the model
    save_disease_model(best_model, 'disease_prediction_model.pkl')

# Execute the main function
if __name__ == "__main__":
    main()