import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# ====================================
# Function to load the dataset
# ====================================
def load_handwritten_data(file_path):
    """
    Loads the dataset for handwritten character recognition.
    :param file_path: Path to the dataset file (CSV format).
    :return: Features and target arrays.
    """
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:].values / 255.0  # Normalize pixel values
    y = data.iloc[:, 0].values  # Target labels
    return X, y

# ====================================
# Function to preprocess the data
# ====================================
def preprocess_handwritten_data(X, y):
    """
    Preprocesses the handwritten data by reshaping and encoding labels.
    :param X: Features array.
    :param y: Target labels.
    :return: Preprocessed feature set and encoded labels.
    """
    X = X.reshape(-1, 28, 28, 1)  # Assuming 28x28 images
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    return X, y_encoded

# ====================================
# Function to build the model
# ====================================
def build_handwritten_model():
    """
    Builds a CNN model for handwritten character recognition.
    :return: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(26, activation='softmax')  # Assuming 26 alphabets (A-Z)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ====================================
# Function to train the model
# ====================================
def train_handwritten_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Trains the handwritten character recognition model.
    :param model: Compiled CNN model.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_val: Validation features.
    :param y_val: Validation labels.
    :param epochs: Number of epochs.
    :param batch_size: Batch size.
    :return: Trained model and training history.
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, history

# ====================================
# Function to evaluate the model
# ====================================
def evaluate_handwritten_model(model, X_test, y_test):
    """
    Evaluates the model on the test dataset.
    :param model: Trained CNN model.
    :param X_test: Test features.
    :param y_test: Test labels.
    :return: None
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Accuracy:", accuracy_score(y_true, y_pred_classes))
    print("Classification Report:\n", classification_report(y_true, y_pred_classes))

# ====================================
# Function to visualize training history
# ====================================
def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss curves.
    :param history: Training history object.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.show()

# ====================================
# Main function to execute the pipeline
# ====================================
def main():
    """
    Main function to execute the handwritten character recognition pipeline.
    """
    file_path = 'handwritten_data.csv'  # Replace with actual dataset path

    # Load the dataset
    X, y = load_handwritten_data(file_path)

    # Preprocess the data
    X, y = preprocess_handwritten_data(X, y)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build the model
    model = build_handwritten_model()

    # Train the model
    model, history = train_handwritten_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_handwritten_model(model, X_test, y_test)

    # Plot training history
    plot_training_history(history)

    # Save the model
    model.save('handwritten_character_recognition_model.h5')

# Execute the main function
if __name__ == "__main__":
    main()
