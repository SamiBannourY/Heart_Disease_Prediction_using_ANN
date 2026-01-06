Heart Disease Prediction using ANN

🧠 Overview

Heart disease is a major global health issue. Early detection can significantly improve patient outcomes. In this project, we apply a neural network classifier to a heart disease dataset to automate diagnostic predictions. 


Goals:

Learn a deep learning pipeline from preprocessing to evaluation

Build a simple ANN model using Keras

Evaluate model performance using accuracy and confusion matrix

📦 Dataset

The dataset contains 13 medical features and one binary target (0: no heart disease, 1: heart disease).
Typical features include:

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol level

Fasting blood sugar
… and more. 


You can replace the placeholder dataset with the UCI Heart Disease dataset or any formatted CSV with similar attributes.

🛠 Built With

Python

TensorFlow

Keras

scikit-learn

pandas & numpy

matplotlib

🚀 How It Works
1. Load Libraries

Import deep learning, data processing, and evaluation libraries:

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

2. Load Dataset
data = pd.read_csv("Dataset--Heart-Disease-Prediction-using-ANN.csv")

3. Preprocess Data

Separate features (X) and label (y)

Scale feature values

Train/test split

4. Build ANN
model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=13))
model.add(Dense(units=14, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

5. Train Model
model.fit(X_train, y_train, batch_size=8, epochs=100)

6. Evaluate

Predict on test set

Compute confusion matrix

Calculate accuracy

📊 Results

This model typically achieves around ~88% accuracy on the test dataset. 


You can further evaluate performance via:

Precision, Recall, F1 Score

ROC/AUC

Training vs validation curves

🔍 Next Improvements

Consider extending this project by:

Hyperparameter tuning

Adding cross-validation

Improving preprocessing (feature engineering)

Adding UI (Streamlit/Flask)

Exporting model (TensorFlow SavedModel / ONNX)
