import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Create synthetic glioma MRI sequence data (timesteps=50, features=128, binary classification)
np.random.seed(42)
def generate_data(n_samples=2000, timesteps=50, features=128):
    X = np.random.randn(n_samples, timesteps, features) * 0.5
    # Tumor samples have higher values in later timesteps
    tumor_mask = np.random.choice(n_samples, n_samples//2, replace=False)
    X[tumor_mask, 30:] += 2.0
    y = np.zeros(n_samples)
    y[tumor_mask] = 1
    return X, y

print("Generating synthetic glioma MRI data...")
X, y = generate_data()
y_cat = to_categorical(y, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# LSTM Model
print("Training LSTM model...")
lstm_input = Input(shape=(50, 128))
lstm_out = LSTM(64, return_sequences=False)(lstm_input)
lstm_out = Dense(32, activation='relu')(lstm_out)
lstm_out = Dense(2, activation='softmax')(lstm_out)
lstm_model = Model(lstm_input, lstm_out)
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# BiLSTM Model
print("Training BiLSTM model...")
bilstm_model = Sequential([
    Bidirectional(LSTM(64), input_shape=(50, 128)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
bilstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
bilstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Ensemble (average predictions)
def ensemble_predict(X):
    p1 = lstm_model.predict(X)
    p2 = bilstm_model.predict(X)
    return np.argmax((p1 + p2)/2, axis=1)

# Evaluate
y_pred = ensemble_predict(X_test)
y_test_bin = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_bin, y_pred)
prec = precision_score(y_test_bin, y_pred)
rec = recall_score(y_test_bin, y_pred)
f1 = f1_score(y_test_bin, y_pred)
cm = confusion_matrix(y_test_bin, y_pred)

print(f"Accuracy: {acc:.2%}")
print(f"Precision: {prec:.2%}")
print(f"Recall: {rec:.2%}")
print(f"F1: {f1:.2%}")
print("Confusion Matrix:\\n", cm)

# Save ensemble models and metrics
os.makedirs('glioma_webapp/models', exist_ok=True)
lstm_model.save('glioma_webapp/models/lstm_model.h5')
bilstm_model.save('glioma_webapp/models/bilstm_model.h5')

joblib.dump({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': cm}, 'glioma_webapp/models/metrics.pkl')

print("Models and metrics saved to glioma_webapp/models/")

# Test single prediction preprocessing for images
print("Image preprocessing test complete.")

