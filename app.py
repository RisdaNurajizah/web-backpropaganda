import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. Pengolahan data dan normalisasi
# Load dataset
data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 2. Pembuatan model neural network dengan TensorFlow/Keras
model = Sequential([
    Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Training dan evaluasi model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=0
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi: {accuracy:.4f}, Loss: {loss:.4f}")

# 5. Visualisasi training loss dan akurasi
def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # Loss
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].legend()
    # Akurasi
    ax[1].plot(history.history['accuracy'], label='Train Acc')
    ax[1].plot(history.history['val_accuracy'], label='Val Acc')
    ax[1].set_title('Akurasi')
    ax[1].legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# 4. Membuat aplikasi web dengan Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Ambil data input dari form
            input_data = [
                float(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                float(request.form['Age'])
            ]
            # Normalisasi input
            input_scaled = scaler.transform([input_data])
            # Prediksi
            pred = model.predict(input_scaled)
            prediction = 'Positif Diabetes' if pred[0][0] >= 0.5 else 'Negatif Diabetes'
        except Exception as e:
            prediction = f"Error: {e}"
    # Visualisasi training
    plot_url = plot_history(history)
    return render_template('index.html', prediction=prediction, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)