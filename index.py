import streamlit as st
import librosa
import numpy as np
import pandas as pd
from joblib import load
import tempfile

def predict_audio_class(audio_path: str, model_path: str = 'decision_tree_model.joblib'):
    """
    This function loads the model and makes a prediction using three features
    (Spectral Centroid, Spectral Entropy, and Spectral Kurtosis) in the specified order.

    :param audio_path: Path to the audio file (e.g., wav, m4a, etc.)
    :param model_path: Path to the saved model (.joblib),
                       which was trained specifically on these three features.
    :return: Predicted class (label).
    """

    # Load the audio
    y, sr = librosa.load(audio_path, sr=None)

    # 1. Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_spectral_centroid = np.mean(spectral_centroids)

    # 2. Spectral Entropy
    power_spectrum = np.abs(np.fft.fft(y))**2
    ps_norm = power_spectrum / np.sum(power_spectrum)
    spectral_entropy = -np.sum(ps_norm * np.log2(ps_norm + 1e-12))

    # 3. Spectral Kurtosis
    spectral_kurtosis = pd.Series(power_spectrum).kurtosis()

    # Собираем признаки в нужном порядке
    features = np.array([
        avg_spectral_centroid,
        spectral_entropy,
        spectral_kurtosis
    ]).reshape(1, -1)

    # Загрузка обученной модели
    model = load(model_path)

    # Предсказание
    prediction = model.predict(features)
    return prediction[0]


def main():
    st.title("Audio Classification App")
    st.write("Upload an audio file (wav, mp3, m4a, etc.) to get a prediction")

    audio_file = st.file_uploader("Select an audio file", type=['wav', 'mp3', 'm4a'])

    if audio_file is not None:
        # Сохраняем загруженный файл во временный
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name

        # Путь к модели
        model_path = "decision_tree_model.joblib"

        # Получаем числовой результат предсказания (0 или 1)
        predicted_class = predict_audio_class(tmp_file_path, model_path)

        # Превращаем числовой результат в текст
        if predicted_class == 0:
            st.write("Predicted class: **Empty**")
        elif predicted_class == 1:
            st.write("Predicted class: **Full**")
        else:
            # На случай, если модель будет возвращать что-то отличное от 0 или 1
            st.write(f"Predicted class: **{predicted_class}**")

if __name__ == "__main__":
    main()
