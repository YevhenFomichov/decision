import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile

def predict_audio_class(audio_path: str, model_path: str = 'decision_tree_model.tflite'):
    """
    This function loads the TFLite model and makes a prediction using three features:
    (Spectral Centroid, Spectral Entropy, and Spectral Kurtosis) in the specified order.
    
    :param audio_path: Path to the audio file (e.g., wav, m4a, mp3, etc.)
    :param model_path: Path to the saved TFLite model (.tflite)
    :return: Predicted class (label)
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # 1. Spectral Centroid: Compute the average spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_spectral_centroid = np.mean(spectral_centroids)

    # 2. Spectral Entropy: Calculate using the power spectrum
    power_spectrum = np.abs(np.fft.fft(y))**2
    ps_norm = power_spectrum / np.sum(power_spectrum)
    spectral_entropy = -np.sum(ps_norm * np.log2(ps_norm + 1e-12))

    # 3. Spectral Kurtosis: Compute the kurtosis of the power spectrum
    spectral_kurtosis = pd.Series(power_spectrum).kurtosis()

    # Assemble features in the required order and convert them to float32 type
    features = np.array([
        avg_spectral_centroid,
        spectral_entropy,
        spectral_kurtosis
    ], dtype=np.float32).reshape(1, -1)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor with the prepared features
    interpreter.set_tensor(input_details[0]['index'], features)

    # Run inference
    interpreter.invoke()

    # Get the prediction from the output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0]

def main():
    st.title("Audio Classification App")
    st.write("Upload an audio file (wav, mp3, m4a, etc.) to get a prediction")

    audio_file = st.file_uploader("Select an audio file", type=['wav', 'mp3', 'm4a'])

    if audio_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name

        # Path to the TFLite model
        model_path = "decision_tree_model.tflite"

        # Get the numeric prediction result (e.g., 0 or 1)
        predicted_class = predict_audio_class(tmp_file_path, model_path)

        # Convert the numeric result to a human-readable label
        if predicted_class == 0:
            st.write("Predicted class: **Empty**")
        elif predicted_class == 1:
            st.write("Predicted class: **Full**")
        else:
            # In case the model returns a value different from 0 or 1
            st.write(f"Predicted class: **{predicted_class}**")

if __name__ == "__main__":
    main()
