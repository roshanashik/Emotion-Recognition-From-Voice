import os
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import speech_recognition as sp
from pydub import AudioSegment
import tempfile
import io
import sounddevice as sd
import queue
import soundfile as sf
import vggish_input, vggish_params, vggish_slim
from tensorflow.keras.layers import Layer

class PositionalEncoding(Layer):
    def __init__(self, position, d_model, **kwargs):
        # Use kwargs to handle additional arguments like 'trainable'
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.positional_encoding = self.get_positional_encoding()

    def get_positional_encoding(self):
        angle_rads = self._get_angles(
            np.arange(self.position)[:, np.newaxis],
            np.arange(self.d_model)[np.newaxis, :],
            self.d_model,
        )
        # Apply sin to even indices and cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

    # Add this method to allow proper serialization and deserialization
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config

# Real-time audio settings
audio_queue = queue.Queue()
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# Updated Feature Extraction
def extract_features(audio, sample_rate=16000, vggish_model=None):  # Changed 'sr' to 'sample_rate'
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    mfcc_features = np.vstack((mfccs, delta_mfccs, delta2_mfccs)).T

    min_length = 100
    if mfcc_features.shape[0] < min_length:
        mfcc_features = np.pad(mfcc_features, ((0, min_length - mfcc_features.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc_features = mfcc_features[:min_length, :]

    vggish_features = extract_vggish_features(audio, sample_rate, vggish_model)  # Updated 'sr' to 'sample_rate'
    if vggish_features is None:
        return None

    combined_features = np.hstack((mfcc_features, vggish_features))
    return combined_features

def extract_vggish_features(audio, sample_rate, vggish_model):  # Updated 'sr' to 'sample_rate'
    try:
        infer = vggish_model.signatures['serving_default']
        examples = vggish_input.waveform_to_examples(audio, sample_rate)  # Updated 'sr' to 'sample_rate'
        embedding = infer(input_tensor=tf.convert_to_tensor(examples, dtype=tf.float32))['embedding']
        embedding_resized = np.resize(embedding.numpy(), (100, embedding.shape[1]))
        return embedding_resized
    except Exception as e:
        print(f"Error extracting VGGish embeddings: {e}")
        return None

# Emotion Prediction
def predict_emotion(audio, sample_rate, vggish_model, emotion_model):
    features = extract_features(audio, sample_rate, vggish_model)  # Fixed the issue with sr
    if features is None:
        return "Error: Could not extract features"
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    predictions = emotion_model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)[0]
    emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    return emotion_labels[predicted_class]

# Transcription
def transcribe_audio_advanced(file_path):
    recognizer = sp.Recognizer()
    with sp.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sp.UnknownValueError:
        return "Could not understand the audio."
    except sp.RequestError as e:
        return f"Request Error: {e}"

# Save uploaded file locally
def save_file_locally(uploaded_file):
    temp_file_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_file_path

# Convert MP3 to WAV
def convert_to_wav(mp3_path):
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

# Real-time emotion prediction
# Real-time emotion prediction and transcription with audio playback
def predict_emotion_real_time(vggish_model, emotion_model):
    st.write("### 🎤 Recording... Speak Now")
    duration = 5  # Record for 5 seconds
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        file_name = temp_audio.name
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
            audio_data = []
            for _ in range(int(16000 * duration / 512)):
                audio_data.append(audio_queue.get())
        audio_data = np.concatenate(audio_data, axis=0).flatten()

        # Save the recorded audio to a temporary file for playback
        sf.write(file_name, audio_data, 16000)

        # Play the recorded audio
        st.audio(file_name, format="audio/wav")

        # Transcribe the audio in real-time
        transcription = transcribe_audio_advanced(file_name)
        st.write("### 📝 Transcribed Text")
        st.markdown(f"<h4 style='font-size: 20px;'>{transcription}</h4>", unsafe_allow_html=True)

        # Process the audio for emotion prediction
        emotion = predict_emotion(audio_data, 16000, vggish_model, emotion_model)
        return emotion
# Streamlit UI
st.set_page_config(page_title="Emotion Recognition & Speech-to-Text", layout="centered")
st.title("🎙️ Emotion Recognition and Speech-to-Text")
st.markdown("Upload an audio file (MP3/WAV) or choose real-time prediction to detect its emotion and transcribe content.")

vggish_model_path = r'C:\Users\USER\Documents\Emotion Recognition from Voice\Pre-Trained Models\vggish_saved_model'
vggish_model = tf.saved_model.load(vggish_model_path)
emotion_model = tf.keras.models.load_model(r"C:\Users\USER\Documents\Emotion Recognition from Voice\emotion-recognition-model-2-other-default-v1\best_emotion_model_87.h5",custom_objects={'PositionalEncoding': PositionalEncoding})


# File-based prediction
uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
if uploaded_file:
    # Save uploaded file
    file_path = save_file_locally(uploaded_file)

    # Convert MP3 to WAV if necessary
    if uploaded_file.name.endswith(".mp3"):
        file_path = convert_to_wav(file_path)

    # Play the uploaded file
    st.audio(file_path, format="audio/wav")

    # Emotion Prediction
    st.write("### 🧠 Detected Emotion")
    audio, sr = librosa.load(file_path, sr=16000)
    emotion = predict_emotion(audio, sr, vggish_model, emotion_model)
    st.markdown(f"<h2 style='color: #2e8b57;'>{emotion}</h2>", unsafe_allow_html=True)

    # Speech-to-Text
    st.write("### 📝 Transcribed Text")
    transcription = transcribe_audio_advanced(file_path)
    st.markdown(f"<h4 style='font-size: 20px;'>{transcription}</h4>", unsafe_allow_html=True)

# Real-time prediction
if st.button("Real-Time Emotion Prediction"):
    emotion = predict_emotion_real_time(vggish_model, emotion_model)
    st.write("### 🧠 Detected Emotion (Real-Time)")
    st.markdown(f"<h2 style='color: #2e8b57;'>{emotion}</h2>", unsafe_allow_html=True)

# Cleanup temporary files
if os.path.exists("temp_audio"):
    for file in os.listdir("temp_audio"):
        os.remove(os.path.join("temp_audio", file))


