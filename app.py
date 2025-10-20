import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from config.model_config import ModelConfig
from src.utils.preprocessor import DataPreprocessor
from src.models.chatmodel import ChatbotModel
from src.training.trainer import ModelTrainer

st.title("Chatbot Text Generator")

MODEL_PATH = "model/chatbot_model.h5"
TOKENIZER_PATH = "model/tokenizer.json"

config = ModelConfig()

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH) as f:
        tokenizer = tokenizer_from_json(f.read())
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

st.success("Model and tokenizer loaded successfully!")

# Input section
seed_text = st.text_input("Enter a seed phrase", "India is a")
num_words = st.slider("Number of words to generate", 5, 50, 10)

if st.button("Generate Text"):
    # Prepare the text
    token_text = tokenizer.texts_to_sequences([seed_text])[0]
    max_seq_len = config.MAX_SEQUENCE_LENGTH - 1
    text = seed_text

    for _ in range(num_words):
        padded = pad_sequences([token_text], maxlen=max_seq_len, padding=config.PADDING_TYPE)
        pred = model.predict(padded, verbose=0)
        pos = np.argmax(pred, axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == pos:
                text += " " + word
                token_text.append(pos)
                break

    st.write("### Generated Text:")
    st.success(text)
