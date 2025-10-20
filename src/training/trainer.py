"""Model training module"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ModelTrainer:
    """Handles model training, saving, and inference"""
    
    def __init__(self, model, preprocessor, config):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.tokenizer = preprocessor.get_tokenizer()
        self.max_sequence_length = preprocessor.get_max_sequence_length()
    
    def train(self, X, y, model_save_path="model/chatbot_model.h5", history_file="model/training_history.txt", epochs=None):
        """Trains the model and saves it along with epoch history"""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        print(f"\n Training started for {epochs} epochs...\n")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=self.config.BATCH_SIZE,
            verbose=1
        )
        
        self.model.save(model_save_path)
        print(f"Model saved at {model_save_path}")
        
        # Save tokenizer
        tokenizer_path = "model/tokenizer.json"
        with open(tokenizer_path, "w", encoding="utf-8") as f:
            f.write(self.tokenizer.to_json())
        print(f"Tokenizer saved at {tokenizer_path}")

        with open(history_file, "w") as f:
            f.write("epoch,loss,accuracy\n")
            for i in range(len(history.history['loss'])):
                f.write(f"{i+1},{history.history['loss'][i]},{history.history['accuracy'][i]}\n")
        print(f"✅ Training history saved at {history_file}")

        return history
        
    def load_model(self, model_path="model/chatbot_model.h5", tokenizer_json="model/tokenizer.json"):
        """Loads model and tokenizer"""
        self.model = tf.keras.models.load_model(model_path)
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        import json
        with open(tokenizer_json) as f:
            self.tokenizer = tokenizer_from_json(json.load(f))
        print(f"✅ Model and tokenizer loaded successfully.")
    
    def generate_text(self, seed_text, num_words=10):
        """Generates text from a seed"""
        text = seed_text.strip()
        for _ in range(num_words):
            token_text = self.tokenizer.texts_to_sequences([text])[0]
            padded_token_text = pad_sequences(
                [token_text], 
                maxlen=self.max_sequence_length - 1, 
                padding=self.config.PADDING_TYPE
            )
            prediction = self.model.predict(padded_token_text, verbose=0)
            pos = np.argmax(prediction, axis=-1)[0]
            for word, index in self.tokenizer.word_index.items():
                if index == pos:
                    text += " " + word
                    break
        return text
