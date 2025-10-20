"""Chatbot model definition"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


class ChatbotModel:
    """Chatbot model class"""
    
    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size
        self.model = self._build_model()
    
    def _build_model(self):
        """Builds and returns the model architecture"""
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.config.EMBEDDING_DIM,
                input_length=self.config.MAX_SEQUENCE_LENGTH - 1
            ),
            LSTM(self.config.LSTM_UNITS, return_sequences=True),
            Dropout(self.config.DROPOUT_RATE),
            LSTM(self.config.LSTM_UNITS),
            Dropout(self.config.DROPOUT_RATE),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        return model
    
    def compile_model(self):
        """Compiles the model"""
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Returns the Keras model"""
        return self.model
    
    def summary(self):
        """Prints model summary"""
        return self.model.summary()