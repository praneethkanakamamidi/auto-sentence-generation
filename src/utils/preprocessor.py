"""Data preprocessing utilities"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class DataPreprocessor:
    """Handles data preprocessing for the chatbot model"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = Tokenizer()
        self.max_sequence_length = None
        
    def prepare_training_data(self, text_data):
        """Prepares training data from text"""
        if isinstance(text_data, list):
            text_data = " ".join([line.strip() for line in text_data if line.strip()])
        self.tokenizer.fit_on_texts([text_data])
        input_sequences = self._create_input_sequences(text_data)
        padded_sequences = self._pad_sequences(input_sequences)
        X, y = self._split_features_labels(padded_sequences)
        y_categorical = to_categorical(y, num_classes=self.get_vocab_size())
        return X, y_categorical
    
    def _create_input_sequences(self, text_data):
        """Creates input sequences from text data"""
        input_sequences = []
        
        for sentence in text_data.split('\n'):
            tokenized_sentence = self.tokenizer.texts_to_sequences([sentence])[0]
            
            for i in range(1, len(tokenized_sentence)):
                input_sequences.append(tokenized_sentence[:i+1])
        
        return input_sequences
    
    def _pad_sequences(self, input_sequences):
        """Pads sequences to uniform length"""
        if not input_sequences:
            raise ValueError("No input sequences found")
            
        self.max_sequence_length = max([len(x) for x in input_sequences])
        
        return pad_sequences(
            input_sequences, 
            maxlen=self.max_sequence_length, 
            padding=self.config.PADDING_TYPE
        )
    
    def _split_features_labels(self, padded_sequences):
        """Splits sequences into features and labels"""
        X = padded_sequences[:, :-1]
        y = padded_sequences[:, -1]
        return X, y
    
    def get_vocab_size(self):
        """Returns the vocabulary size"""
        return len(self.tokenizer.word_index) + 1
    
    def get_tokenizer(self):
        """Returns the tokenizer"""
        return self.tokenizer
    
    def get_max_sequence_length(self):
        """Returns the maximum sequence length"""
        return self.max_sequence_length