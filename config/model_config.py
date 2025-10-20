"""Model configuration parameters"""

class ModelConfig:
    """Configuration class for model parameters"""
    
    # Model architecture
    EMBEDDING_DIM = 100
    LSTM_UNITS = 150
    DROPOUT_RATE = 0.2
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Data processing
    MAX_SEQUENCE_LENGTH = 56
    PADDING_TYPE = 'pre'
    TRUNCATING_TYPE = 'pre'
    
    # Inference
    PREDICTION_LENGTH = 10
    DELAY_SECONDS = 2