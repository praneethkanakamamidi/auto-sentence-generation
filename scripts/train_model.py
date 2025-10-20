"""Main training script"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.model_config import ModelConfig
from src.utils.preprocessor import DataPreprocessor
from src.models.chatmodel import ChatbotModel
from src.training.trainer import ModelTrainer


def main():
    """Main training function"""
    
    config = ModelConfig()
    
    print("Loading FAQ data...")
    with open('data/independence_day.txt') as f:
        lines = f.readlines()

    
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(config)
    X, y = preprocessor.prepare_training_data(lines)
    
    print(f"Training data shape: {X.shape}")
    print(f"Vocabulary size: {preprocessor.get_vocab_size()}")
    
    print("Building model...")
    chatbot_model = ChatbotModel(config, preprocessor.get_vocab_size())
    chatbot_model.compile_model()
    
    print("Model summary:")
    chatbot_model.summary()
    
    print("Starting training...")
    trainer = ModelTrainer(
        chatbot_model.get_model(), 
        preprocessor, 
        config
    )
    
    trainer.train(X, y)
    
    print("Training completed! Starting interactive mode...")
    trainer.generate_text()


if __name__ == "__main__":
    main()