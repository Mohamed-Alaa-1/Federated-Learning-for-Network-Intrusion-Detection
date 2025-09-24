import numpy as np
import torch.nn as nn
class DNNModel(nn.Module):
    """Enhanced DNN model with reduced overfitting parameters."""
    
    def __init__(self, input_size: int, dropout_rate: float = 0.2):  # Reduced dropout
        super(DNNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),  # Reduced from 128
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),  # Reduced from 64
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),  # Same
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),  # Even less dropout
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

class LSTMModel(nn.Module):
    """Enhanced LSTM model with reduced overfitting parameters."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.2):  # Reduced complexity
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 32),  # Reduced from 64
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 8),  # Reduced from 16
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use the last output
        output = self.fc_layers(lstm_out[:, -1, :])
        return output
    
class ModelFactory:
    """Enhanced model creation with reduced overfitting parameters."""
    
    @staticmethod
    def create_dnn_model(input_shape: int, dropout_rate: float = 0.2) -> nn.Module:
        """Create a PyTorch DNN model with reduced overfitting."""
        return DNNModel(input_shape, dropout_rate)
    
    @staticmethod
    def create_lstm_model(input_shape: int, n_timesteps: int = 1, dropout_rate: float = 0.2) -> nn.Module:
        """Create a PyTorch LSTM model with reduced overfitting."""
        return LSTMModel(input_shape, dropout_rate=dropout_rate)
    
    @staticmethod
    def prepare_lstm_data(X: np.ndarray, n_timesteps: int = 1) -> np.ndarray:
        """Reshape data for LSTM input."""
        return X.reshape((X.shape[0], n_timesteps, X.shape[1]))