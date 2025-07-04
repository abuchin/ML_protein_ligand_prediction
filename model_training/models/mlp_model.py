"""
Multi-Layer Perceptron (MLP) model for protein-ligand binding prediction using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Tuple
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    """Custom dataset for PyTorch."""
    
    def __init__(self, X, y):
        """
        Initialize the dataset.
        
        Args:
            X: Feature array
            y: Target array
        """
        self.X = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        self.y = torch.LongTensor(y.values if hasattr(y, 'values') else y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """Multi-Layer Perceptron neural network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the MLP.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output classes
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MLPModel(BaseModel):
    """MLP model implementation using PyTorch."""
    
    def __init__(self, input_size, hidden_size=256, output_size=2, batch_size=32, 
                 learning_rate=0.001, epochs=10, test_size=0.2, random_state=42):
        """
        Initialize the MLP model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output classes
            batch_size: Batch size for training
            learning_rate: Learning rate
            epochs: Number of training epochs
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        super().__init__("MLP", test_size, random_state)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model components
        self.model = MLP(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Data loaders
        self.train_loader = None
        self.test_loader = None
    
    def _initialize_model(self, **kwargs):
        """Not used for MLP as model is initialized in __init__."""
        pass
    
    def split_data(self, X, y):
        """
        Split data and create PyTorch data loaders.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create datasets and data loaders
        self.train_dataset = MyDataset(X_train, y_train)
        self.test_dataset = MyDataset(X_test, y_test)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Store for evaluation
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
    
    def _train_model(self):
        """Train the MLP model."""
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(self.train_loader):.4f}")
    
    def _predict_model(self) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Returns:
            Predictions array
        """
        self.model.eval()
        y_pred = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                y_pred.extend(predicted.cpu().numpy())
        
        return np.array(y_pred)
    
    def predict_proba(self) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Returns:
            Probability predictions array
        """
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path where to save the model
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"MLP model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"MLP model loaded from {path}")
    
    def get_model_info(self) -> dict:
        """
        Get extended model information including hyperparameters.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        base_info.update({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'device': str(self.device)
        })
        return base_info 