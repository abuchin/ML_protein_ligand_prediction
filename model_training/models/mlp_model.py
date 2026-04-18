"""
Multi-Layer Perceptron (MLP) model for protein-ligand binding prediction using PyTorch.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class _BindingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
        self.y = torch.LongTensor(y if isinstance(y, np.ndarray) else y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.net(x)


class MLPModel(BaseModel):
    """MLP model implementation using PyTorch with early stopping."""

    def __init__(self, input_size, hidden_size=256, output_size=2, batch_size=32,
                 learning_rate=0.001, epochs=50, patience=5,
                 test_size=0.2, random_state=42):
        super().__init__("MLP", test_size, random_state)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MLP(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_loader = None
        self.test_loader = None
        self._val_loader = None

    def _initialize_model(self, **kwargs):
        pass  # initialised in __init__

    def split_data(self, X, y) -> None:
        """Split into train/val/test, apply StandardScaler fitted on train only."""
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        # 80 % train+val / 20 % test
        X_tv, X_test, y_tv, y_test = train_test_split(
            X_arr, y_arr, stratify=y_arr,
            test_size=self.test_size, random_state=self.random_state
        )
        # 10 % of train+val → validation (for early stopping)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, stratify=y_tv,
            test_size=0.1, random_state=self.random_state
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.train_loader = DataLoader(
            _BindingDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        self._val_loader = DataLoader(
            _BindingDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            _BindingDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False
        )

        logger.info(
            f"MLP data split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test — "
            f"StandardScaler applied"
        )

    def _eval_loss(self, loader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                total += self.loss_fn(self.model(data), target).item()
        return total / len(loader)

    def _train_model(self) -> None:
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(data), target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss = self._eval_loss(self._val_loader)
            logger.info(
                f"Epoch {epoch+1}/{self.epochs}  "
                f"train_loss={train_loss/len(self.train_loader):.4f}  "
                f"val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience={self.patience})")
                    break

        self.model.load_state_dict(best_state)
        logger.info(f"Restored best model weights (val_loss={best_val_loss:.4f})")

    def _predict_model(self) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                _, predicted = torch.max(self.model(data), 1)
                preds.extend(predicted.cpu().numpy())
        return np.array(preds)

    def predict_proba(self) -> np.ndarray:
        self.model.eval()
        probs = []
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                probs.extend(torch.softmax(self.model(data), dim=1).cpu().numpy())
        return np.array(probs)

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        logger.info(f"MLP model saved to {path}")

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"MLP model loaded from {path}")

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'patience': self.patience,
            'device': str(self.device),
        })
        return info
