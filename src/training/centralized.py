import logging
import time 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset
from tqdm import tqdm
from typing import Any, Dict
from base_trainer import PyTorchTrainer
from evaluation import EvaluationMetrics
from models import ModelFactory
RANDOM_SEED = 42
class CentralizedTrainer:
    """Enhanced centralized model training with cross-validation and learning curves."""
    
    def __init__(self, n_folds: int = 3, random_state: int = RANDOM_SEED):
        self.n_folds = n_folds
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        self.learning_curves = {}
        
    def train_model_cv(self, model_fn, X: np.ndarray, y: np.ndarray, 
                       model_name: str, **model_kwargs) -> Dict[str, Any]:
        """Train model with cross-validation and progress tracking."""
        self.logger.info(f"Starting {self.n_folds}-fold cross-validation for {model_name}")
        self.logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"Model parameters: {model_kwargs}")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_results = []
        fold_times = []
        all_learning_curves = []
        
        # Progress bar for CV folds
        fold_progress = tqdm(
            enumerate(skf.split(X, y)), 
            total=self.n_folds,
            desc=f"CV Folds for {model_name}",
            unit="fold",
            leave=False
        )
        
        for fold, (train_idx, val_idx) in fold_progress:
            fold_start_time = time.time()
            fold_progress.set_description(f"CV Fold {fold + 1}/{self.n_folds} - {model_name}")
            
            self.logger.debug(f"Training fold {fold + 1}/{self.n_folds} for {model_name}")
            self.logger.debug(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_fold)
            X_val_scaled = self.scaler.transform(X_val_fold)
            
            # Handle LSTM data reshaping
            if 'lstm' in model_name.lower():
                X_train_scaled = ModelFactory.prepare_lstm_data(X_train_scaled)
                X_val_scaled = ModelFactory.prepare_lstm_data(X_val_scaled)
            
            # Create and train model
            if isinstance(model_fn, type) and issubclass(model_fn, (DecisionTreeClassifier, RandomForestClassifier, LogisticRegression)):
                # Sklearn model
                self.logger.debug(f"Training sklearn model: {model_fn.__name__}")
                # Reduced parameters for sklearn models to prevent overfitting
                if model_fn == DecisionTreeClassifier:
                    model_kwargs.update({'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 10})
                elif model_fn == RandomForestClassifier:
                    model_kwargs.update({'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 10})
                
                model = model_fn(**model_kwargs)
                model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train_fold)
                y_pred = model.predict(X_val_scaled.reshape(X_val_scaled.shape[0], -1))
                y_pred_proba = model.predict_proba(X_val_scaled.reshape(X_val_scaled.shape[0], -1))[:, 1] if hasattr(model, 'predict_proba') else None
            else:
                # PyTorch model
                self.logger.debug(f"Training PyTorch model: {model_name}")
                input_size = X_train_scaled.shape[-1]
                model = model_fn(input_size, **model_kwargs)
                
                # Create data loaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train_scaled),
                    torch.FloatTensor(y_train_fold)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_scaled),
                    torch.FloatTensor(y_val_fold)
                )
                
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=64, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=64, shuffle=False
                )
                
                # Training with reduced learning rate to prevent overfitting
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Reduced LR, added weight decay
                trainer = PyTorchTrainer(model, criterion, optimizer, patience=15)
                
                trainer.fit(train_loader, val_loader, epochs=30, fold_num=fold + 1, total_folds=self.n_folds)
                
                # Store learning curves
                if 'DNN' in model_name or 'LSTM' in model_name:
                    all_learning_curves.append({
                        'train_losses': trainer.train_losses.copy(),
                        'val_losses': trainer.val_losses.copy(),
                        'train_accuracies': trainer.train_accuracies.copy(),
                        'val_accuracies': trainer.val_accuracies.copy()
                    })
                
                # Validation
                _, y_val_fold, y_pred, y_pred_proba, _ = trainer.validate(val_loader)
            
            # Calculate metrics
            fold_metrics = EvaluationMetrics.calculate_metrics(y_val_fold, y_pred, y_pred_proba)
            cv_results.append(fold_metrics)
            fold_time = time.time() - fold_start_time
            fold_times.append(fold_time)
            
            self.logger.info(f"Fold {fold + 1} completed in {fold_time:.2f}s - "
                             f"Accuracy: {fold_metrics['accuracy']:.4f}, "
                             f"F1: {fold_metrics['f1_score']:.4f}")
            
            # Update progress with current fold results
            fold_progress.set_postfix({
                'Acc': f"{fold_metrics['accuracy']:.3f}",
                'F1': f"{fold_metrics['f1_score']:.3f}",
                'Time': f"{fold_time:.1f}s"
            })
        
        fold_progress.close()
        
        # Store learning curves for deep learning models
        if all_learning_curves:
            self.learning_curves[model_name] = all_learning_curves
        
        # Aggregate results
        aggregated_results = EvaluationMetrics.aggregate_cv_results(cv_results)
        aggregated_results['training_time'] = {
            'mean': np.mean(fold_times),
            'std': np.std(fold_times),
            'total': np.sum(fold_times)
        }
        
        self.logger.info(f"Cross-validation completed for {model_name}")
        self.logger.info(f"Average accuracy: {aggregated_results['accuracy']['mean']:.4f} ± {aggregated_results['accuracy']['std']:.4f}")
        self.logger.info(f"Average F1-score: {aggregated_results['f1_score']['mean']:.4f} ± {aggregated_results['f1_score']['std']:.4f}")
        self.logger.info(f"Total training time: {aggregated_results['training_time']['total']:.2f}s")
        
        return {
            'model_name': model_name,
            'cv_results': cv_results,
            'aggregated_metrics': aggregated_results
        }