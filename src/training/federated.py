import logging
import time
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset  
from tqdm import tqdm
from typing import Dict, Tuple, Any
from src.models import ModelFactory
from src.evaluation import EvaluationMetrics
from src.base_trainer import PyTorchTrainer
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FederatedTrainer:
    """Enhanced federated learning with cross-validation and reduced overfitting."""
    
    def __init__(self, n_rounds: int = 8, local_epochs: int = 3, n_folds: int = 3):  # Reduced parameters
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.n_folds = n_folds
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        self.learning_curves = {}
        
    def federated_averaging_cv(self, client_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                               model_fn) -> Dict[str, Any]:
        """Perform federated averaging with cross-validation and progress tracking."""
        self.logger.info(f"Starting Federated Learning with {self.n_folds}-fold CV")
        self.logger.info(f"Federated parameters: rounds={self.n_rounds}, local_epochs={self.local_epochs}")
        self.logger.info(f"Client datasets: {list(client_datasets.keys())}")
        
        # Log client dataset statistics
        for client_id, (X, y) in client_datasets.items():
            unique_labels, counts = np.unique(y, return_counts=True)
            self.logger.info(f"Client {client_id}: samples={len(X)}, features={X.shape[1]}, "
                             f"classes={dict(zip(unique_labels, counts))}")
        
        # Combine all client data for CV splitting
        all_X = []
        all_y = []
        client_indices = []
        
        print("Preparing federated datasets...")
        for client_id, (X, y) in tqdm(client_datasets.items(), desc="Processing clients", unit="client"):
            all_X.append(X)
            all_y.append(y)
            client_indices.extend([client_id] * len(X))
        
        combined_X = np.vstack(all_X)
        combined_y = np.hstack(all_y)
        client_indices = np.array(client_indices)
        
        self.logger.info(f"Combined dataset: samples={len(combined_X)}, features={combined_X.shape[1]}")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_SEED)
        cv_results = []
        all_learning_curves = []
        
        # Progress bar for CV folds
        fold_progress = tqdm(
            enumerate(skf.split(combined_X, combined_y)), 
            total=self.n_folds,
            desc="Federated CV Folds",
            unit="fold"
        )
        
        for fold, (train_idx, val_idx) in fold_progress:
            fold_start_time = time.time()
            fold_progress.set_description(f"Federated CV Fold {fold + 1}/{self.n_folds}")
            
            self.logger.info(f"Starting federated training fold {fold + 1}/{self.n_folds}")
            
            # Split training data back into clients
            fold_client_data = {}
            for client_id in client_datasets.keys():
                client_mask = client_indices[train_idx] == client_id
                if np.any(client_mask):
                    client_train_idx = train_idx[client_mask]
                    fold_client_data[client_id] = (
                        combined_X[client_train_idx],
                        combined_y[client_train_idx]
                    )
                    self.logger.debug(f"Fold {fold + 1}, Client {client_id}: {len(client_train_idx)} training samples")
            
            # Validation data
            X_val = combined_X[val_idx]
            y_val = combined_y[val_idx]
            self.logger.debug(f"Fold {fold + 1}: {len(val_idx)} validation samples")
            
            # Train federated model for this fold
            global_model, fold_learning_curves = self._train_federated_fold(fold_client_data, model_fn, X_val.shape[1], fold + 1)
            all_learning_curves.append(fold_learning_curves)
            
            # Evaluate
            X_val_scaled = self.scaler.fit_transform(X_val)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_scaled),
                torch.FloatTensor(y_val)
            )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(global_model.parameters())
            trainer = PyTorchTrainer(global_model, criterion, optimizer)
            _, y_val, y_pred, y_pred_proba, _ = trainer.validate(val_loader)
            
            fold_metrics = EvaluationMetrics.calculate_metrics(y_val, y_pred, y_pred_proba)
            cv_results.append(fold_metrics)
            fold_time = time.time() - fold_start_time
            
            self.logger.info(f"Federated fold {fold + 1} completed in {fold_time:.2f}s - "
                             f"Accuracy: {fold_metrics['accuracy']:.4f}, "
                             f"F1: {fold_metrics['f1_score']:.4f}")
            
            # Update progress with current fold results
            fold_progress.set_postfix({
                'Acc': f"{fold_metrics['accuracy']:.3f}",
                'F1': f"{fold_metrics['f1_score']:.3f}"
            })
        
        fold_progress.close()
        
        # Store learning curves
        self.learning_curves['Federated_DNN'] = all_learning_curves
        
        aggregated_results = EvaluationMetrics.aggregate_cv_results(cv_results)
        
        self.logger.info("Federated learning cross-validation completed")
        self.logger.info(f"Average accuracy: {aggregated_results['accuracy']['mean']:.4f} ± {aggregated_results['accuracy']['std']:.4f}")
        self.logger.info(f"Average F1-score: {aggregated_results['f1_score']['mean']:.4f} ± {aggregated_results['f1_score']['std']:.4f}")
        
        return {
            'model_name': 'Federated_DNN',
            'cv_results': cv_results,
            'aggregated_metrics': aggregated_results
        }
    
    def _train_federated_fold(self, client_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                              model_fn, input_shape: int, fold_num: int) -> Tuple[nn.Module, Dict]:
        """Train federated model for a single fold with progress tracking."""
        self.logger.debug(f"Training federated model for fold {fold_num}")
        global_model = model_fn(input_shape)
        
        # Learning curve tracking
        fold_learning_curves = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': []
        }
        
        # Progress bar for federated rounds
        rounds_progress = tqdm(
            range(self.n_rounds), 
            desc=f"FL Rounds (Fold {fold_num})",
            unit="round",
            leave=False,
            position=1
        )
        
        for round_num in rounds_progress:
            round_start_time = time.time()
            self.logger.debug(f"Fold {fold_num}, Round {round_num + 1}/{self.n_rounds}")
            
            client_models = []
            round_train_losses = []
            round_train_accs = []
            
            # Progress for clients in this round
            client_progress = tqdm(
                client_data.items(),
                desc=f"Round {round_num + 1} Clients",
                unit="client",
                leave=False,
                position=2
            )
            
            for client_id, (X_client, y_client) in client_progress:
                client_progress.set_description(f"Training {client_id}")
                self.logger.debug(f"Training client {client_id} with {len(X_client)} samples")
                
                # Create client model and copy global weights
                client_model = model_fn(input_shape)
                client_model.load_state_dict(global_model.state_dict())
                
                # Scale client data
                X_client_scaled = self.scaler.fit_transform(X_client)
                
                # Create client data loader
                client_dataset = TensorDataset(
                    torch.FloatTensor(X_client_scaled),
                    torch.FloatTensor(y_client)
                )
                client_loader = torch.utils.data.DataLoader(
                    client_dataset, batch_size=32, shuffle=True
                )
                
                # Train locally with reduced learning rate
                criterion = nn.BCELoss()
                optimizer = optim.Adam(client_model.parameters(), lr=0.0005, weight_decay=1e-5)
                trainer = PyTorchTrainer(client_model, criterion, optimizer, patience=5)
                
                # Train for local epochs and collect metrics
                epoch_losses = []
                epoch_accs = []
                for local_epoch in range(self.local_epochs):
                    loss, acc = trainer.train_epoch(client_loader)
                    epoch_losses.append(loss)
                    epoch_accs.append(acc)
                
                round_train_losses.extend(epoch_losses)
                round_train_accs.extend(epoch_accs)
                client_models.append(client_model)
            
            client_progress.close()
            
            # Federated averaging
            if client_models:
                self.logger.debug(f"Performing federated averaging with {len(client_models)} client models")
                global_state_dict = global_model.state_dict()
                for key in global_state_dict.keys():
                    global_state_dict[key] = torch.mean(
                        torch.stack([client.state_dict()[key] for client in client_models]), 
                        dim=0
                    )
                global_model.load_state_dict(global_state_dict)
            
            # Store learning metrics
            if round_train_losses and round_train_accs:
                fold_learning_curves['train_losses'].append(np.mean(round_train_losses))
                fold_learning_curves['train_accuracies'].append(np.mean(round_train_accs))
                # For simplicity, use train metrics as val metrics in federated setting
                fold_learning_curves['val_losses'].append(np.mean(round_train_losses))
                fold_learning_curves['val_accuracies'].append(np.mean(round_train_accs))
            
            round_time = time.time() - round_start_time
            rounds_progress.set_postfix({'Round': f"{round_num + 1}/{self.n_rounds}", 'Time': f"{round_time:.1f}s"})
            self.logger.debug(f"Fold {fold_num}, Round {round_num + 1} completed in {round_time:.2f}s")
        
        rounds_progress.close()
        self.logger.debug(f"Federated training completed for fold {fold_num}")
        return global_model, fold_learning_curves