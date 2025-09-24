import logging
import time
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm   

class PyTorchTrainer:
    """PyTorch model trainer with early stopping, learning curves, and progress tracking."""
    
    def __init__(self, model, criterion, optimizer, device=DEVICE, patience=15):  # Increased patience
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.logger = logging.getLogger(__name__)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, epoch_progress=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (output.squeeze() > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if epoch_progress:
                epoch_progress.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100 * correct / total:.2f}%"
                })
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                probabilities = output.squeeze().cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        return avg_loss, np.array(all_targets), np.array(all_predictions), np.array(all_probabilities), accuracy
    
    def fit(self, train_loader, val_loader, epochs=30, fold_num=1, total_folds=3):  # Reduced epochs
        """Train the model with early stopping and learning curve tracking."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Reset learning curves
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.logger.debug(f"Starting training for fold {fold_num}/{total_folds}, epochs={epochs}")
        
        # Progress bar for epochs
        epoch_progress = tqdm(
            range(epochs),
            desc=f"Fold {fold_num}/{total_folds} Training",
            unit="epoch",
            leave=False,
            position=1
        )
        
        for epoch in epoch_progress:
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, _, _, _, val_acc = self.validate(val_loader)
            
            # Store metrics for learning curves
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Update progress
            epoch_progress.set_postfix({
                'Train Loss': f"{train_loss:.4f}",
                'Val Loss': f"{val_loss:.4f}",
                'Train Acc': f"{train_acc:.3f}",
                'Val Acc': f"{val_acc:.3f}"
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                self.logger.debug(f"Early stopping triggered at epoch {epoch + 1}")
                epoch_progress.set_description(f"Early stopping at epoch {epoch + 1}")
                break
        
        epoch_progress.close()
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.logger.debug(f"Training completed for fold {fold_num}, best validation loss: {best_val_loss:.4f}")