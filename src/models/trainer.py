"""
Model training utilities with early stopping, learning rate scheduling,
and checkpoint management.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
from pathlib import Path
import json
import time
from tqdm import tqdm

from ..config import config


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if improved, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return True
        
        if self.mode == 'min':
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta
        
        if improved:
            self.best_score = metric
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class LabelSmoothingLoss(nn.Module):
    """Binary cross-entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothed loss.
        
        Args:
            pred: Predictions (logits or probabilities)
            target: Ground truth labels
        """
        # Apply label smoothing
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # If pred is logits, apply sigmoid
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Binary cross-entropy
        loss = -target_smooth * torch.log(pred + 1e-9) - (1 - target_smooth) * torch.log(1 - pred + 1e-9)
        return loss.mean()


class ModelTrainer:
    """
    Training manager for the adversarial detection model.
    
    Features:
    - AdamW optimizer with weight decay
    - Cosine annealing learning rate schedule
    - Early stopping
    - Label smoothing
    - Class weighting for imbalanced data
    - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
    
    def setup_training(
        self,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 10000,
        label_smoothing: float = 0.1,
        class_weights: Optional[List[float]] = None,
        early_stopping_patience: int = 5
    ):
        """Configure training components."""
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        def warmup_cosine(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
        
        self.scheduler = LambdaLR(self.optimizer, warmup_cosine)
        
        # Loss function
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(label_smoothing)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Class weights
        self.class_weights = class_weights
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits'].squeeze(-1)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Apply class weights if specified
            if self.class_weights is not None:
                weights = torch.where(
                    labels == 1,
                    torch.tensor(self.class_weights[1], device=self.device),
                    torch.tensor(self.class_weights[0], device=self.device)
                )
                loss = (loss * weights).mean() if isinstance(loss, torch.Tensor) and loss.dim() > 0 else loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            total_loss += loss.item() * input_ids.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.size(0)
            
            # Confusion matrix components
            total_tp += ((preds == 1) & (labels == 1)).sum().item()
            total_fp += ((preds == 1) & (labels == 0)).sum().item()
            total_fn += ((preds == 0) & (labels == 1)).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        precision = total_tp / (total_tp + total_fp + 1e-9)
        recall = total_tp / (total_tp + total_fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        all_probs = []
        all_labels = []
        
        for batch in tqdm(val_loader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits'].squeeze(-1)
            
            loss = self.criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            total_loss += loss.item() * input_ids.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.size(0)
            
            total_tp += ((preds == 1) & (labels == 1)).sum().item()
            total_fp += ((preds == 1) & (labels == 0)).sum().item()
            total_fn += ((preds == 0) & (labels == 1)).sum().item()
            total_tn += ((preds == 0) & (labels == 0)).sum().item()
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        precision = total_tp / (total_tp + total_fp + 1e-9)
        recall = total_tp / (total_tp + total_fn + 1e-9)  # Detection rate
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        
        fpr = total_fp / (total_fp + total_tn + 1e-9)  # False positive rate
        fnr = total_fn / (total_tp + total_fn + 1e-9)  # False negative rate
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'detection_rate': recall,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'confusion_matrix': {
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'tn': total_tn
            }
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        save_every: int = 5
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Model size: {self.model.get_model_size_mb():.2f} MB")
        print("-" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            self.val_history.append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"Detection: {val_metrics['detection_rate']:.4f}, FNR: {val_metrics['fnr']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                print(f"  -> New best model saved!")
            
            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, val_metrics)
            
            # Early stopping check
            improved = self.early_stopping(val_metrics['loss'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pt', epoch, val_metrics)
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch
        }
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        return checkpoint
