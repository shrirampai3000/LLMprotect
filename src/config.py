"""Centralized configuration management for the project."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the CNN-Transformer model."""
    # Embedding
    vocab_size: int = 30000
    max_seq_length: int = 512
    embedding_dim: int = 256
    
    # CNN layers
    cnn_filters: List[int] = field(default_factory=lambda: [128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5])
    
    # Transformer
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_hidden_dim: int = 1024
    transformer_dropout: float = 0.1
    
    # Classification head
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    classifier_dropout: float = 0.3
    
    # Detection threshold
    adversarial_threshold: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Training params
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # Learning rate scheduling
    warmup_steps: int = 500
    lr_scheduler: str = "cosine"
    
    # Regularization
    label_smoothing: float = 0.1
    
    # Class weights (for imbalanced data)
    class_weights: List[float] = field(default_factory=lambda: [1.0, 1.5])
    
    # Checkpointing
    save_every_n_epochs: int = 5


@dataclass
class CryptoConfig:
    """Configuration for cryptographic operations."""
    # Key management
    key_storage_path: Path = field(default_factory=lambda: Path("keys"))
    key_rotation_days: int = 90
    
    # Authorization tokens
    token_expiry_seconds: int = 300  # 5 minutes
    clock_skew_tolerance: int = 30  # seconds
    
    # Nonce
    nonce_size_bytes: int = 32
    nonce_ttl_seconds: int = 900  # 15 minutes
    
    # Merkle tree
    merkle_log_path: Path = field(default_factory=lambda: Path("audit_logs"))


@dataclass
class DataConfig:
    """Configuration for dataset handling."""
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    
    # HuggingFace datasets to use
    hf_datasets: List[str] = field(default_factory=lambda: [
        "deepset/prompt-injections",
        "Anthropic/hh-rlhf",
    ])
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Preprocessing
    max_prompt_length: int = 512
    min_prompt_length: int = 5
    
    # Augmentation
    augmentation_ratio: float = 0.2  # 20% of data augmented


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Metrics to compute
    primary_metrics: List[str] = field(default_factory=lambda: [
        "detection_rate", "fnr", "fpr", "precision", "f1", "auc_roc"
    ])
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "mcc", "average_precision"
    ])
    
    # Targets
    target_detection_rate: float = 0.99
    target_fnr: float = 0.01
    target_fpr: float = 0.05
    target_latency_ms: float = 50.0


@dataclass
class ProjectConfig:
    """Master configuration containing all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # General
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    device: str = "cuda"  # or "cpu"
    seed: int = 42
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data.data_dir.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.crypto.key_storage_path.mkdir(parents=True, exist_ok=True)
        self.crypto.merkle_log_path.mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = ProjectConfig()
