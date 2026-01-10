"""
CNN-Transformer Hybrid Architecture for Adversarial Prompt Detection.

Architecture:
1. Embedding Layer (256-dim)
2. CNN Feature Extractor (local pattern detection)
3. Transformer Encoder (4 layers, 8 heads)
4. Classification Head (Dense layers with dropout)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNFeatureExtractor(nn.Module):
    """
    1D Convolutional layers for extracting local patterns.
    
    Detects local adversarial patterns like "ignore previous",
    Base64 strings, and escape sequences.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        filters: list = None,
        kernel_sizes: list = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        filters = filters or [128, 256]
        kernel_sizes = kernel_sizes or [3, 5]
        
        self.convs = nn.ModuleList()
        in_channels = input_dim
        
        for i, (num_filters, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            self.convs.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = num_filters
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.output_dim = filters[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embedding_dim)
            
        Returns:
            Tensor of shape (batch, seq_len//2, output_dim)
        """
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        for conv in self.convs:
            x = conv(x)
        
        x = self.pool(x)
        
        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional attention output."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        if return_attention:
            return output, attention
        return output, None


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with residual connections and layer norm."""
        # Self-attention with residual
        attn_out, attention = self.attention(x, mask, return_attention)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attention


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through all transformer layers.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask
            return_attention: Whether to return attention weights from all layers
            
        Returns:
            Output tensor and optionally list of attention weights
        """
        all_attentions = [] if return_attention else None
        
        for layer in self.layers:
            x, attention = layer(x, mask, return_attention)
            if return_attention and attention is not None:
                all_attentions.append(attention)
        
        return x, all_attentions


class ClassificationHead(nn.Module):
    """Classification head with dense layers and dropout."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [512, 256]
        
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i == 0 else dropout * 0.7)
            ])
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pooled features (batch, input_dim)
            
        Returns:
            Classification logits (batch, 1)
        """
        return self.classifier(x)


class AdversarialDetector(nn.Module):
    """
    Complete CNN-Transformer hybrid model for adversarial prompt detection.
    
    Architecture:
    1. Token Embedding (learnable)
    2. Positional Encoding (sinusoidal)
    3. CNN Feature Extractor (local patterns)
    4. Transformer Encoder (4 layers, 8 heads)
    5. Global Average Pooling
    6. Classification Head (512 -> 256 -> 1)
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 256,
        max_seq_length: int = 512,
        cnn_filters: list = None,
        cnn_kernel_sizes: list = None,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        transformer_hidden_dim: int = 1024,
        classifier_hidden_dims: list = None,
        dropout: float = 0.1,
        classifier_dropout: float = 0.3
    ):
        super().__init__()
        
        cnn_filters = cnn_filters or [128, 256]
        cnn_kernel_sizes = cnn_kernel_sizes or [3, 5]
        classifier_hidden_dims = classifier_hidden_dims or [512, 256]
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=embedding_dim,
            max_len=max_seq_length,
            dropout=dropout
        )
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(
            input_dim=embedding_dim,
            filters=cnn_filters,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout
        )
        
        # Project CNN output to transformer dimension
        self.cnn_proj = nn.Linear(cnn_filters[-1], embedding_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=embedding_dim,
            num_layers=num_transformer_layers,
            num_heads=num_attention_heads,
            d_ff=transformer_hidden_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=embedding_dim,
            hidden_dims=classifier_hidden_dims,
            dropout=classifier_dropout
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Mask for padding tokens (batch, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with 'logits', 'probabilities', and optionally 'attentions'
        """
        # Embedding + positional encoding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # CNN feature extraction
        cnn_features = self.cnn(x)
        
        # Project back to embedding dimension
        cnn_features = self.cnn_proj(cnn_features)
        
        # Adjust mask for pooled sequence
        if attention_mask is not None:
            # CNN pooling reduces sequence length by factor of 2
            pooled_len = cnn_features.size(1)
            # Simple approach: take every other mask value
            mask_indices = torch.arange(0, min(pooled_len * 2, attention_mask.size(1)), 2)
            if mask_indices.size(0) > pooled_len:
                mask_indices = mask_indices[:pooled_len]
            transformer_mask = attention_mask[:, mask_indices[:pooled_len]]
        else:
            transformer_mask = None
        
        # Transformer encoding
        transformer_out, attentions = self.transformer(
            cnn_features, transformer_mask, return_attention
        )
        
        # Global average pooling
        if transformer_mask is not None:
            # Masked mean
            mask_expanded = transformer_mask.unsqueeze(-1).float()
            pooled = (transformer_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = transformer_out.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        probabilities = torch.sigmoid(logits)
        
        result = {
            'logits': logits,
            'probabilities': probabilities
        }
        
        if return_attention:
            result['attentions'] = attentions
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with thresholding.
        
        Args:
            input_ids: Token IDs
            attention_mask: Padding mask
            threshold: Classification threshold
            
        Returns:
            Dictionary with 'predictions', 'probabilities', and 'is_adversarial'
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = outputs['probabilities']
            predictions = (probs > threshold).float()
            
            return {
                'predictions': predictions,
                'probabilities': probs,
                'is_adversarial': predictions.squeeze(-1).bool()
            }
    
    @classmethod
    def from_config(cls, config) -> 'AdversarialDetector':
        """Create model from configuration object."""
        return cls(
            vocab_size=config.model.vocab_size,
            embedding_dim=config.model.embedding_dim,
            max_seq_length=config.model.max_seq_length,
            cnn_filters=config.model.cnn_filters,
            cnn_kernel_sizes=config.model.cnn_kernel_sizes,
            num_transformer_layers=config.model.num_transformer_layers,
            num_attention_heads=config.model.num_attention_heads,
            transformer_hidden_dim=config.model.transformer_hidden_dim,
            classifier_hidden_dims=config.model.classifier_hidden_dims,
            dropout=config.model.transformer_dropout,
            classifier_dropout=config.model.classifier_dropout
        )
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
