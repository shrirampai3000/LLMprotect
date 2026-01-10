"""Tests for ML model components."""
import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.architecture import (
    AdversarialDetector,
    CNNFeatureExtractor,
    TransformerEncoder,
    MultiHeadAttention,
    PositionalEncoding,
    ClassificationHead
)


class TestPositionalEncoding:
    """Tests for positional encoding."""
    
    def test_output_shape(self):
        """Test output shape matches input."""
        pe = PositionalEncoding(d_model=256, max_len=512)
        x = torch.randn(2, 100, 256)
        
        output = pe(x)
        
        assert output.shape == x.shape
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        pe = PositionalEncoding(d_model=128, max_len=1024)
        
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(1, seq_len, 128)
            output = pe(x)
            assert output.shape == x.shape


class TestCNNFeatureExtractor:
    """Tests for CNN feature extractor."""
    
    def test_output_shape(self):
        """Test output shape after convolution and pooling."""
        cnn = CNNFeatureExtractor(input_dim=256, filters=[128, 256])
        x = torch.randn(2, 100, 256)
        
        output = cnn(x)
        
        # After pooling with factor 2
        assert output.shape[0] == 2  # Batch
        assert output.shape[1] == 50  # Sequence halved
        assert output.shape[2] == 256  # Output filters
    
    def test_output_dim_property(self):
        """Test output dimension property."""
        cnn = CNNFeatureExtractor(filters=[64, 128, 256])
        assert cnn.output_dim == 256


class TestMultiHeadAttention:
    """Tests for multi-head attention."""
    
    def test_output_shape(self):
        """Test output shape."""
        attn = MultiHeadAttention(d_model=256, num_heads=8)
        x = torch.randn(2, 50, 256)
        
        output, _ = attn(x)
        
        assert output.shape == x.shape
    
    def test_attention_weights_shape(self):
        """Test attention weights shape."""
        attn = MultiHeadAttention(d_model=256, num_heads=8)
        x = torch.randn(2, 50, 256)
        
        output, attention = attn(x, return_attention=True)
        
        assert attention is not None
        assert attention.shape == (2, 8, 50, 50)  # batch, heads, seq, seq
    
    def test_with_mask(self):
        """Test attention with mask."""
        attn = MultiHeadAttention(d_model=256, num_heads=8)
        x = torch.randn(2, 50, 256)
        mask = torch.ones(2, 50)
        mask[:, 40:] = 0  # Mask last 10 positions
        
        output, _ = attn(x, mask=mask)
        
        assert output.shape == x.shape


class TestTransformerEncoder:
    """Tests for transformer encoder."""
    
    def test_output_shape(self):
        """Test output shape."""
        encoder = TransformerEncoder(d_model=256, num_layers=4, num_heads=8)
        x = torch.randn(2, 50, 256)
        
        output, _ = encoder(x)
        
        assert output.shape == x.shape
    
    def test_attention_collection(self):
        """Test attention weights collection from all layers."""
        encoder = TransformerEncoder(d_model=256, num_layers=4, num_heads=8)
        x = torch.randn(2, 50, 256)
        
        output, attentions = encoder(x, return_attention=True)
        
        assert len(attentions) == 4  # One per layer


class TestClassificationHead:
    """Tests for classification head."""
    
    def test_output_shape(self):
        """Test output shape."""
        head = ClassificationHead(input_dim=256, hidden_dims=[512, 256])
        x = torch.randn(2, 256)
        
        output = head(x)
        
        assert output.shape == (2, 1)


class TestAdversarialDetector:
    """Tests for complete adversarial detector model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = AdversarialDetector(
            vocab_size=1000,
            embedding_dim=64,
            max_seq_length=128,
            cnn_filters=[32, 64],
            num_transformer_layers=2,
            num_attention_heads=4,
            transformer_hidden_dim=256,
            classifier_hidden_dims=[128, 64]
        )
    
    def test_forward_pass(self):
        """Test forward pass."""
        input_ids = torch.randint(0, 1000, (2, 100))
        attention_mask = torch.ones(2, 100)
        
        outputs = self.model(input_ids, attention_mask)
        
        assert 'logits' in outputs
        assert 'probabilities' in outputs
        assert outputs['logits'].shape == (2, 1)
        assert outputs['probabilities'].shape == (2, 1)
    
    def test_probability_range(self):
        """Test that probabilities are in valid range."""
        input_ids = torch.randint(0, 1000, (5, 50))
        
        outputs = self.model(input_ids)
        probs = outputs['probabilities']
        
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
    
    def test_attention_output(self):
        """Test attention output."""
        input_ids = torch.randint(0, 1000, (2, 100))
        
        outputs = self.model(input_ids, return_attention=True)
        
        assert 'attentions' in outputs
        assert outputs['attentions'] is not None
    
    def test_predict_method(self):
        """Test predict method."""
        input_ids = torch.randint(0, 1000, (3, 50))
        
        results = self.model.predict(input_ids, threshold=0.5)
        
        assert 'predictions' in results
        assert 'probabilities' in results
        assert 'is_adversarial' in results
    
    def test_parameter_count(self):
        """Test parameter counting."""
        param_count = self.model.count_parameters()
        
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_model_size(self):
        """Test model size calculation."""
        size_mb = self.model.get_model_size_mb()
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
    
    def test_eval_mode(self):
        """Test model behavior in eval mode."""
        self.model.eval()
        
        input_ids = torch.randint(0, 1000, (2, 50))
        
        with torch.no_grad():
            outputs1 = self.model(input_ids)
            outputs2 = self.model(input_ids)
        
        # Should be deterministic in eval mode
        assert torch.allclose(outputs1['logits'], outputs2['logits'])


class TestModelGradients:
    """Tests for gradient flow."""
    
    def test_backward_pass(self):
        """Test that gradients flow correctly."""
        model = AdversarialDetector(
            vocab_size=500,
            embedding_dim=32,
            max_seq_length=64,
            cnn_filters=[16, 32],
            num_transformer_layers=1,
            num_attention_heads=2,
            transformer_hidden_dim=64,
            classifier_hidden_dims=[32]
        )
        
        input_ids = torch.randint(0, 500, (2, 30))
        labels = torch.tensor([0.0, 1.0])
        
        outputs = model(input_ids)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs['logits'].squeeze(), labels
        )
        
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
