"""
Inference pipeline for adversarial prompt detection.

Features:
- Single and batch prediction
- Attention visualization for explainability
- Latency measurement
- ONNX export support
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import time
import json


class InferencePipeline:
    """
    Optimized inference pipeline for the adversarial detection model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained adversarial detection model
            tokenizer: Tokenizer instance
            device: Device for inference
            threshold: Classification threshold
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
    
    @torch.no_grad()
    def predict(
        self,
        text: str,
        return_attention: bool = False
    ) -> Dict:
        """
        Predict on a single text input.
        
        Args:
            text: Input prompt text
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Tokenize
        encoding = self.tokenizer.encode(text)
        input_ids = torch.tensor([encoding['input_ids']], device=self.device)
        attention_mask = torch.tensor([encoding['attention_mask']], device=self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask, return_attention=return_attention)
        
        probability = outputs['probabilities'].item()
        is_adversarial = probability > self.threshold
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        result = {
            'text': text,
            'probability': probability,
            'is_adversarial': is_adversarial,
            'confidence': probability if is_adversarial else 1 - probability,
            'latency_ms': latency
        }
        
        if return_attention and 'attentions' in outputs:
            result['attentions'] = outputs['attentions']
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict on a batch of texts.
        
        Args:
            texts: List of input prompts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = [self.tokenizer.encode(t) for t in batch_texts]
            
            input_ids = torch.tensor(
                [e['input_ids'] for e in encodings],
                device=self.device
            )
            attention_mask = torch.tensor(
                [e['attention_mask'] for e in encodings],
                device=self.device
            )
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            probabilities = outputs['probabilities'].squeeze(-1).cpu().numpy()
            
            for text, prob in zip(batch_texts, probabilities):
                is_adversarial = prob > self.threshold
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'probability': float(prob),
                    'is_adversarial': bool(is_adversarial),
                    'confidence': float(prob if is_adversarial else 1 - prob)
                })
        
        total_time = (time.time() - start_time) * 1000
        avg_latency = total_time / len(texts)
        
        # Add timing info to first result
        if results:
            results[0]['total_batch_time_ms'] = total_time
            results[0]['avg_latency_ms'] = avg_latency
        
        return results
    
    def get_attention_visualization(
        self,
        text: str,
        layer_idx: int = -1,
        head_idx: int = 0
    ) -> Dict:
        """
        Get attention weights for visualization.
        
        Args:
            text: Input text
            layer_idx: Which transformer layer to visualize (-1 for last)
            head_idx: Which attention head to visualize
            
        Returns:
            Dictionary with tokens and attention weights
        """
        result = self.predict(text, return_attention=True)
        
        if 'attentions' not in result or not result['attentions']:
            return {'error': 'No attention weights available'}
        
        # Get normalized text tokens
        encoding = self.tokenizer.encode(text)
        input_ids = encoding['input_ids']
        
        # Decode tokens for display
        tokens = []
        for token_id in input_ids:
            if token_id in self.tokenizer.id_to_token:
                tokens.append(self.tokenizer.id_to_token[token_id])
            else:
                tokens.append('[UNK]')
        
        # Get attention from specified layer
        attention = result['attentions'][layer_idx]  # (batch, heads, seq, seq)
        attention = attention[0, head_idx].cpu().numpy()  # (seq, seq)
        
        # Average attention received by each token
        token_importance = attention.mean(axis=0)
        
        return {
            'tokens': tokens,
            'attention_matrix': attention.tolist(),
            'token_importance': token_importance.tolist(),
            'is_adversarial': result['is_adversarial'],
            'probability': result['probability']
        }
    
    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        tokenizer,
        model_class,
        model_kwargs: Dict,
        device: str = 'cuda'
    ) -> 'InferencePipeline':
        """
        Load a trained model for inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer: Tokenizer instance
            model_class: Model class
            model_kwargs: Arguments for model initialization
            device: Device for inference
            
        Returns:
            InferencePipeline instance
        """
        model = model_class(**model_kwargs)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, tokenizer, device)
    
    def export_onnx(self, output_path: Path, sample_length: int = 512):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            sample_length: Sequence length for export
        """
        import torch.onnx
        
        # Create dummy input
        dummy_input_ids = torch.zeros(1, sample_length, dtype=torch.long, device=self.device)
        dummy_attention_mask = torch.ones(1, sample_length, dtype=torch.long, device=self.device)
        
        # Export
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits', 'probabilities'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'},
                'probabilities': {0: 'batch_size'}
            },
            opset_version=14
        )
        
        print(f"Model exported to {output_path}")
    
    def benchmark_latency(
        self,
        sample_text: str,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict:
        """
        Benchmark inference latency.
        
        Returns:
            Dictionary with latency statistics
        """
        # Warmup
        for _ in range(warmup_iterations):
            self.predict(sample_text)
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.time()
            self.predict(sample_text)
            latencies.append((time.time() - start) * 1000)
        
        latencies = np.array(latencies)
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        }
