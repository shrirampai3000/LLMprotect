#!/usr/bin/env python3
"""
Inference script using the trained model.

Usage:
    python run_inference.py "Your prompt here"
    python run_inference.py  # Interactive mode
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.architecture import AdversarialDetector
from src.models.inference import InferencePipeline
from src.data.tokenizer import PromptTokenizer
from src.core.pipeline import IntentBindingPipeline


def load_trained_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """Load the trained model from checkpoint."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Get config from checkpoint, or infer from saved weights
    config = checkpoint.get('config', {})
    
    # Auto-detect vocab_size from embedding weights if not in config
    state_dict = checkpoint['model_state_dict']
    actual_vocab_size = state_dict['embedding.weight'].shape[0]
    
    # Create model with same architecture
    model = AdversarialDetector(
        vocab_size=actual_vocab_size,  # Use actual vocab size from saved weights
        embedding_dim=config.get('embedding_dim', 256),
        max_seq_length=config.get('max_seq_length', 512),
        cnn_filters=config.get('cnn_filters', [128, 256]),
        cnn_kernel_sizes=config.get('cnn_kernel_sizes', [3, 5]),
        num_transformer_layers=config.get('num_transformer_layers', 4),
        num_attention_heads=config.get('num_attention_heads', 8),
        transformer_hidden_dim=config.get('transformer_hidden_dim', 1024),
        classifier_hidden_dims=config.get('classifier_hidden_dims', [512, 256]),
        dropout=0.1,
        classifier_dropout=0.3
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create tokenizer (load from saved vocabulary if available)
    tokenizer = PromptTokenizer(vocab_size=actual_vocab_size, max_length=512)
    vocab_path = Path("checkpoints/vocab.json")
    if vocab_path.exists():
        tokenizer.load_vocab(vocab_path)
    
    # Create inference pipeline
    inference = InferencePipeline(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        threshold=0.5
    )
    
    return inference


def main():
    print("=" * 60)
    print("ADVERSARIAL DETECTION - TRAINED MODEL")
    print("=" * 60)
    
    # Load model
    print("\nLoading trained model...")
    try:
        inference = load_trained_model()
        print("✓ Model loaded successfully!")
    except FileNotFoundError:
        print("✗ No trained model found. Run 'python train.py' first.")
        return
    
    # Check for command line argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        result = inference.predict(prompt)
        
        is_adv = "🚨 ADVERSARIAL" if result['is_adversarial'] else "✅ BENIGN"
        print(f"\nPrompt: {prompt}")
        print(f"Result: {is_adv}")
        print(f"Probability: {result['probability']:.4f}")
        return
    
    # Interactive mode
    print("\nEnter prompts to analyze (type 'quit' to exit):\n")
    
    while True:
        try:
            prompt = input("Prompt> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            result = inference.predict(prompt)
            
            is_adv = "🚨 ADVERSARIAL" if result['is_adversarial'] else "✅ BENIGN"
            print(f"  {is_adv} (probability: {result['probability']:.4f})\n")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
