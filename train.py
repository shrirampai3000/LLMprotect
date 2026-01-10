#!/usr/bin/env python3
"""
Training Script for Adversarial Detection Model.

This script:
1. Loads datasets from HuggingFace
2. Trains the CNN-Transformer model
3. Evaluates and saves checkpoints
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader

from src.config import ProjectConfig, config
from src.data.dataset import (
    RealDatasetLoader, 
    create_combined_dataset, 
    create_data_splits,
    create_data_loaders,
    AdversarialDataset
)
from src.data.tokenizer import PromptTokenizer
from src.data.augmentations import DataAugmenter
from src.models.architecture import AdversarialDetector
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import MetricsCalculator, generate_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adversarial Detection Model")
    
    # Data
    parser.add_argument("--use-synthetic", action="store_true", default=True,
                        help="Include synthetic prompts in training")
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Directory for caching datasets")
    
    # Model
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--early-stopping", type=int, default=5,
                        help="Early stopping patience")
    
    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, or cpu")
    
    # Mode
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on existing model")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for testing (limited data)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("ADVERSARIAL DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n📥 Loading datasets...")
    
    loader = RealDatasetLoader(cache_dir=Path(args.cache_dir))
    
    if args.quick:
        # Quick mode: use only synthetic data
        from src.data.generator import AdversarialPromptGenerator, BenignPromptGenerator
        
        adv_gen = AdversarialPromptGenerator()
        ben_gen = BenignPromptGenerator()
        
        all_data = adv_gen.generate_batch(500) + ben_gen.generate_batch(500)
        stats = {"total": 1000, "adversarial": 500, "benign": 500}
    else:
        all_data, stats = create_combined_dataset(
            loader, 
            include_synthetic=args.use_synthetic
        )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total examples: {stats['total']}")
    print(f"   Adversarial: {stats['adversarial']}")
    print(f"   Benign: {stats['benign']}")
    
    # Step 2: Create data splits
    print("\n📑 Creating train/val/test splits...")
    
    train_data, val_data, test_data = create_data_splits(all_data)
    
    print(f"   Train: {len(train_data)}")
    print(f"   Val: {len(val_data)}")
    print(f"   Test: {len(test_data)}")
    
    # Step 3: Initialize tokenizer
    print("\n🔤 Building tokenizer...")
    
    # Use custom tokenizer (no external dependencies)
    tokenizer = PromptTokenizer(vocab_size=30522, max_length=512)
    tokenizer.build_vocab([d['text'] for d in train_data], min_freq=2)
    vocab_size = len(tokenizer)
    print(f"   Built custom tokenizer (vocab: {vocab_size})")
    
    # Save vocabulary for inference
    vocab_path = Path(args.checkpoint_dir) / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab(vocab_path)
    print(f"   Saved vocabulary to: {vocab_path}")
    
    # Step 4: Create data loaders
    print("\n🔄 Creating data loaders...")
    
    augmenter = DataAugmenter(augmentation_probability=0.5)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        augmenter=augmenter
    )
    
    # Step 5: Initialize model
    print("\n🧠 Initializing model...")
    
    model = AdversarialDetector(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        max_seq_length=512,
        cnn_filters=[128, 256],
        cnn_kernel_sizes=[3, 5],
        num_transformer_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        transformer_hidden_dim=1024,
        classifier_hidden_dims=[512, 256],
        dropout=0.1,
        classifier_dropout=0.3
    )
    
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Model size: {model.get_model_size_mb():.2f} MB")
    
    # Step 6: Train
    if not args.eval_only:
        print("\n🏋️ Starting training...")
        
        checkpoint_dir = Path(args.checkpoint_dir)
        trainer = ModelTrainer(model, device=device, checkpoint_dir=checkpoint_dir)
        
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * args.epochs
        
        trainer.setup_training(
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=min(500, total_steps // 10),
            total_steps=total_steps,
            label_smoothing=0.1,
            class_weights=[1.0, 1.5],  # Slightly weight adversarial class
            early_stopping_patience=args.early_stopping
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_every=args.save_every
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Final epoch: {history['final_epoch']}")
        print(f"   Best val loss: {history['best_val_loss']:.4f}")
    else:
        print("\n📂 Loading trained model...")
        trainer = ModelTrainer(model, device=device, checkpoint_dir=Path(args.checkpoint_dir))
        
        try:
            trainer.load_checkpoint("best_model.pt")
            print("   Loaded best_model.pt")
        except FileNotFoundError:
            print("   No checkpoint found, using untrained model")
    
    # Step 7: Evaluate
    print("\n📈 Running evaluation...")
    
    model.eval()
    
    all_probs = []
    all_labels = []
    latencies = []
    
    import time
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            start = time.time()
            outputs = model(input_ids, attention_mask)
            latencies.append((time.time() - start) * 1000 / len(input_ids))
            
            probs = outputs['probabilities'].squeeze(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    import numpy as np
    
    calculator = MetricsCalculator(threshold=0.5)
    result = calculator.compute_all(
        np.array(all_labels),
        np.array(all_probs),
        latencies=latencies,
        model_size_mb=model.get_model_size_mb()
    )
    
    # Print report
    report = generate_evaluation_report(result)
    print(report)
    
    # Save evaluation results
    results_path = Path(args.checkpoint_dir) / "evaluation_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\n📁 Results saved to: {results_path}")
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
