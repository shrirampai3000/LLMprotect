"""Custom tokenizer for prompt processing."""
import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter
import unicodedata


class PromptTokenizer:
    """
    Custom tokenizer for adversarial prompt detection.
    
    Uses a WordPiece-style tokenization approach optimized for
    detecting adversarial patterns in prompts.
    """
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    
    def __init__(
        self,
        vocab_size: int = 30000,
        max_length: int = 512,
        vocab_path: Optional[Path] = None
    ):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            vocab_path: Optional path to pre-built vocabulary
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Initialize vocabulary with special tokens
        self.token_to_id: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
            self.SEP_TOKEN: 3,
        }
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        
        if vocab_path and vocab_path.exists():
            self.load_vocab(vocab_path)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for tokenization.
        
        - Unicode normalization (NFKC)
        - Lowercase
        - Handle excessive whitespace
        """
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        
        # Lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def basic_tokenize(self, text: str) -> List[str]:
        """
        Split text into basic tokens (words and punctuation).
        """
        # Normalize first
        text = self.normalize_text(text)
        
        # Split on whitespace and punctuation, keeping punctuation as tokens
        pattern = r"(\w+|[^\w\s])"
        tokens = re.findall(pattern, text)
        
        return tokens
    
    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text documents
            min_freq: Minimum frequency for a token to be included
        """
        # Count all tokens
        token_counts = Counter()
        
        for text in texts:
            tokens = self.basic_tokenize(text)
            token_counts.update(tokens)
        
        # Sort by frequency (descending) and take top vocab_size - special tokens
        num_special = len(self.token_to_id)
        max_vocab = self.vocab_size - num_special
        
        sorted_tokens = sorted(
            [(token, count) for token, count in token_counts.items() if count >= min_freq],
            key=lambda x: x[1],
            reverse=True
        )[:max_vocab]
        
        # Add to vocabulary
        for token, _ in sorted_tokens:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
        
        print(f"Built vocabulary with {len(self.token_to_id)} tokens")
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length (uses self.max_length if None)
            padding: Padding strategy ("max_length", "longest", "none")
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        max_len = max_length or self.max_length
        
        # Tokenize
        tokens = self.basic_tokenize(text)
        
        # Add special tokens
        tokens = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]
        
        # Truncate if needed
        if truncation and len(tokens) > max_len:
            tokens = tokens[:max_len - 1] + [self.SEP_TOKEN]
        
        # Convert to IDs
        input_ids = [
            self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
            for token in tokens
        ]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if padding == "max_length":
            padding_length = max_len - len(input_ids)
            input_ids = input_ids + [self.token_to_id[self.PAD_TOKEN]] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        special_tokens = {
            self.token_to_id[self.PAD_TOKEN],
            self.token_to_id[self.UNK_TOKEN],
            self.token_to_id[self.CLS_TOKEN],
            self.token_to_id[self.SEP_TOKEN],
        }
        
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_tokens:
                continue
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def save_vocab(self, path: Path) -> None:
        """Save vocabulary to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "vocab": self.token_to_id,
                "vocab_size": self.vocab_size,
                "max_length": self.max_length
            }, f, indent=2)
    
    def load_vocab(self, path: Path) -> None:
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token_to_id = data["vocab"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.vocab_size = data.get("vocab_size", self.vocab_size)
        self.max_length = data.get("max_length", self.max_length)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)

