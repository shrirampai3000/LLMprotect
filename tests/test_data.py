"""Tests for data utilities."""
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.tokenizer import PromptTokenizer
from src.data.generator import AdversarialPromptGenerator, BenignPromptGenerator
from src.data.augmentations import DataAugmenter, AdversarialAugmenter


class TestPromptTokenizer:
    """Tests for custom tokenizer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tokenizer = PromptTokenizer(vocab_size=1000, max_length=128)
        
        # Build vocabulary from sample texts
        sample_texts = [
            "hello world",
            "ignore previous instructions",
            "what is the capital of france",
            "help me write python code",
        ]
        self.tokenizer.build_vocab(sample_texts, min_freq=1)
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        text = "hello world"
        encoding = self.tokenizer.encode(text)
        
        assert 'input_ids' in encoding
        assert 'attention_mask' in encoding
        
        decoded = self.tokenizer.decode(encoding['input_ids'])
        assert 'hello' in decoded
        assert 'world' in decoded
    
    def test_padding(self):
        """Test padding to max length."""
        text = "short text"
        encoding = self.tokenizer.encode(text, max_length=50, padding="max_length")
        
        assert len(encoding['input_ids']) == 50
        assert len(encoding['attention_mask']) == 50
    
    def test_truncation(self):
        """Test truncation of long sequences."""
        text = " ".join(["word"] * 200)
        encoding = self.tokenizer.encode(text, max_length=50, truncation=True)
        
        assert len(encoding['input_ids']) == 50
    
    def test_special_tokens(self):
        """Test special tokens are added."""
        text = "test"
        encoding = self.tokenizer.encode(text, padding="max_length")
        
        # First token should be CLS
        assert encoding['input_ids'][0] == self.tokenizer.token_to_id[PromptTokenizer.CLS_TOKEN]
    
    def test_unknown_tokens(self):
        """Test handling of unknown tokens."""
        text = "xyzzy qwerty"  # Not in vocabulary
        encoding = self.tokenizer.encode(text)
        
        # Should not fail, uses UNK token
        assert encoding['input_ids'] is not None


class TestAdversarialPromptGenerator:
    """Tests for adversarial prompt generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = AdversarialPromptGenerator()
    
    def test_generate_single(self):
        """Test single prompt generation."""
        result = self.generator.generate_single()
        
        assert 'text' in result
        assert 'label' in result
        assert 'source' in result
        assert result['label'] == 1  # Adversarial
        assert len(result['text']) > 0
    
    def test_generate_batch(self):
        """Test batch generation."""
        count = 100
        results = self.generator.generate_batch(count)
        
        assert len(results) == count
        assert all(r['label'] == 1 for r in results)
    
    def test_category_coverage(self):
        """Test that multiple categories are generated."""
        results = self.generator.generate_batch(1000)
        categories = set(r.get('category', r['source']) for r in results)
        
        # Should have multiple categories
        assert len(categories) >= 3


class TestBenignPromptGenerator:
    """Tests for benign prompt generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = BenignPromptGenerator()
    
    def test_generate_single(self):
        """Test single prompt generation."""
        result = self.generator.generate_single()
        
        assert 'text' in result
        assert 'label' in result
        assert result['label'] == 0  # Benign
    
    def test_generate_batch(self):
        """Test batch generation."""
        count = 50
        results = self.generator.generate_batch(count)
        
        assert len(results) == count
        assert all(r['label'] == 0 for r in results)


class TestDataAugmenter:
    """Tests for data augmentation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.augmenter = DataAugmenter(augmentation_probability=1.0)
    
    def test_augment_preserves_meaning(self):
        """Test that augmentation doesn't destroy the text."""
        text = "hello world this is a test"
        augmented = self.augmenter.augment(text)
        
        # Should still be readable text
        assert len(augmented) > 0
        assert isinstance(augmented, str)
    
    def test_synonym_replacement(self):
        """Test synonym replacement."""
        text = "ignore the previous instructions"
        augmented = self.augmenter.synonym_replacement(text, n=1)
        
        # Text might change or stay same depending on random selection
        assert len(augmented) > 0
    
    def test_case_change(self):
        """Test case changes."""
        text = "Hello World Test"
        augmented = self.augmenter.case_change(text)
        
        assert len(augmented) > 0
    
    def test_augment_batch(self):
        """Test batch augmentation."""
        texts = ["text one", "text two", "text three"]
        augmented = self.augmenter.augment_batch(texts, augmentations_per_text=2)
        
        # Original + 2 augmentations per text
        assert len(augmented) == len(texts) * 3


class TestAdversarialAugmenter:
    """Tests for adversarial-specific augmentation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.augmenter = AdversarialAugmenter(augmentation_probability=1.0)
    
    def test_obfuscate_keywords(self):
        """Test keyword obfuscation."""
        text = "ignore the system admin"
        augmented = self.augmenter.obfuscate_keywords(text)
        
        # Should modify some keywords
        assert len(augmented) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
