"""Data augmentation utilities for the adversarial detection dataset."""
import random
import re
from typing import List, Dict, Optional


class DataAugmenter:
    """
    Augments prompts using various techniques to improve model robustness.
    
    Techniques:
    1. Synonym replacement
    2. Random character operations (insert, delete, swap)
    3. Case changes
    4. Whitespace perturbations
    5. Paraphrasing patterns
    """
    
    def __init__(self, augmentation_probability: float = 0.5):
        """
        Initialize the augmenter.
        
        Args:
            augmentation_probability: Probability of applying each augmentation
        """
        self.aug_prob = augmentation_probability
        self._init_synonyms()
    
    def _init_synonyms(self):
        """Initialize synonym mappings for common words."""
        self.synonyms = {
            "ignore": ["disregard", "skip", "bypass", "overlook", "neglect"],
            "previous": ["prior", "earlier", "preceding", "former", "past"],
            "instructions": ["directives", "commands", "guidelines", "orders", "rules"],
            "reveal": ["show", "display", "expose", "uncover", "disclose"],
            "secret": ["confidential", "private", "hidden", "classified", "restricted"],
            "help": ["assist", "aid", "support", "guide", "help"],
            "write": ["compose", "create", "draft", "generate", "produce"],
            "explain": ["describe", "clarify", "elaborate", "illustrate", "detail"],
            "code": ["program", "script", "software", "implementation", "logic"],
            "data": ["information", "records", "details", "content", "material"],
            "access": ["obtain", "retrieve", "get", "reach", "acquire"],
            "system": ["platform", "application", "service", "framework", "infrastructure"],
            "user": ["person", "individual", "customer", "client", "operator"],
            "password": ["credentials", "key", "authentication", "passcode", "secret"],
            "execute": ["run", "perform", "carry out", "implement", "complete"],
            "delete": ["remove", "erase", "clear", "wipe", "eliminate"],
        }
    
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """
        Replace random words with their synonyms.
        
        Args:
            text: Input text
            n: Number of words to replace
            
        Returns:
            Augmented text
        """
        words = text.split()
        word_indices = list(range(len(words)))
        random.shuffle(word_indices)
        
        replaced = 0
        for idx in word_indices:
            word = words[idx].lower().strip('.,!?')
            if word in self.synonyms:
                synonym = random.choice(self.synonyms[word])
                # Preserve original case
                if words[idx][0].isupper():
                    synonym = synonym.capitalize()
                words[idx] = synonym
                replaced += 1
                if replaced >= n:
                    break
        
        return ' '.join(words)
    
    def random_char_insert(self, text: str, n: int = 1) -> str:
        """Insert random characters at random positions (simulates typos)."""
        chars = list(text)
        positions = random.sample(range(len(chars)), min(n, len(chars)))
        
        for pos in sorted(positions, reverse=True):
            char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz ')
            chars.insert(pos, char_to_insert)
        
        return ''.join(chars)
    
    def random_char_delete(self, text: str, n: int = 1) -> str:
        """Delete random characters (simulates typos)."""
        chars = list(text)
        if len(chars) <= n:
            return text
        
        positions = random.sample(range(len(chars)), n)
        for pos in sorted(positions, reverse=True):
            del chars[pos]
        
        return ''.join(chars)
    
    def random_char_swap(self, text: str, n: int = 1) -> str:
        """Swap adjacent characters (simulates typos)."""
        chars = list(text)
        positions = random.sample(range(len(chars) - 1), min(n, len(chars) - 1))
        
        for pos in positions:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        
        return ''.join(chars)
    
    def case_change(self, text: str) -> str:
        """Randomly change case of some words."""
        words = text.split()
        
        for i in range(len(words)):
            if random.random() < 0.2:
                if random.random() < 0.5:
                    words[i] = words[i].upper()
                else:
                    words[i] = words[i].lower()
        
        return ' '.join(words)
    
    def whitespace_perturbation(self, text: str) -> str:
        """Add or remove whitespace randomly."""
        # Add extra spaces
        if random.random() < 0.5:
            words = text.split()
            result = []
            for word in words:
                result.append(word)
                if random.random() < 0.1:
                    result.append('')  # Extra space
            text = ' '.join(result)
        
        return text
    
    def add_prefix_suffix(self, text: str) -> str:
        """Add common conversational prefixes or suffixes."""
        prefixes = [
            "Hey, ",
            "Hi there, ",
            "Please ",
            "Could you please ",
            "I need help: ",
            "Quick question: ",
            "",
            "",  # More likely to not add
            "",
        ]
        
        suffixes = [
            "",
            "",  # More likely to not add
            "",
            " Thanks!",
            " Thank you.",
            " Please.",
            " ASAP",
            "?",
        ]
        
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        return prefix + text + suffix
    
    def augment(self, text: str, num_augmentations: int = 1) -> str:
        """
        Apply random augmentations to the text.
        
        Args:
            text: Input text
            num_augmentations: Number of augmentation techniques to apply
            
        Returns:
            Augmented text
        """
        augmentation_funcs = [
            lambda t: self.synonym_replacement(t),
            lambda t: self.random_char_insert(t),
            lambda t: self.random_char_delete(t),
            lambda t: self.random_char_swap(t),
            lambda t: self.case_change(t),
            lambda t: self.whitespace_perturbation(t),
            lambda t: self.add_prefix_suffix(t),
        ]
        
        # Select random augmentations
        selected = random.sample(augmentation_funcs, min(num_augmentations, len(augmentation_funcs)))
        
        result = text
        for func in selected:
            if random.random() < self.aug_prob:
                result = func(result)
        
        return result
    
    def augment_batch(self, texts: List[str], augmentations_per_text: int = 2) -> List[str]:
        """
        Augment a batch of texts.
        
        Args:
            texts: List of input texts
            augmentations_per_text: Number of augmented versions per text
            
        Returns:
            List of augmented texts (original texts + augmented versions)
        """
        result = []
        for text in texts:
            result.append(text)  # Keep original
            for _ in range(augmentations_per_text):
                result.append(self.augment(text))
        return result


class AdversarialAugmenter(DataAugmenter):
    """
    Specialized augmenter for adversarial prompts.
    
    Includes additional techniques specific to adversarial patterns.
    """
    
    def __init__(self, augmentation_probability: float = 0.5):
        super().__init__(augmentation_probability)
        self._init_adversarial_patterns()
    
    def _init_adversarial_patterns(self):
        """Initialize adversarial-specific augmentation patterns."""
        self.pattern_variations = {
            "ignore": ["IGNORE", "Ignore", "ignor3", "1gnore", "!gnore"],
            "system": ["SYSTEM", "System", "syst3m", "5ystem", "$ystem"],
            "admin": ["ADMIN", "Admin", "adm1n", "4dmin", "@dmin"],
            "override": ["OVERRIDE", "Override", "overr1de", "0verride"],
            "bypass": ["BYPASS", "Bypass", "byp4ss", "bypa$$"],
        }
    
    def obfuscate_keywords(self, text: str) -> str:
        """Obfuscate adversarial keywords with character substitutions."""
        result = text
        for keyword, variations in self.pattern_variations.items():
            if keyword in result.lower():
                variation = random.choice(variations)
                result = re.sub(keyword, variation, result, flags=re.IGNORECASE)
        return result
    
    def add_noise_characters(self, text: str) -> str:
        """Add invisible or noise characters between words."""
        noise_chars = ['\u200b', '\u200c', '\u200d', '']  # Zero-width chars + empty
        words = text.split()
        result = []
        for word in words:
            result.append(word)
            if random.random() < 0.1:
                result.append(random.choice(noise_chars))
        return ' '.join(result)
    
    def augment(self, text: str, num_augmentations: int = 1) -> str:
        """Apply adversarial-specific augmentations."""
        # First apply base augmentations
        result = super().augment(text, num_augmentations)
        
        # Then apply adversarial-specific augmentations
        if random.random() < self.aug_prob:
            result = self.obfuscate_keywords(result)
        
        if random.random() < self.aug_prob * 0.5:
            result = self.add_noise_characters(result)
        
        return result
