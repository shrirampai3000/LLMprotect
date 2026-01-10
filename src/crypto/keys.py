"""
Ed25519 Key Management for Cryptographic Intent Binding.

Features:
- Secure key generation using Ed25519
- Key storage with encryption (AES-256-GCM)
- Key rotation support
- HSM integration placeholder
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


@dataclass
class KeyPair:
    """Represents an Ed25519 key pair."""
    private_key: SigningKey
    public_key: VerifyKey
    created_at: datetime
    expires_at: Optional[datetime] = None
    key_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Serialize key metadata (not the private key!)."""
        return {
            'public_key': self.public_key.encode(encoder=HexEncoder).decode(),
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'key_id': self.key_id
        }


class KeyManager:
    """
    Manages Ed25519 key pairs for cryptographic operations.
    
    Security features:
    - Password-protected private key storage
    - PBKDF2 key derivation (100k iterations)
    - AES-256-GCM encryption for at-rest protection
    - Key rotation support
    """
    
    def __init__(self, storage_path: Path, password: Optional[str] = None):
        """
        Initialize the key manager.
        
        Args:
            storage_path: Directory for key storage
            password: Password for key encryption (required for production)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.password = password
        
        self.current_keypair: Optional[KeyPair] = None
        self._key_cache: dict = {}
    
    def generate_keypair(self, key_id: Optional[str] = None, validity_days: int = 90) -> KeyPair:
        """
        Generate a new Ed25519 key pair.
        
        Args:
            key_id: Optional identifier for the key
            validity_days: Number of days until key expires
            
        Returns:
            New KeyPair instance
        """
        # Generate Ed25519 key pair
        signing_key = SigningKey.generate()
        verify_key = signing_key.verify_key
        
        now = datetime.utcnow()
        expires = now + timedelta(days=validity_days)
        
        # Generate key ID if not provided
        if key_id is None:
            key_id = hashlib.sha256(
                verify_key.encode() + now.isoformat().encode()
            ).hexdigest()[:16]
        
        keypair = KeyPair(
            private_key=signing_key,
            public_key=verify_key,
            created_at=now,
            expires_at=expires,
            key_id=key_id
        )
        
        self.current_keypair = keypair
        return keypair
    
    def _derive_encryption_key(self, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if not self.password:
            raise ValueError("Password required for key encryption")
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # AES-256
            salt=salt,
            iterations=100000
        )
        return kdf.derive(self.password.encode())
    
    def save_keypair(self, keypair: KeyPair, name: str = "current") -> Path:
        """
        Save key pair to encrypted storage.
        
        Args:
            keypair: The key pair to save
            name: Name for the key file
            
        Returns:
            Path to the saved key file
        """
        key_path = self.storage_path / f"{name}.key"
        
        # Serialize private key
        private_key_bytes = bytes(keypair.private_key)
        
        if self.password:
            # Encrypt with AES-256-GCM
            salt = os.urandom(16)
            nonce = os.urandom(12)
            encryption_key = self._derive_encryption_key(salt)
            
            aesgcm = AESGCM(encryption_key)
            ciphertext = aesgcm.encrypt(nonce, private_key_bytes, None)
            
            # Store encrypted key with metadata
            data = {
                'encrypted': True,
                'salt': salt.hex(),
                'nonce': nonce.hex(),
                'ciphertext': ciphertext.hex(),
                'metadata': keypair.to_dict()
            }
        else:
            # Development mode: store unencrypted (NOT for production!)
            data = {
                'encrypted': False,
                'private_key': private_key_bytes.hex(),
                'metadata': keypair.to_dict()
            }
        
        with open(key_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return key_path
    
    def load_keypair(self, name: str = "current") -> Optional[KeyPair]:
        """
        Load key pair from encrypted storage.
        
        Args:
            name: Name of the key file
            
        Returns:
            KeyPair or None if not found
        """
        key_path = self.storage_path / f"{name}.key"
        
        if not key_path.exists():
            return None
        
        with open(key_path, 'r') as f:
            data = json.load(f)
        
        if data.get('encrypted'):
            if not self.password:
                raise ValueError("Password required to load encrypted key")
            
            salt = bytes.fromhex(data['salt'])
            nonce = bytes.fromhex(data['nonce'])
            ciphertext = bytes.fromhex(data['ciphertext'])
            
            encryption_key = self._derive_encryption_key(salt)
            aesgcm = AESGCM(encryption_key)
            
            try:
                private_key_bytes = aesgcm.decrypt(nonce, ciphertext, None)
            except Exception as e:
                raise ValueError(f"Failed to decrypt key: {e}")
        else:
            private_key_bytes = bytes.fromhex(data['private_key'])
        
        # Reconstruct key pair
        signing_key = SigningKey(private_key_bytes)
        verify_key = signing_key.verify_key
        
        metadata = data.get('metadata', {})
        
        keypair = KeyPair(
            private_key=signing_key,
            public_key=verify_key,
            created_at=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat())),
            expires_at=datetime.fromisoformat(metadata['expires_at']) if metadata.get('expires_at') else None,
            key_id=metadata.get('key_id')
        )
        
        self.current_keypair = keypair
        self._key_cache[name] = keypair
        
        return keypair
    
    def get_or_create_keypair(self, name: str = "current") -> KeyPair:
        """
        Get existing key pair or create new one.
        
        Args:
            name: Name for the key
            
        Returns:
            KeyPair instance
        """
        # Try to load existing
        keypair = self.load_keypair(name)
        
        if keypair is None:
            # Generate new
            keypair = self.generate_keypair()
            self.save_keypair(keypair, name)
        elif keypair.expires_at and keypair.expires_at < datetime.utcnow():
            # Key expired, generate new
            print(f"Key '{name}' expired, generating new key pair")
            keypair = self.generate_keypair()
            self.save_keypair(keypair, name)
        
        return keypair
    
    def rotate_keys(self) -> KeyPair:
        """
        Rotate keys by generating new pair and archiving old.
        
        Returns:
            New KeyPair
        """
        if self.current_keypair:
            # Archive old key
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.save_keypair(self.current_keypair, f"archived_{timestamp}")
        
        # Generate new
        new_keypair = self.generate_keypair()
        self.save_keypair(new_keypair, "current")
        
        return new_keypair
    
    def get_public_key_hex(self) -> str:
        """Get current public key as hex string."""
        if not self.current_keypair:
            self.get_or_create_keypair()
        return self.current_keypair.public_key.encode(encoder=HexEncoder).decode()
    
    def list_keys(self) -> list:
        """List all stored keys."""
        keys = []
        for key_file in self.storage_path.glob("*.key"):
            try:
                with open(key_file, 'r') as f:
                    data = json.load(f)
                metadata = data.get('metadata', {})
                keys.append({
                    'name': key_file.stem,
                    'key_id': metadata.get('key_id'),
                    'created_at': metadata.get('created_at'),
                    'expires_at': metadata.get('expires_at'),
                    'encrypted': data.get('encrypted', False)
                })
            except Exception:
                continue
        return keys
