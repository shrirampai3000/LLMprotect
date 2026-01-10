"""
Digital Signature and Authorization Token Management.

Implements the Intent Binding Protocol with:
- Ed25519 digital signatures
- Timestamp validation
- Nonce-based replay prevention
- Authorization token generation/verification
"""
import os
import json
import hashlib
import time
from typing import Optional, Dict, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError

from .keys import KeyManager


@dataclass
class AuthorizationToken:
    """
    Authorization token for intent binding.
    
    Structure:
    - prompt_hash: SHA-256 hash of normalized prompt
    - action: The action being authorized
    - target: Target resource (e.g., "mcp://database/query")
    - timestamp: Unix timestamp of creation
    - expires_at: Expiration timestamp
    - nonce: 32-byte random nonce (prevents replay)
    - signature: Ed25519 signature of the above
    """
    prompt_hash: str
    action: str
    target: str
    timestamp: int
    expires_at: int
    nonce: str
    signature: str
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AuthorizationToken':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if token has expired."""
        return int(time.time()) > self.expires_at


class NonceStore:
    """
    In-memory nonce store for replay prevention.
    
    In production, use Redis with TTL for distributed systems.
    """
    
    def __init__(self, ttl_seconds: int = 900):
        """
        Initialize nonce store.
        
        Args:
            ttl_seconds: Time-to-live for nonces (default: 15 minutes)
        """
        self.ttl = ttl_seconds
        self._nonces: Dict[str, int] = {}  # nonce -> expiry timestamp
    
    def add(self, nonce: str) -> bool:
        """
        Add a nonce to the store.
        
        Returns:
            True if added successfully, False if nonce already exists
        """
        self._cleanup()
        
        if nonce in self._nonces:
            return False
        
        self._nonces[nonce] = int(time.time()) + self.ttl
        return True
    
    def check(self, nonce: str) -> bool:
        """
        Check if nonce has been used.
        
        Returns:
            True if nonce is valid (not used), False if used/expired
        """
        self._cleanup()
        return nonce not in self._nonces
    
    def _cleanup(self):
        """Remove expired nonces."""
        current = int(time.time())
        expired = [n for n, exp in self._nonces.items() if exp < current]
        for n in expired:
            del self._nonces[n]


class AuthorizationManager:
    """
    Manages authorization token creation and verification.
    
    Implements the Intent Binding Protocol:
    1. Normalize prompt and compute hash
    2. Generate nonce and timestamp
    3. Sign the authorization payload
    4. Verify signatures and prevent replay attacks
    """
    
    def __init__(
        self,
        key_manager: KeyManager,
        token_expiry_seconds: int = 300,
        clock_skew_tolerance: int = 30
    ):
        """
        Initialize authorization manager.
        
        Args:
            key_manager: KeyManager instance with signing keys
            token_expiry_seconds: Token validity period (default: 5 minutes)
            clock_skew_tolerance: Allowed clock drift (default: 30 seconds)
        """
        self.key_manager = key_manager
        self.token_expiry = token_expiry_seconds
        self.clock_skew = clock_skew_tolerance
        self.nonce_store = NonceStore()
        
        # Ensure we have a key pair
        self.key_manager.get_or_create_keypair()
    
    @staticmethod
    def normalize_prompt(prompt: str) -> str:
        """
        Normalize prompt for consistent hashing.
        
        - Convert to lowercase
        - Strip leading/trailing whitespace
        - Collapse multiple spaces
        """
        normalized = prompt.lower().strip()
        normalized = ' '.join(normalized.split())
        return normalized
    
    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Compute SHA-256 hash of normalized prompt."""
        normalized = AuthorizationManager.normalize_prompt(prompt)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def create_authorization(
        self,
        prompt: str,
        action: str,
        target: str
    ) -> AuthorizationToken:
        """
        Create a signed authorization token.
        
        Args:
            prompt: The user prompt being authorized
            action: Action type (e.g., "execute_tool", "query_database")
            target: Target resource identifier
            
        Returns:
            Signed AuthorizationToken
        """
        # Get current key pair
        keypair = self.key_manager.current_keypair
        if not keypair:
            raise ValueError("No signing key available")
        
        # Generate token components
        prompt_hash = self.hash_prompt(prompt)
        timestamp = int(time.time())
        expires_at = timestamp + self.token_expiry
        nonce = os.urandom(32).hex()
        
        # Construct message to sign
        message = self._construct_message(
            prompt_hash, action, target, timestamp, nonce
        )
        
        # Sign with Ed25519
        signed = keypair.private_key.sign(message.encode('utf-8'))
        signature = signed.signature.hex()
        
        # Create token
        token = AuthorizationToken(
            prompt_hash=prompt_hash,
            action=action,
            target=target,
            timestamp=timestamp,
            expires_at=expires_at,
            nonce=nonce,
            signature=signature
        )
        
        # Note: Nonce is added to store ONLY when token is verified/used,
        # not on creation. This allows the same token to be verified once.
        
        return token
    
    def verify_authorization(
        self,
        token: AuthorizationToken,
        expected_prompt: Optional[str] = None
    ) -> Dict:
        """
        Verify an authorization token.
        
        Args:
            token: The authorization token to verify
            expected_prompt: Optional prompt to verify against
            
        Returns:
            Dict with 'valid' bool and 'error' message if invalid
        """
        try:
            # 1. Check timestamp validity
            current_time = int(time.time())
            
            if token.is_expired():
                return {'valid': False, 'error': 'Token has expired'}
            
            # Check clock skew
            if abs(current_time - token.timestamp) > self.clock_skew + self.token_expiry:
                return {'valid': False, 'error': 'Token timestamp out of acceptable range'}
            
            # 2. Check nonce (replay prevention)
            if not self.nonce_store.check(token.nonce):
                # Nonce has been used
                return {'valid': False, 'error': 'Nonce has already been used (replay attack detected)'}
            
            # 3. Verify prompt hash if expected prompt provided
            if expected_prompt:
                expected_hash = self.hash_prompt(expected_prompt)
                if token.prompt_hash != expected_hash:
                    return {'valid': False, 'error': 'Prompt hash mismatch'}
            
            # 4. Verify signature
            keypair = self.key_manager.current_keypair
            if not keypair:
                return {'valid': False, 'error': 'No verification key available'}
            
            message = self._construct_message(
                token.prompt_hash, token.action, token.target,
                token.timestamp, token.nonce
            )
            
            signature = bytes.fromhex(token.signature)
            
            try:
                keypair.public_key.verify(
                    message.encode('utf-8'),
                    signature
                )
            except BadSignatureError:
                return {'valid': False, 'error': 'Invalid signature'}
            
            # 5. Mark nonce as used
            self.nonce_store.add(token.nonce)
            
            return {
                'valid': True,
                'prompt_hash': token.prompt_hash,
                'action': token.action,
                'target': token.target,
                'expires_in': token.expires_at - current_time
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Verification failed: {str(e)}'}
    
    def _construct_message(
        self,
        prompt_hash: str,
        action: str,
        target: str,
        timestamp: int,
        nonce: str
    ) -> str:
        """Construct the message to be signed."""
        # Concatenate fields with delimiters
        return f"{prompt_hash}|{action}|{target}|{timestamp}|{nonce}"
    
    def revoke_token(self, token: AuthorizationToken):
        """
        Revoke a token by marking its nonce as used.
        
        This prevents future use of the token.
        """
        self.nonce_store.add(token.nonce)
    
    def get_stats(self) -> Dict:
        """Get authorization statistics."""
        return {
            'active_nonces': len(self.nonce_store._nonces),
            'key_id': self.key_manager.current_keypair.key_id if self.key_manager.current_keypair else None,
            'token_expiry_seconds': self.token_expiry,
            'clock_skew_tolerance': self.clock_skew
        }
