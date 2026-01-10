"""Tests for cryptographic components."""
import pytest
import time
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.crypto.keys import KeyManager, KeyPair
from src.crypto.signing import AuthorizationManager, AuthorizationToken, NonceStore
from src.crypto.audit_chain import AuditLog


class TestKeyManager:
    """Tests for Ed25519 key management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.key_manager = KeyManager(self.temp_dir, password="test_password_123")
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_keypair(self):
        """Test key pair generation."""
        keypair = self.key_manager.generate_keypair()
        
        assert keypair is not None
        assert keypair.private_key is not None
        assert keypair.public_key is not None
        assert keypair.key_id is not None
        assert keypair.created_at is not None
        assert keypair.expires_at is not None
    
    def test_save_and_load_keypair(self):
        """Test encrypted key storage and retrieval."""
        # Generate and save
        original = self.key_manager.generate_keypair()
        self.key_manager.save_keypair(original, "test_key")
        
        # Load and compare
        loaded = self.key_manager.load_keypair("test_key")
        
        assert loaded is not None
        assert loaded.key_id == original.key_id
        assert bytes(loaded.public_key) == bytes(original.public_key)
    
    def test_get_or_create_keypair(self):
        """Test get_or_create_keypair creates new key if none exists."""
        keypair1 = self.key_manager.get_or_create_keypair("new_key")
        keypair2 = self.key_manager.get_or_create_keypair("new_key")
        
        assert keypair1.key_id == keypair2.key_id
    
    def test_key_rotation(self):
        """Test key rotation."""
        original = self.key_manager.get_or_create_keypair()
        original_id = original.key_id
        
        new_key = self.key_manager.rotate_keys()
        
        assert new_key.key_id != original_id
    
    def test_list_keys(self):
        """Test listing stored keys."""
        self.key_manager.generate_keypair("key1")
        self.key_manager.save_keypair(self.key_manager.current_keypair, "key1")
        
        self.key_manager.generate_keypair("key2")
        self.key_manager.save_keypair(self.key_manager.current_keypair, "key2")
        
        keys = self.key_manager.list_keys()
        assert len(keys) >= 2


class TestAuthorizationManager:
    """Tests for authorization token creation and verification."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.key_manager = KeyManager(self.temp_dir)
        self.auth_manager = AuthorizationManager(self.key_manager)
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_authorization(self):
        """Test authorization token creation."""
        token = self.auth_manager.create_authorization(
            prompt="Hello, world!",
            action="execute",
            target="test://resource"
        )
        
        assert token is not None
        assert token.prompt_hash is not None
        assert token.action == "execute"
        assert token.target == "test://resource"
        assert token.signature is not None
        assert len(token.nonce) == 64  # 32 bytes as hex
    
    def test_verify_valid_authorization(self):
        """Test verification of valid token."""
        prompt = "Test prompt"
        token = self.auth_manager.create_authorization(
            prompt=prompt,
            action="execute",
            target="test://resource"
        )
        
        # Create fresh auth manager to test verification
        result = self.auth_manager.verify_authorization(token, prompt)
        
        assert result['valid'] == True
    
    def test_reject_tampered_signature(self):
        """Test that tampered signatures are rejected."""
        token = self.auth_manager.create_authorization(
            prompt="Test",
            action="execute",
            target="test://resource"
        )
        
        # Tamper with signature
        tampered_token = AuthorizationToken(
            prompt_hash=token.prompt_hash,
            action=token.action,
            target=token.target,
            timestamp=token.timestamp,
            expires_at=token.expires_at,
            nonce=token.nonce,
            signature="00" * 64  # Invalid signature
        )
        
        result = self.auth_manager.verify_authorization(tampered_token)
        assert result['valid'] == False
    
    def test_reject_expired_token(self):
        """Test that expired tokens are rejected."""
        # Create authorization manager with very short expiry
        short_auth = AuthorizationManager(self.key_manager, token_expiry_seconds=1)
        
        token = short_auth.create_authorization(
            prompt="Test",
            action="execute",
            target="test://resource"
        )
        
        # Wait for expiry
        time.sleep(2)
        
        result = short_auth.verify_authorization(token)
        assert result['valid'] == False
        assert 'expired' in result.get('error', '').lower()
    
    def test_prompt_hash_consistency(self):
        """Test that prompt hashing is consistent."""
        prompt = "  Hello   World  "
        
        hash1 = AuthorizationManager.hash_prompt(prompt)
        hash2 = AuthorizationManager.hash_prompt(prompt)
        
        assert hash1 == hash2
        
        # Slightly different should hash same (normalization)
        hash3 = AuthorizationManager.hash_prompt("hello world")
        assert hash1 == hash3


class TestNonceStore:
    """Tests for nonce-based replay prevention."""
    
    def test_nonce_uniqueness(self):
        """Test that used nonces are rejected."""
        store = NonceStore(ttl_seconds=60)
        
        nonce = "test_nonce_123"
        
        # First use should succeed
        assert store.add(nonce) == True
        
        # Second use should fail
        assert store.add(nonce) == False
    
    def test_nonce_expiry(self):
        """Test that expired nonces are cleaned up."""
        store = NonceStore(ttl_seconds=1)
        
        nonce = "test_nonce_456"
        store.add(nonce)
        
        # Wait for expiry
        time.sleep(2)
        
        # Should be able to add again after cleanup
        assert store.check(nonce) == True


class TestAuditLog:
    """Tests for hash-chain audit logging."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.audit_log = AuditLog(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_entry(self):
        """Test adding audit entries."""
        entry = self.audit_log.add_entry(
            prompt_hash="test_hash",
            action="execute",
            decision="approved",
            risk_score=0.1,
            signature="test_sig"
        )
        
        assert entry is not None
        assert entry.prompt_hash == "test_hash"
        assert entry.decision == "approved"
        assert entry.entry_hash != ""
    
    def test_hash_chain(self):
        """Test that entries are chained by hash."""
        entry1 = self.audit_log.add_entry(
            prompt_hash="hash_1",
            action="execute",
            decision="approved",
            risk_score=0.1
        )
        
        entry2 = self.audit_log.add_entry(
            prompt_hash="hash_2",
            action="execute",
            decision="denied",
            risk_score=0.8
        )
        
        # Second entry should reference first entry's hash
        assert entry2.previous_hash == entry1.entry_hash
    
    def test_integrity_verification(self):
        """Test log integrity verification."""
        # Add some entries
        for i in range(5):
            self.audit_log.add_entry(
                prompt_hash=f"hash_{i}",
                action="execute",
                decision="approved" if i % 2 == 0 else "denied",
                risk_score=0.1 * i
            )
        
        # Verify integrity
        result = self.audit_log.verify_integrity()
        assert result['valid'] == True
    
    def test_tamper_detection(self):
        """Test that tampering is detected."""
        # Add entries
        for i in range(3):
            self.audit_log.add_entry(
                prompt_hash=f"hash_{i}",
                action="execute",
                decision="approved",
                risk_score=0.1
            )
        
        # Tamper with middle entry
        self.audit_log.entries[1].decision = "TAMPERED"
        
        # Should detect tampering
        result = self.audit_log.verify_integrity()
        assert result['valid'] == False
    
    def test_query_entries(self):
        """Test querying entries with filters."""
        # Add entries with different decisions
        for i in range(5):
            self.audit_log.add_entry(
                prompt_hash=f"hash_{i}",
                action="execute",
                decision="approved" if i < 3 else "denied",
                risk_score=0.1
            )
        
        approved = self.audit_log.get_entries(decision="approved")
        assert len(approved) == 3
        
        denied = self.audit_log.get_entries(decision="denied")
        assert len(denied) == 2
    
    def test_statistics(self):
        """Test audit statistics."""
        for i in range(3):
            self.audit_log.add_entry(
                prompt_hash=f"hash_{i}",
                action="execute",
                decision="approved",
                risk_score=0.2
            )
        
        stats = self.audit_log.get_statistics()
        assert stats['total_entries'] == 3
        assert stats['decisions']['approved'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
