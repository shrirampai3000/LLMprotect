"""
Scoped Credential Management.

Implements least-privilege credential assignment with:
- Per-action credential scoping
- Dynamic rotation (every 15 minutes)
- Automatic revocation on suspicious activity
"""
import os
import time
import hashlib
from typing import Dict, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock


@dataclass
class ScopedCredential:
    """
    Credential with limited scope and expiration.
    """
    credential_id: str
    token: str
    action_types: Set[str]
    resources: Set[str]
    created_at: datetime
    expires_at: datetime
    max_uses: int
    current_uses: int = 0
    is_revoked: bool = False
    
    def is_valid(self) -> bool:
        """Check if credential is still valid."""
        if self.is_revoked:
            return False
        if datetime.utcnow() > self.expires_at:
            return False
        if self.current_uses >= self.max_uses:
            return False
        return True
    
    def can_access(self, action: str, resource: str) -> bool:
        """Check if credential can access given action/resource."""
        if not self.is_valid():
            return False
        
        # Check action type
        if '*' not in self.action_types and action not in self.action_types:
            return False
        
        # Check resource
        if '*' not in self.resources and resource not in self.resources:
            # Check prefix match
            if not any(resource.startswith(r) for r in self.resources):
                return False
        
        return True
    
    def use(self):
        """Record a use of this credential."""
        self.current_uses += 1


class CredentialManager:
    """
    Manages scoped credentials for the Intent Binding system.
    
    Features:
    - Least-privilege credential assignment
    - Time-limited tokens (default: 15 minutes)
    - Use-limited tokens (default: 100 uses)
    - Automatic rotation
    - Revocation on suspicious activity
    """
    
    def __init__(
        self,
        rotation_minutes: int = 15,
        max_uses_per_credential: int = 100
    ):
        """
        Initialize credential manager.
        
        Args:
            rotation_minutes: Credential validity period
            max_uses_per_credential: Maximum uses per credential
        """
        self.rotation_minutes = rotation_minutes
        self.max_uses = max_uses_per_credential
        
        self._credentials: Dict[str, ScopedCredential] = {}
        self._lock = Lock()
        
        # Predefined action types
        self.ACTION_TYPES = {
            'read': 'Read-only operations',
            'write': 'Write operations',
            'execute': 'Command execution',
            'query': 'Database queries',
            'admin': 'Administrative operations'
        }
    
    def create_credential(
        self,
        action_types: List[str],
        resources: List[str],
        validity_minutes: Optional[int] = None,
        max_uses: Optional[int] = None
    ) -> ScopedCredential:
        """
        Create a new scoped credential.
        
        Args:
            action_types: Allowed action types
            resources: Allowed resources (can use wildcards)
            validity_minutes: Override default validity
            max_uses: Override default max uses
            
        Returns:
            New ScopedCredential
        """
        with self._lock:
            # Generate credential
            credential_id = hashlib.sha256(
                os.urandom(32) + str(time.time()).encode()
            ).hexdigest()[:24]
            
            token = os.urandom(32).hex()
            
            now = datetime.utcnow()
            validity = validity_minutes or self.rotation_minutes
            expires = now + timedelta(minutes=validity)
            
            credential = ScopedCredential(
                credential_id=credential_id,
                token=token,
                action_types=set(action_types),
                resources=set(resources),
                created_at=now,
                expires_at=expires,
                max_uses=max_uses or self.max_uses
            )
            
            self._credentials[credential_id] = credential
            return credential
    
    def validate_credential(
        self,
        credential_id: str,
        token: str,
        action: str,
        resource: str
    ) -> Dict:
        """
        Validate a credential for a specific action/resource.
        
        Args:
            credential_id: The credential ID
            token: The credential token
            action: Action being performed
            resource: Target resource
            
        Returns:
            Validation result dictionary
        """
        with self._lock:
            if credential_id not in self._credentials:
                return {'valid': False, 'error': 'Credential not found'}
            
            credential = self._credentials[credential_id]
            
            # Verify token
            if credential.token != token:
                return {'valid': False, 'error': 'Invalid token'}
            
            # Check validity
            if not credential.is_valid():
                if credential.is_revoked:
                    return {'valid': False, 'error': 'Credential has been revoked'}
                if datetime.utcnow() > credential.expires_at:
                    return {'valid': False, 'error': 'Credential has expired'}
                if credential.current_uses >= credential.max_uses:
                    return {'valid': False, 'error': 'Credential usage limit exceeded'}
            
            # Check access
            if not credential.can_access(action, resource):
                return {
                    'valid': False,
                    'error': f'Credential not authorized for action={action}, resource={resource}'
                }
            
            # Record use
            credential.use()
            
            return {
                'valid': True,
                'credential_id': credential_id,
                'remaining_uses': credential.max_uses - credential.current_uses,
                'expires_in_seconds': int((credential.expires_at - datetime.utcnow()).total_seconds())
            }
    
    def revoke_credential(self, credential_id: str, reason: str = "") -> bool:
        """
        Revoke a credential immediately.
        
        Args:
            credential_id: The credential to revoke
            reason: Reason for revocation
            
        Returns:
            True if revoked successfully
        """
        with self._lock:
            if credential_id in self._credentials:
                self._credentials[credential_id].is_revoked = True
                return True
            return False
    
    def revoke_all(self):
        """Revoke all credentials (emergency use)."""
        with self._lock:
            for credential in self._credentials.values():
                credential.is_revoked = True
    
    def cleanup_expired(self) -> int:
        """
        Remove expired credentials from memory.
        
        Returns:
            Number of credentials removed
        """
        with self._lock:
            now = datetime.utcnow()
            expired_ids = [
                cid for cid, cred in self._credentials.items()
                if cred.expires_at < now
            ]
            
            for cid in expired_ids:
                del self._credentials[cid]
            
            return len(expired_ids)
    
    def get_active_credentials(self) -> List[Dict]:
        """Get list of active credentials (metadata only)."""
        with self._lock:
            now = datetime.utcnow()
            return [
                {
                    'credential_id': cred.credential_id,
                    'action_types': list(cred.action_types),
                    'resources': list(cred.resources),
                    'expires_in_seconds': int((cred.expires_at - now).total_seconds()),
                    'uses_remaining': cred.max_uses - cred.current_uses,
                    'is_revoked': cred.is_revoked
                }
                for cred in self._credentials.values()
                if cred.is_valid()
            ]
    
    def get_statistics(self) -> Dict:
        """Get credential manager statistics."""
        with self._lock:
            total = len(self._credentials)
            active = sum(1 for c in self._credentials.values() if c.is_valid())
            revoked = sum(1 for c in self._credentials.values() if c.is_revoked)
            expired = sum(
                1 for c in self._credentials.values()
                if not c.is_revoked and datetime.utcnow() > c.expires_at
            )
            
            return {
                'total_credentials': total,
                'active': active,
                'revoked': revoked,
                'expired': expired,
                'max_uses_per_credential': self.max_uses,
                'rotation_minutes': self.rotation_minutes
            }
