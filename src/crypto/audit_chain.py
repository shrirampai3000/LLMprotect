"""
Simple Hash-Chain Audit Log.

Provides tamper-evident logging using sequential hash chaining
(not blockchain-based, just cryptographic integrity).
"""
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: int
    prompt_hash: str
    action: str
    decision: str
    risk_score: float
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_hash: str = ""
    previous_hash: str = ""
    
    def compute_hash(self) -> str:
        """Compute hash of this entry (excluding entry_hash)."""
        data = {
            'timestamp': self.timestamp,
            'prompt_hash': self.prompt_hash,
            'action': self.action,
            'decision': self.decision,
            'risk_score': self.risk_score,
            'signature': self.signature,
            'previous_hash': self.previous_hash
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class AuditLog:
    """
    Simple hash-chain audit log for tamper-evident logging.
    
    Each entry contains a hash of the previous entry, creating
    a chain where any modification breaks the chain integrity.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize audit log.
        
        Args:
            storage_path: Optional path for persistent storage
        """
        self.storage_path = storage_path
        self.entries: List[AuditEntry] = []
        
        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_entries()
    
    def add_entry(
        self,
        prompt_hash: str,
        action: str,
        decision: str,
        risk_score: float,
        signature: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Add a new entry to the audit log.
        
        Args:
            prompt_hash: Hash of the prompt
            action: Action type
            decision: Decision made
            risk_score: Risk score
            signature: Optional signature
            metadata: Optional additional metadata
            
        Returns:
            The created AuditEntry
        """
        # Get previous hash
        if self.entries:
            previous_hash = self.entries[-1].entry_hash
        else:
            previous_hash = "0" * 64  # Genesis hash
        
        entry = AuditEntry(
            timestamp=int(time.time()),
            prompt_hash=prompt_hash,
            action=action,
            decision=decision,
            risk_score=risk_score,
            signature=signature,
            metadata=metadata or {},
            previous_hash=previous_hash
        )
        
        # Compute entry hash
        entry.entry_hash = entry.compute_hash()
        
        self.entries.append(entry)
        
        # Persist if storage configured
        if self.storage_path:
            self._save_entry(entry)
        
        return entry
    
    def verify_integrity(self) -> Dict:
        """
        Verify the integrity of the entire audit log.
        
        Returns:
            Dict with 'valid' boolean and 'message' string
        """
        if not self.entries:
            return {'valid': True, 'message': 'Audit log is empty'}
        
        # Check first entry has genesis hash
        if self.entries[0].previous_hash != "0" * 64:
            return {
                'valid': False,
                'message': 'First entry does not have genesis hash'
            }
        
        # Check each entry's hash and chain
        for i, entry in enumerate(self.entries):
            # Verify entry hash
            computed_hash = entry.compute_hash()
            if computed_hash != entry.entry_hash:
                return {
                    'valid': False,
                    'message': f'Entry {i} hash mismatch - possible tampering'
                }
            
            # Verify chain (except first entry)
            if i > 0:
                if entry.previous_hash != self.entries[i-1].entry_hash:
                    return {
                        'valid': False,
                        'message': f'Chain broken at entry {i}'
                    }
        
        return {'valid': True, 'message': 'Audit log integrity verified'}
    
    def get_entries(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        decision: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """
        Query audit log entries with filters.
        
        Args:
            start_time: Start timestamp filter
            end_time: End timestamp filter
            decision: Decision filter
            limit: Maximum entries to return
            
        Returns:
            List of matching entries
        """
        results = []
        
        for entry in reversed(self.entries):
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if decision and entry.decision != decision:
                continue
            
            results.append(entry)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get audit log statistics."""
        if not self.entries:
            return {
                'total_entries': 0,
                'decisions': {},
                'avg_risk_score': 0.0,
                'latest_hash': None
            }
        
        decisions = {}
        total_risk = 0.0
        
        for entry in self.entries:
            decisions[entry.decision] = decisions.get(entry.decision, 0) + 1
            total_risk += entry.risk_score
        
        return {
            'total_entries': len(self.entries),
            'decisions': decisions,
            'avg_risk_score': total_risk / len(self.entries),
            'latest_hash': self.entries[-1].entry_hash
        }
    
    def _save_entry(self, entry: AuditEntry):
        """Save entry to persistent storage."""
        if not self.storage_path:
            return
        
        log_file = self.storage_path / "audit_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
    
    def _load_entries(self):
        """Load entries from persistent storage."""
        if not self.storage_path:
            return
        
        log_file = self.storage_path / "audit_log.jsonl"
        if not log_file.exists():
            return
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entry = AuditEntry(**data)
                    self.entries.append(entry)
    
    def export_log(self, filepath: Path):
        """Export audit log to JSON file."""
        data = {
            'entries': [asdict(e) for e in self.entries],
            'integrity': self.verify_integrity(),
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
