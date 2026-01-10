"""
Unified Intent Binding Pipeline.

Integrates ML detection with cryptographic authorization
for complete adversarial manipulation protection.
"""
import time
from typing import Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass

from ..models.inference import InferencePipeline
from ..crypto.keys import KeyManager
from ..crypto.signing import AuthorizationManager, AuthorizationToken
from ..crypto.audit_chain import AuditLog


@dataclass
class ProcessingResult:
    """Result of processing a prompt through the pipeline."""
    prompt: str
    is_adversarial: bool
    risk_score: float
    decision: str  # "approved", "denied", "requires_authorization"
    authorization_token: Optional[AuthorizationToken] = None
    explanation: Optional[str] = None
    processing_time_ms: float = 0.0
    audit_entry_id: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'prompt': self.prompt[:100] + '...' if len(self.prompt) > 100 else self.prompt,
            'is_adversarial': self.is_adversarial,
            'risk_score': self.risk_score,
            'decision': self.decision,
            'has_authorization': self.authorization_token is not None,
            'explanation': self.explanation,
            'processing_time_ms': self.processing_time_ms
        }


class IntentBindingPipeline:
    """
    Complete pipeline for adversarial detection and cryptographic authorization.
    
    Multi-layer security:
    1. ML Detection: Analyze prompt for adversarial patterns
    2. Risk Assessment: Score prompt risk level
    3. Cryptographic Authorization: Sign approved actions
    4. Audit Logging: Record all decisions in tamper-proof log
    """
    
    def __init__(
        self,
        ml_pipeline: Optional[InferencePipeline] = None,
        key_manager: Optional[KeyManager] = None,
        audit_log_path: Optional[Path] = None,
        adversarial_threshold: float = 0.5,
        auto_deny_threshold: float = 0.9
    ):
        """
        Initialize the Intent Binding Pipeline.
        
        Args:
            ml_pipeline: ML inference pipeline (optional for crypto-only mode)
            key_manager: Cryptographic key manager
            audit_log_path: Path for audit logs
            adversarial_threshold: Threshold for adversarial classification
            auto_deny_threshold: Auto-deny if risk score exceeds this
        """
        self.ml_pipeline = ml_pipeline
        self.adversarial_threshold = adversarial_threshold
        self.auto_deny_threshold = auto_deny_threshold
        
        # Initialize cryptographic components
        if key_manager is None:
            key_manager = KeyManager(Path("keys"))
        self.key_manager = key_manager
        self.auth_manager = AuthorizationManager(key_manager)
        
        # Initialize audit log
        audit_path = audit_log_path or Path("audit_logs")
        self.audit_log = AuditLog(audit_path)
    
    def process(
        self,
        prompt: str,
        action: str = "execute",
        target: str = "default",
        force_authorization: bool = False
    ) -> ProcessingResult:
        """
        Process a prompt through the complete pipeline.
        
        Args:
            prompt: User input prompt
            action: Action type being requested
            target: Target resource
            force_authorization: Force crypto authorization even if benign
            
        Returns:
            ProcessingResult with decision and optional authorization
        """
        start_time = time.time()
        
        # Step 1: ML Detection
        if self.ml_pipeline:
            detection_result = self.ml_pipeline.predict(prompt)
            risk_score = detection_result['probability']
            is_adversarial = detection_result['is_adversarial']
        else:
            # No ML model - use simple heuristic for demo
            risk_score = self._simple_heuristic(prompt)
            is_adversarial = risk_score > self.adversarial_threshold
        
        # Step 2: Decision Logic
        if risk_score >= self.auto_deny_threshold:
            # High-risk: Auto-deny
            decision = "denied"
            explanation = f"Prompt classified as high-risk adversarial (score: {risk_score:.3f})"
            authorization_token = None
            
        elif is_adversarial:
            # Medium risk: Requires explicit authorization
            decision = "requires_authorization"
            explanation = f"Prompt flagged as potentially adversarial (score: {risk_score:.3f}). Manual authorization required."
            authorization_token = None
            
        else:
            # Low risk: Approved with crypto binding
            decision = "approved"
            explanation = "Prompt passed adversarial detection"
            
            # Generate authorization token
            authorization_token = self.auth_manager.create_authorization(
                prompt=prompt,
                action=action,
                target=target
            )
        
        # Force authorization if requested (for testing)
        if force_authorization and authorization_token is None and decision != "denied":
            authorization_token = self.auth_manager.create_authorization(
                prompt=prompt,
                action=action,
                target=target
            )
            decision = "approved_with_override"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Step 3: Audit Logging
        prompt_hash = AuthorizationManager.hash_prompt(prompt)
        audit_entry = self.audit_log.add_entry(
            prompt_hash=prompt_hash,
            action=action,
            decision=decision,
            risk_score=risk_score,
            signature=authorization_token.signature if authorization_token else None,
            metadata={
                'target': target,
                'processing_time_ms': processing_time
            }
        )
        
        return ProcessingResult(
            prompt=prompt,
            is_adversarial=is_adversarial,
            risk_score=risk_score,
            decision=decision,
            authorization_token=authorization_token,
            explanation=explanation,
            processing_time_ms=processing_time,
            audit_entry_id=len(self.audit_log.entries) - 1
        )
    
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
            Verification result dictionary
        """
        return self.auth_manager.verify_authorization(token, expected_prompt)
    
    def _simple_heuristic(self, prompt: str) -> float:
        """
        Simple heuristic-based detection (fallback without ML model).
        
        Checks for common adversarial patterns.
        """
        prompt_lower = prompt.lower()
        
        # High-risk patterns
        high_risk_patterns = [
            "ignore previous",
            "ignore all",
            "ignore your",
            "ignore my",
            "ignore the",
            "disregard",
            "forget your instructions",
            "forget your",
            "system prompt",
            "jailbreak",
            "jail break",
            "developer mode",
            "dan mode",
            "bypass",
            "override",
            "pretend you",
            "act as if",
            "you are now",
            "new instructions",
        ]
        
        # Medium-risk patterns
        medium_risk_patterns = [
            "hypothetically",
            "imagine you",
            "roleplay",
            "let's play",
            "in a story",
            "creative writing",
            "for educational",
            "for research",
        ]
        
        risk_score = 0.0
        
        for pattern in high_risk_patterns:
            if pattern in prompt_lower:
                risk_score += 0.3
        
        for pattern in medium_risk_patterns:
            if pattern in prompt_lower:
                risk_score += 0.15
        
        # Check for encoding patterns
        if any(c in prompt for c in ['\\x', '\\u', 'base64', '&#']):
            risk_score += 0.2
        
        # Cap at 1.0
        return min(risk_score, 1.0)
    
    def get_audit_summary(self) -> Dict:
        """Get summary of audit log."""
        integrity = self.audit_log.verify_integrity()
        stats = self.audit_log.get_statistics()
        
        return {
            'integrity': integrity,
            'statistics': stats
        }
    
    def process_batch(
        self,
        prompts: List[str],
        action: str = "execute",
        target: str = "default"
    ) -> List[ProcessingResult]:
        """Process multiple prompts."""
        return [
            self.process(prompt, action, target)
            for prompt in prompts
        ]
