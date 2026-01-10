"""Core module initialization."""
from .pipeline import IntentBindingPipeline
from .credentials import CredentialManager, ScopedCredential

__all__ = [
    "IntentBindingPipeline",
    "CredentialManager",
    "ScopedCredential",
]
