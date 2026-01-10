# Cryptographic Keys

This directory contains cryptographic keys used for intent binding signatures.

## Key Generation

Keys are auto-generated on first run. To manually regenerate:

```python
from src.crypto.keys import KeyManager

km = KeyManager()
km.generate_key_pair()
km.save_keys("keys/")
```

## Security Notice

⚠️ **NEVER commit actual key files to version control!**

The `*.key` files are excluded via `.gitignore`. Each deployment should generate its own unique keys.

## Key Types

- **Ed25519 Private Key**: Used for signing intent bindings
- **Ed25519 Public Key**: Used for verification

## First-Time Setup

When you clone this repository, keys will be automatically generated when you first run:
- `python demo.py`
- `python run_inference.py`
- The API server (`python -m src.api.server`)
