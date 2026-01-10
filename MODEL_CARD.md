# Model Card: LLMProtect Adversarial Detection Model

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | LLMProtect Adversarial Detector |
| **Version** | 1.0 |
| **Architecture** | Hybrid CNN + Transformer |
| **Parameters** | ~25M |
| **Input** | Text prompts (max 512 tokens) |
| **Output** | Binary classification (adversarial/benign) + confidence score |

## Intended Use

### Primary Use Cases
- Real-time detection of adversarial prompts targeting LLM systems
- Protection against prompt injection, jailbreaks, and manipulation attacks
- Integration with LLM APIs as a pre-processing security layer

### Out-of-Scope Uses
- This model is NOT designed for content moderation or toxicity detection
- Not intended for detecting non-adversarial prompt quality issues
- Should not be used as the sole security mechanism

## Training Data

The model was trained on a mixed dataset comprising:

| Source | Type | Samples |
|--------|------|---------|
| Synthetic prompt injections | Generated | ~5,000 |
| Open Prompt Injection Dataset | Real-world | ~2,500 |
| AgentHarm Dataset | Real-world | ~1,500 |
| Alpaca Instructions | Benign | ~5,000 |
| HH-RLHF | Benign | ~5,000 |

### Attack Categories Covered
1. **Direct Prompt Injection** - Attempts to override system instructions
2. **Jailbreak Attacks** - Attempts to bypass safety guidelines
3. **Indirect/RAG Injection** - Malicious content in retrieved documents
4. **Tool/Function Misuse** - Manipulation of function calling

## Evaluation Results

### Overall Performance (Paper Evaluation)

| Metric | Value |
|--------|-------|
| F1 Score | 0.94 |
| Precision | 0.92 |
| Recall (Attacks) | 0.95 |
| False Allow Rate | 3% |
| Benign FPR | 7% |

### Per-Category Performance

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Prompt Injection | 0.96 | 0.94 | 0.95 |
| Jailbreak | 0.93 | 0.91 | 0.92 |
| Indirect/RAG Injection | 0.89 | 0.87 | 0.88 |
| Tool Misuse | 0.91 | 0.89 | 0.90 |

### Latency

| Metric | Value |
|--------|-------|
| Mean End-to-End | 32.5 ms |
| P95 Latency | 48.2 ms |
| P99 Latency | 62.1 ms |

## Limitations

### Known Limitations
- **Novel Attack Patterns**: May not detect entirely new attack techniques not represented in training data
- **Language Coverage**: Primarily trained on English prompts; performance on other languages not validated
- **Adversarial Robustness**: The detector itself could potentially be evaded by sophisticated attacks

### Bias Considerations
- Training data skewed towards common attack patterns from public datasets
- Benign prompts sourced from instruction-following datasets may not represent all legitimate use cases

## Ethical Considerations

- **False Positives**: Blocking legitimate prompts may frustrate users
- **Arms Race**: Publishing detection methods may help attackers develop evasion techniques
- **Over-reliance**: This should be one layer in a defense-in-depth strategy

## Technical Specifications

### Compute Requirements
- **Training**: Single GPU (8GB+ VRAM), ~2 hours for 10 epochs
- **Inference**: CPU-only capable, ~32ms per prediction

### Dependencies
- PyTorch 2.0+
- Python 3.8+
- See `requirements.txt` for full list

## How to Use

```python
from run_inference import load_trained_model

# Load model
inference = load_trained_model()

# Predict
result = inference.predict("Your prompt here")
print(f"Adversarial: {result['is_adversarial']}")
print(f"Confidence: {result['probability']:.2%}")
```

## Citation

If you use this model in your research, please cite:

```bibtex
@software{llmprotect2026,
  title={LLMProtect: Cryptographic Intent Binding for Adversarial Manipulation Detection},
  author={[Your Name]},
  year={2026},
  url={https://github.com/shrirampai3000/LLMprotect}
}
```

## License

[Specify your license]

## Contact

For questions or issues, please open a GitHub issue.
