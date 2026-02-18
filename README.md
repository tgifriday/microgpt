# microgpt

A minimal GPT (Generative Pre-trained Transformer) implementation based on Andrej Karpathy's work - an educational/research project that demonstrates how GPT models work from first principles.

## Project Overview

**microgpt** is a from-scratch implementation of a GPT-2 style language model with:
- Pure Python implementation (no dependencies required)
- Optional PyTorch GPU acceleration
- Multiple tokenizer options (character, word, BPE subword)
- Web scraping capabilities for training data

## Key Files

| File | Purpose |
|------|---------|
| `microgpt.py` | Core "atomic" implementation (~200 lines) with pure Python autograd engine |
| `microgpt-sh.py` | Extended production version (~870 lines) with GPU support, multiple tokenizers |
| `gpt_gpu.py` | PyTorch GPU-accelerated version |
| `USAGE.md` | Comprehensive 400+ line documentation |
| `GPU_EXPERIMENTS.md` | Guide for parallel GPU experiments |

## Architecture Highlights

**Model Design** (GPT-2 variant):
- RMSNorm instead of LayerNorm (no learnable parameters)
- ReLU instead of GeLU
- No biases in linear layers
- Causal masked attention for autoregressive generation

**Default Parameters**:
- Embedding dimension: 16
- Attention heads: 4
- Transformer layers: 1
- Sequence length: 16

**Tokenizers** (3 options):
- Character-level (vocab ~80)
- Word-level (vocab ~12,000)
- BPE subword (learnable merges)

## Dependencies

- **None required** - runs in pure Python
- **Optional**: `pip install torch` for GPU acceleration

## Usage

```bash
# Basic usage with character tokenizer
python3 microgpt-sh.py --tokenizer char

# BPE tokenizer with custom merges
python3 microgpt-sh.py --tokenizer bpe --bpe-merges 1000 \
  --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --prompt-file test_prompts.txt

# GPU-accelerated training
python3 gpt_gpu.py --tokenizer bpe --bpe-merges 1000
```

## Model Architecture

```
Input tokens → Token Embedding + Position Embedding
             → RMSNorm
             → [Transformer Block] × n_layer
             → Language Model Head
             → Logits over vocabulary
```

Each Transformer Block contains:
- Multi-head causal self-attention
- MLP with ReLU activation (4× hidden dimension)
- Residual connections

## Training

- Adam optimizer with learning rate decay
- Cross-entropy loss on next-token prediction
- Pure Python: single document per step
- PyTorch: batched training (64 docs/step)

## License

Educational/Research project based on Andrej Karpathy's work.
