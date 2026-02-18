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

## Command-Line Arguments

### Training Data

| Argument | Description | Default |
|----------|-------------|---------|
| `--urls` | One or more URLs to fetch training text from | Curated list (names.txt + Sherlock Holmes) |
| `--url-file` | Path to a text file containing one URL per line | None |
| `--no-cache` | Skip URL cache and re-download all pages | Cache enabled |

### Model Architecture

| Argument | Description | Default |
|----------|-------------|---------|
| `--n-embd` | Embedding dimension | 16 |
| `--n-head` | Number of attention heads | 4 |
| `--n-layer` | Number of transformer layers | 1 |
| `--block-size` | Maximum sequence length | 16 |

### Training Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--num-steps` | Number of training steps | 1000 |
| `--batch-size` | Batch size for PyTorch training (ignored in pure-Python mode) | 64 |
| `--tokenizer` | Tokenization level: `char`, `word`, or `bpe` | char |
| `--bpe-merges` | Number of BPE merge operations (only used with `--tokenizer bpe`) | 1000 |

### Execution Mode

| Argument | Description | Default |
|----------|-------------|---------|
| `--no-gpu` | Force pure-Python CPU mode even if PyTorch/CUDA is available | GPU enabled if available |
| `--interactive` | Prompt for parameters interactively before training | False |
| `--prompt-file` | Path to a text file with one test prompt per line (runs after training) | Interactive REPL |

### Examples

```bash
# Train on custom URLs with larger model
python3 microgpt-sh.py --urls 'https://example.com/text1.txt' 'https://example.com/text2.txt' \
  --n-embd 64 --n-head 8 --n-layer 3 --block-size 128 \
  --num-steps 5000 --tokenizer bpe --bpe-merges 2000

# Train from URL file with batch training
python3 microgpt-sh.py --url-file urls.txt --batch-size 32 --num-steps 10000

# Run test prompts after training (non-interactive)
python3 microgpt-sh.py --prompt-file test_prompts.txt --tokenizer word

# Force CPU mode (disable GPU)
python3 microgpt-sh.py --no-gpu --tokenizer char

# Interactive parameter configuration
python3 microgpt-sh.py --interactive
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
