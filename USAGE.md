# microgpt-sh.py — Usage Guide

A self-contained GPT language model that trains on text from web pages and generates
new text in the style of its training data. Supports character-level, word-level, and
BPE subword tokenization, with automatic GPU acceleration when available.

Based on Karpathy's minimal GPT implementation, extended for multi-page ingestion,
configurable parameters, and GPU support with pure-Python fallback.

---

## Quick Start

```bash
# Simplest possible run (uses default URLs and parameters)
python3 microgpt-sh.py

# Train on the Bible with word-level tokens
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32
```

After training completes, the model generates 20 random text samples, then drops
into an interactive prompt where you can type text prefixes and the model will
continue them.

---

## Tokenizer Progression: Simple → Intermediate → Advanced

The three tokenizer modes represent increasing levels of sophistication.
Run these in order to see the progression.

### 1. Simple: Character Level

Each letter is a token. The model learns character patterns and common syllables.

```bash
python3 microgpt-sh.py \
  --tokenizer char \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 256 --n-embd 64 --n-head 8 --n-layer 2 \
  --num-steps 5000 --batch-size 32
```

- **Vocab**: ~80 tokens (letters + punctuation)
- **Context**: 256 characters ≈ 40–50 words
- **Output**: Word-like fragments, occasionally recognizable phrases
- **Training**: Fast — tiny vocab means small embedding table

### 2. Intermediate: Word Level

Each word is a token. The model learns grammar and phrase structure directly.

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32
```

- **Vocab**: ~12,000 tokens (every unique word in the Bible)
- **Context**: 128 words ≈ half a page
- **Output**: Coherent sentences in the style of the training text
- **Trade-off**: Words not in the training data are unknown (e.g., "honoring" vs "honouring")

### 3. Advanced: BPE Subword

Byte Pair Encoding learns common subword units. Handles unseen words by splitting
them into known parts (e.g., "un" + "right" + "eous" + "ness").

```bash
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 1000 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32
```

- **Vocab**: ~1,080 tokens (79 base chars + 1,000 learned merges)
- **Context**: 128 subword tokens ≈ 60–100 words
- **Output**: Flexible — no out-of-vocabulary problem, blends char-level robustness with word-level efficiency
- **Note**: First run trains BPE merges (~2 min), cached for future runs

### Side-by-Side Comparison

| | Char | Word | BPE (1K merges) |
|---|---|---|---|
| "And God said" | 13 tokens | 3 tokens | ~4–5 tokens |
| Vocab size (Bible) | ~80 | ~12,000 | ~1,080 |
| `block_size=128` sees | ~25 words | 128 words | ~60–100 words |
| Unknown word handling | Never — all chars known | Skipped entirely | Split into subparts |
| Sentence quality | Needs 256+ block_size | Good at 128 | Good at 128 |

---

## Requirements

- **Python 3.8+** (no external dependencies required for basic usage)
- **PyTorch** (optional, for GPU acceleration) — install with:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

### Execution Tiers

The script automatically selects the best available backend:

| Priority | Condition                      | Backend                        |
|----------|--------------------------------|--------------------------------|
| 1st      | PyTorch + NVIDIA CUDA GPU      | `torch` on `cuda`              |
| 2nd      | PyTorch + Apple Silicon        | `torch` on `mps`               |
| 3rd      | PyTorch installed, no GPU      | `torch` on `cpu` (vectorized)  |
| 4th      | No PyTorch / `--no-gpu` flag   | Pure Python scalar autograd    |

---

## Command-Line Reference

```
python3 microgpt-sh.py [OPTIONS]
```

### Data Source Options

| Flag | Description |
|------|-------------|
| `--urls URL [URL ...]` | One or more URLs to fetch training text from |
| `--url-file PATH` | Path to a text file with one URL per line (lines starting with `#` are ignored) |
| `--no-cache` | Re-download all pages, ignoring the local cache |

If no URLs are specified, a default set of public-domain texts is used.

Fetched pages are cached in `.microgpt_cache/` next to the script. Subsequent runs
with the same URLs load from disk instantly.

### Tokenizer Options

| Flag | Description |
|------|-------------|
| `--tokenizer {char,word,bpe}` | Tokenization strategy (default: `char`) |
| `--bpe-merges N` | Number of BPE merge operations (default: 1000, only with `--tokenizer bpe`) |

See [Tokenizer Guide](#tokenizer-guide) below for details on each mode.

### Model Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--n-embd N` | 16 | Embedding dimension — width of each layer's representation |
| `--n-head N` | 4 | Number of attention heads per layer |
| `--n-layer N` | 1 | Number of transformer layers (depth) |
| `--block-size N` | 16 | Maximum sequence length (context window) |

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps N` | 1000 | Number of training iterations |
| `--batch-size N` | 64 | Documents per training step (PyTorch only) |

### Other Options

| Flag | Description |
|------|-------------|
| `--no-gpu` | Force pure-Python mode even if PyTorch/CUDA is available |
| `--interactive` | Prompt for each parameter interactively before training |
| `-h, --help` | Show help message and exit |

---

## Tokenizer Guide

The tokenizer controls how text is split into discrete symbols for the model.
This is the single biggest lever affecting output quality.

### Character Level (`--tokenizer char`)

Each character (letter, digit, punctuation) is one token.

- **Vocab size**: ~70–100 tokens (for English text)
- **Pros**: Simple, no unknown tokens, tiny vocabulary
- **Cons**: Very long sequences needed for sentences; `block_size=256` gives ~40 words of context
- **Best for**: Learning the basics, very small models, name generation

```bash
python3 microgpt-sh.py --tokenizer char --block-size 256 --n-embd 64
```

### Word Level (`--tokenizer word`)

Each whitespace-delimited word or punctuation mark is one token.

- **Vocab size**: depends on corpus (Bible ≈ 12K, Sherlock Holmes ≈ 8K)
- **Pros**: `block_size=128` gives 128 words of context; model learns grammar directly
- **Cons**: Words not in the training data are skipped (out-of-vocabulary problem)
- **Best for**: Structured/repetitive texts (Bible, legal documents), sentence generation

```bash
python3 microgpt-sh.py --tokenizer word --block-size 128 --n-embd 64
```

### BPE Subword (`--tokenizer bpe`)

Byte Pair Encoding learns common subword units from the training data.
The `--bpe-merges` flag controls vocabulary size: more merges = larger vocab,
more whole-word tokens, fewer tokens per sentence.

- **Vocab size**: base characters + `bpe-merges` + 1
- **Pros**: Handles unseen words by splitting into known subparts; best of both worlds
- **Cons**: BPE training takes a few minutes on first run (cached for subsequent runs)
- **Best for**: General-purpose text, mixed vocabularies, best quality

```bash
# Smaller vocab, more subword splitting
python3 microgpt-sh.py --tokenizer bpe --bpe-merges 500

# Balanced (default)
python3 microgpt-sh.py --tokenizer bpe --bpe-merges 1000

# Larger vocab, most common words stay whole
python3 microgpt-sh.py --tokenizer bpe --bpe-merges 3000
```

BPE merges are cached in `.microgpt_cache/` and reused automatically.

---

## Parameter Tuning Guide

### Model Size vs. Quality

| Config | Params | Speed | Quality | Use Case |
|--------|--------|-------|---------|----------|
| `--n-embd 16 --n-head 4 --n-layer 1` | ~4K | Seconds | Fragments | Quick tests |
| `--n-embd 64 --n-head 8 --n-layer 2` | ~50K | Minutes | Short phrases | Character patterns |
| `--n-embd 64 --n-head 8 --n-layer 3` | ~75K | Minutes | Sentences | Prose generation |
| `--n-embd 128 --n-head 8 --n-layer 4` | ~400K | ~1 hour | Paragraphs | Best quality |

### Block Size (Context Window)

The model can only "see" this many tokens at once. What it means depends on
the tokenizer:

| Block Size | Char Mode | Word Mode | BPE Mode (1K merges) |
|------------|-----------|-----------|----------------------|
| 16 | ~3 words | 16 words | ~8-12 words |
| 64 | ~12 words | 64 words | ~30-50 words |
| 128 | ~25 words | 128 words | ~60-100 words |
| 256 | ~50 words | 256 words (~1 page) | ~120-200 words |

### Training Steps

- **1,000**: Quick test, basic patterns learned
- **5,000**: Good for small models and small datasets
- **10,000**: Recommended for medium configs with sentence-level output
- **20,000+**: Diminishing returns unless model is large enough to benefit

### Batch Size (PyTorch only)

- **16–32**: Safe for large block sizes (256+) on 24GB GPU
- **64**: Good default for block sizes up to 128
- **128+**: Faster training, may run out of GPU memory with large models

---

## Recommended Configurations

### Name Generation (like the original)

```bash
python3 microgpt-sh.py \
  --tokenizer char \
  --urls 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt' \
  --n-embd 16 --n-head 4 --n-layer 1 --block-size 16 \
  --num-steps 1000
```

### Bible — Sentence Generation

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --n-embd 64 --n-head 8 --n-layer 3 --block-size 128 \
  --num-steps 10000 --batch-size 32
```

### Multi-Source with BPE

```bash
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 1500 \
  --urls \
    'https://www.gutenberg.org/cache/epub/1661/pg1661.txt' \
    'https://www.gutenberg.org/cache/epub/11/pg11.txt' \
  --n-embd 64 --n-head 8 --n-layer 3 --block-size 128 \
  --num-steps 15000 --batch-size 32
```

### Using a URL File

Create a file `my_urls.txt`:

```
# Classic literature
https://www.gutenberg.org/cache/epub/1661/pg1661.txt
https://www.gutenberg.org/cache/epub/11/pg11.txt

# Names dataset
https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
```

Then:

```bash
python3 microgpt-sh.py --url-file my_urls.txt --tokenizer word
```

### Interactive Parameter Mode

```bash
python3 microgpt-sh.py --interactive
```

Prompts you for each parameter with defaults shown in brackets — press Enter
to keep the default or type a new value.

---

## Interactive Text Completion

After training, the script enters an interactive REPL:

```
============================================================
Interactive mode — type a text prefix and the model will
continue it. Type 'quit' or Ctrl-C to exit.
============================================================

Prompt> honor the lord

  honor the lord of the king. And he said, Go, go thou, go down to the
```

The model encodes your prompt and generates a continuation in the style of its
training data. This is **text completion**, not a chatbot — the model continues
your text as if it were part of the training corpus.

**Tips:**
- Use words/phrases that appear in the training data for best results
- With word-level tokenization, unknown words are silently skipped
- With BPE tokenization, unknown words are split into known subparts
- Type `quit`, `exit`, `q`, or press Ctrl-C to exit

---

## Caching

The script maintains a `.microgpt_cache/` directory alongside itself:

- **URL content**: Fetched pages are cached by URL hash. Subsequent runs with
  the same URLs load instantly from disk.
- **BPE merges**: Learned BPE merge rules are cached by corpus hash + merge count.
  First run trains BPE (may take a few minutes on large corpora), subsequent runs
  load the cached merges.

To clear the cache:

```bash
rm -rf /path/to/microgpt/.microgpt_cache/
```

To force re-download without clearing BPE cache:

```bash
python3 microgpt-sh.py --no-cache
```

---

## Architecture

The model follows GPT-2 with minor simplifications:

- **RMSNorm** instead of LayerNorm (no learnable gain)
- **ReLU** instead of GeLU activation
- **No biases** in linear layers
- **Character/word/BPE tokenization** instead of GPT-2's fixed BPE
- **Single-file implementation** — everything in one Python script

### Pure-Python Mode

When PyTorch is not available (or `--no-gpu` is used), the script falls back to
a from-scratch implementation including:

- Scalar autograd engine (`Value` class with automatic differentiation)
- Hand-coded Adam optimizer
- Single-document training (no batching)

This mode is significantly slower but requires zero dependencies beyond Python 3.
