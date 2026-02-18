"""
Multi-page microGPT: train and inference a GPT on text scraped from a list of web pages.
Based on the most atomic GPT implementation in pure, dependency-free Python.
GPU-accelerated via PyTorch when available, with automatic fallback to pure-Python CPU.

Originally by @karpathy — extended for multi-page, configurable params, and GPU support.
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
import argparse # command-line parameter input
import hashlib  # cache key generation
import re       # word tokenizer
import warnings
import urllib.request
from html.parser import HTMLParser
warnings.filterwarnings("ignore", message=".*NumPy.*")
random.seed(42) # Let there be order among chaos

# --- GPU / PyTorch detection with graceful fallback ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(42)
    USE_TORCH = True
except ImportError:
    USE_TORCH = False

if USE_TORCH:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Backend: PyTorch {torch.__version__} on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
else:
    device = 'cpu'
    print("Backend: pure Python (no PyTorch found — install torch for GPU acceleration)")

# --- HTML-to-text extractor (dependency-free) ---
class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and return plain text."""
    _skip_tags = {'script', 'style', 'noscript', 'head'}

    def __init__(self):
        super().__init__()
        self._pieces = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self):
        return ' '.join(self._pieces)

def html_to_text(html_str):
    extractor = HTMLTextExtractor()
    extractor.feed(html_str)
    return extractor.get_text()

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.microgpt_cache')

def _cache_path(url):
    """Return the local cache file path for a URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    safe_name = re.sub(r'[^\w.]', '_', url.split('/')[-1])[:60]
    return os.path.join(CACHE_DIR, f"{safe_name}_{url_hash}.txt")

def fetch_page_text(url, use_cache=True):
    """Download a URL and return its content as plain text. Caches to disk."""
    if use_cache:
        cp = _cache_path(url)
        if os.path.exists(cp):
            print(f"  cached:   {url}")
            with open(cp, 'r', encoding='utf-8') as f:
                return f.read()

    print(f"  fetching: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MicroGPT/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
        content_type = resp.headers.get('Content-Type', '')
        if 'html' in content_type.lower() or raw.lstrip().startswith('<'):
            text = html_to_text(raw)
        else:
            text = raw

        if use_cache and text:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(_cache_path(url), 'w', encoding='utf-8') as f:
                f.write(text)

        return text
    except Exception as e:
        print(f"  WARNING: could not fetch {url}: {e}")
        return ""

# --- Default list of pages to scrape ---
DEFAULT_URLS = [
    'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt',
    'https://www.gutenberg.org/cache/epub/1661/pg1661.txt',  # Sherlock Holmes
]

# --- Parse command-line arguments for model parameters and URLs ---
parser = argparse.ArgumentParser(
    description='Multi-page microGPT — train a tiny GPT on text from multiple web pages.',
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument('--urls', nargs='+', default=None,
                    help='One or more URLs to fetch training text from.\n'
                         'Default: a curated list of public-domain texts.')
parser.add_argument('--url-file', type=str, default=None,
                    help='Path to a text file containing one URL per line.')
parser.add_argument('--n-embd', type=int, default=None,
                    help='Embedding dimension (default: 16)')
parser.add_argument('--n-head', type=int, default=None,
                    help='Number of attention heads (default: 4)')
parser.add_argument('--n-layer', type=int, default=None,
                    help='Number of transformer layers (default: 1)')
parser.add_argument('--block-size', type=int, default=None,
                    help='Maximum sequence length (default: 16)')
parser.add_argument('--num-steps', type=int, default=None,
                    help='Number of training steps (default: 1000)')
parser.add_argument('--batch-size', type=int, default=None,
                    help='Batch size for PyTorch training (default: 64, ignored in pure-Python mode)')
parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'word', 'bpe'],
                    help='Tokenization level: "char" (default), "word", or "bpe" (subword).')
parser.add_argument('--bpe-merges', type=int, default=1000,
                    help='Number of BPE merge operations (default: 1000).\n'
                         'Only used with --tokenizer bpe. Controls vocab size:\n'
                         'final_vocab = num_unique_chars + bpe_merges + 1 (BOS).')
parser.add_argument('--no-cache', action='store_true',
                    help='Skip URL cache and re-download all pages.')
parser.add_argument('--no-gpu', action='store_true',
                    help='Force pure-Python CPU mode even if PyTorch/CUDA is available.')
parser.add_argument('--prompt-file', type=str, default=None,
                    help='Path to a text file with one test prompt per line.\n'
                         'Runs all prompts after training (replaces interactive REPL).\n'
                         'Results are printed to stdout for logging/comparison.')
parser.add_argument('--interactive', action='store_true',
                    help='Prompt for parameters interactively before training.')
args = parser.parse_args()

if args.no_gpu:
    USE_TORCH = False
    print("GPU disabled by --no-gpu flag, using pure-Python backend.")

def prompt_int(name, default):
    """Prompt user for an integer parameter; return default on empty input."""
    raw = input(f"  {name} [{default}]: ").strip()
    if raw == '':
        return default
    return int(raw)

# Resolve model parameters: CLI flag > interactive prompt > default
defaults = {'n_embd': 16, 'n_head': 4, 'n_layer': 1, 'block_size': 16, 'num_steps': 1000}
if args.interactive:
    print("\n=== Model parameter configuration ===")
    for key in defaults:
        cli_val = getattr(args, key.replace('-', '_'), None)
        if cli_val is not None:
            defaults[key] = cli_val
        else:
            defaults[key] = prompt_int(key, defaults[key])
    print()
else:
    for key in defaults:
        cli_val = getattr(args, key.replace('-', '_'), None)
        if cli_val is not None:
            defaults[key] = cli_val

# --- Resolve URLs ---
urls = DEFAULT_URLS
if args.urls:
    urls = args.urls
elif args.url_file:
    with open(args.url_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# --- Fetch and combine text from all pages ---
use_cache = not args.no_cache
print(f"Fetching text from {len(urls)} page(s)..." + (" (cache disabled)" if not use_cache else ""))
all_text = []
for url in urls:
    text = fetch_page_text(url, use_cache=use_cache)
    if text:
        all_text.append(text)
print(f"Fetched {len(all_text)} page(s) successfully.\n")
combined_text = '\n'.join(all_text)

# Split combined text into "documents" (lines / paragraphs)
docs = [line.strip() for line in combined_text.split('\n') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ============================================================================
# Tokenizer — char-level or word-level
# ============================================================================
TOKEN_MODE = args.tokenizer
print(f"tokenizer: {TOKEN_MODE}")

def word_tokenize(text):
    """Split text into words and punctuation tokens."""
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# --- BPE (Byte Pair Encoding) implementation — pure Python, no dependencies ---
import json

def train_bpe(text, num_merges):
    """Learn BPE merge rules from raw text. Returns (merges, vocab).
    Uses frequency-weighted word types for efficiency (standard BPE optimization).
    merges: list of (token_a, token_b) pairs in merge order.
    vocab: list of all token strings (base chars + merged tokens).
    """
    EOW = '</w>'

    # Count word frequencies — work with types, not tokens
    word_freq = {}
    for w in text.split():
        word_freq[w] = word_freq.get(w, 0) + 1

    # Represent each word type as a tuple of characters + EOW, with its frequency
    corpus = {}
    for word, freq in word_freq.items():
        key = tuple(list(word) + [EOW])
        corpus[key] = corpus.get(key, 0) + freq

    base_chars = sorted(set(ch for word in corpus for ch in word))
    merges = []

    print(f"  BPE training: {len(word_freq)} unique words, base vocab={len(base_chars)}, learning {num_merges} merges...")
    for i in range(num_merges):
        # Count bigrams across word types, weighted by word frequency
        pair_counts = {}
        for word, freq in corpus.items():
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq

        if not pair_counts:
            print(f"  BPE: no more pairs to merge after {i} merges.")
            break

        best_pair = max(pair_counts, key=pair_counts.get)
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # Merge best pair in all word types
        a, b = best_pair
        new_corpus = {}
        for word, freq in corpus.items():
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                    new_word.append(merged_token)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            new_key = tuple(new_word)
            new_corpus[new_key] = new_corpus.get(new_key, 0) + freq
        corpus = new_corpus

        if (i + 1) % 200 == 0 or i == num_merges - 1:
            print(f"  BPE merge {i+1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' -> '{merged_token}' (freq={pair_counts[best_pair]})")

    # Build final vocab: base chars + all merged tokens (in merge order)
    vocab = list(base_chars)
    for a, b_tok in merges:
        merged = a + b_tok
        if merged not in vocab:
            vocab.append(merged)

    return merges, vocab

def bpe_encode_word(word, merges):
    """Apply BPE merges to a single word, returning a list of subword token strings."""
    EOW = '</w>'
    symbols = list(word) + [EOW]
    for a, b in merges:
        merged = a + b
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [merged] + symbols[i + 2:]
            else:
                i += 1
    return symbols

def bpe_encode_text(text, merges):
    """Encode a text string into a list of BPE subword token strings."""
    tokens = []
    for word in text.split():
        tokens.extend(bpe_encode_word(word, merges))
    return tokens

def _bpe_cache_path(corpus_hash, num_merges):
    """Return cache file path for BPE merges."""
    return os.path.join(CACHE_DIR, f"bpe_merges_{corpus_hash}_{num_merges}.json")

def save_bpe_cache(merges, vocab, cache_path):
    """Save BPE merges and vocab to a JSON cache file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {'merges': merges, 'vocab': vocab}
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def load_bpe_cache(cache_path):
    """Load BPE merges and vocab from a JSON cache file. Returns (merges, vocab) or None."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r') as f:
        data = json.load(f)
    merges = [tuple(m) for m in data['merges']]
    return merges, data['vocab']

if TOKEN_MODE == 'bpe':
    num_bpe_merges = args.bpe_merges
    corpus_text = ' '.join(docs)
    corpus_hash = hashlib.sha256(corpus_text[:10000].encode()).hexdigest()[:12]
    cache_path = _bpe_cache_path(corpus_hash, num_bpe_merges)

    cached = load_bpe_cache(cache_path)
    if cached and not args.no_cache:
        bpe_merges, vocab_list = cached
        print(f"  BPE merges loaded from cache ({len(bpe_merges)} merges)")
    else:
        bpe_merges, vocab_list = train_bpe(corpus_text, num_bpe_merges)
        save_bpe_cache(bpe_merges, vocab_list, cache_path)

    token2id = {tok: i for i, tok in enumerate(vocab_list)}
    id2token = {i: tok for tok, i in token2id.items()}
    BOS = len(vocab_list)
    vocab_size = len(vocab_list) + 1
    print(f"vocab size: {vocab_size} (BPE, {len(bpe_merges)} merges, {len(vocab_list)} subword tokens + BOS)")

    def encode_doc(doc):
        """Tokenize a document using BPE: BOS + subword ids + BOS."""
        subtokens = bpe_encode_text(doc, bpe_merges)
        return [BOS] + [token2id[t] for t in subtokens if t in token2id] + [BOS]

    def encode_prompt(text):
        """Encode a user prompt into BPE token ids, returning (ids, skipped)."""
        subtokens = bpe_encode_text(text, bpe_merges)
        ids, skipped = [], []
        for t in subtokens:
            if t in token2id:
                ids.append(token2id[t])
            else:
                skipped.append(t)
        return ids, skipped

    def decode_tokens(token_ids):
        """Convert BPE token ids back to readable text."""
        EOW = '</w>'
        pieces = []
        for tid in token_ids:
            if tid == BOS:
                continue
            tok = id2token.get(tid, '')
            pieces.append(tok)
        # Join and restore spaces at end-of-word boundaries
        text = ''.join(pieces)
        text = text.replace(EOW, ' ')
        return text.strip()

elif TOKEN_MODE == 'word':
    all_tokens_flat = []
    for doc in docs:
        all_tokens_flat.extend(word_tokenize(doc))
    vocab_list = sorted(set(all_tokens_flat))
    token2id = {tok: i for i, tok in enumerate(vocab_list)}
    id2token = {i: tok for tok, i in token2id.items()}
    BOS = len(vocab_list)
    vocab_size = len(vocab_list) + 1
    print(f"vocab size: {vocab_size} (word-level, {len(vocab_list)} unique tokens + BOS)")

    def encode_doc(doc):
        """Tokenize a document at word level: BOS + word ids + BOS."""
        words = word_tokenize(doc)
        return [BOS] + [token2id[w] for w in words if w in token2id] + [BOS]

    def encode_prompt(text):
        """Encode a user prompt into token ids, returning (ids, skipped)."""
        words = word_tokenize(text)
        ids, skipped = [], []
        for w in words:
            if w in token2id:
                ids.append(token2id[w])
            else:
                skipped.append(w)
        return ids, skipped

    def decode_tokens(token_ids):
        """Convert token ids back to readable text."""
        pieces = []
        for tid in token_ids:
            if tid == BOS:
                continue
            tok = id2token.get(tid, '?')
            if pieces and tok not in '.,;:!?\'")-]}' and pieces[-1] not in '(\'"[{':
                pieces.append(' ')
            pieces.append(tok)
        return ''.join(pieces)

else:  # char-level (original)
    vocab_list = sorted(set(''.join(docs)))
    token2id = {ch: i for i, ch in enumerate(vocab_list)}
    id2token = {i: ch for ch, i in token2id.items()}
    BOS = len(vocab_list)
    vocab_size = len(vocab_list) + 1
    print(f"vocab size: {vocab_size} (char-level)")

    def encode_doc(doc):
        """Tokenize a document at char level: BOS + char ids + BOS."""
        return [BOS] + [token2id[ch] for ch in doc] + [BOS]

    def encode_prompt(text):
        """Encode a user prompt into token ids, returning (ids, skipped)."""
        ids, skipped = [], []
        for ch in text:
            if ch in token2id:
                ids.append(token2id[ch])
            else:
                skipped.append(ch)
        return ids, skipped

    def decode_tokens(token_ids):
        """Convert token ids back to readable text."""
        return ''.join(id2token.get(tid, '') for tid in token_ids if tid != BOS)

# ============================================================================
# Model parameters (from user config)
# ============================================================================
n_embd = defaults['n_embd']          # embedding dimension
n_head = defaults['n_head']          # number of attention heads
n_layer = defaults['n_layer']        # number of layers
block_size = defaults['block_size']  # maximum sequence length
num_steps = defaults['num_steps']    # number of training steps
batch_size = args.batch_size if args.batch_size else 64
head_dim = n_embd // n_head
print(f"config: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, block_size={block_size}")

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
temperature = 0.5

# ============================================================================
# PATH A: PyTorch (GPU or CPU) — fast, batched tensor operations
# ============================================================================
if USE_TORCH:

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps

        def forward(self, x):
            ms = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(ms + self.eps)

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(n_embd, n_embd, bias=False)
            self.wk = nn.Linear(n_embd, n_embd, bias=False)
            self.wv = nn.Linear(n_embd, n_embd, bias=False)
            self.wo = nn.Linear(n_embd, n_embd, bias=False)

        def forward(self, x):
            B, T, C = x.shape
            q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
            k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
            v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.wo(y)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
            self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = RMSNorm(n_embd)
            self.attn = CausalSelfAttention()
            self.norm2 = RMSNorm(n_embd)
            self.mlp = MLP()

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, n_embd)
            self.wpe = nn.Embedding(block_size, n_embd)
            self.norm_in = RMSNorm(n_embd)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.08)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.08)

        def forward(self, idx):
            B, T = idx.shape
            tok_emb = self.wte(idx)
            pos_emb = self.wpe(torch.arange(T, device=idx.device))
            x = self.norm_in(tok_emb + pos_emb)
            for block in self.blocks:
                x = block(x)
            return self.lm_head(x)

    # --- Initialize model on device ---
    model = GPT().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num params: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(beta1, beta2), eps=eps_adam)

    # --- Training ---
    print(f"\nTraining for {num_steps} steps (batch_size={batch_size}) on {device}...")
    model.train()
    for step in range(num_steps):
        lr_t = learning_rate * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        batch_tokens = []
        for b in range(batch_size):
            doc = docs[(step * batch_size + b) % len(docs)]
            tokens = encode_doc(doc)
            tokens = tokens[:block_size + 1]
            batch_tokens.append(tokens)

        max_len = max(len(t) for t in batch_tokens)
        input_ids = torch.full((batch_size, max_len - 1), BOS, dtype=torch.long, device=device)
        target_ids = torch.full((batch_size, max_len - 1), -100, dtype=torch.long, device=device)

        for b, tokens in enumerate(batch_tokens):
            n = min(block_size, len(tokens) - 1)
            input_ids[b, :n] = torch.tensor(tokens[:n], dtype=torch.long)
            target_ids[b, :n] = torch.tensor(tokens[1:n+1], dtype=torch.long)

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 50 == 0 or step == num_steps - 1:
            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

    print(f"\nFinal loss: {loss.item():.4f}")

    # --- Inference: random samples ---
    print("\n--- inference (hallucinated text from the learned corpus) ---")
    model.eval()
    with torch.no_grad():
        for sample_idx in range(20):
            tokens = [BOS]
            for pos_id in range(block_size):
                idx = torch.tensor([tokens[-block_size:]], dtype=torch.long, device=device)
                logits = model(idx)
                logits = logits[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                if token_id == BOS:
                    break
                tokens.append(token_id)
            print(f"sample {sample_idx+1:2d}: {decode_tokens(tokens)}")

    # --- Prompt completion (file-driven or interactive REPL) ---
    def _torch_complete(prompt):
        """Run a single prompt through the PyTorch model and return the result."""
        prompt_tokens, skipped = encode_prompt(prompt)
        if skipped:
            print(f"  (skipped unknown tokens: {skipped})")
        tokens = [BOS] + prompt_tokens
        if len(tokens) > block_size:
            tokens = tokens[-block_size:]
        for _ in range(max(block_size - len(tokens), block_size)):
            idx = torch.tensor([tokens[-block_size:]], dtype=torch.long, device=device)
            logits = model(idx)
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            tokens.append(token_id)
        return decode_tokens(tokens)

    if args.prompt_file:
        with open(args.prompt_file) as f:
            test_prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"\n{'=' * 60}")
        print(f"Running {len(test_prompts)} test prompts from: {args.prompt_file}")
        print(f"{'=' * 60}")
        with torch.no_grad():
            for i, prompt in enumerate(test_prompts, 1):
                result = _torch_complete(prompt)
                print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt}")
                print(f"  Output: {result}")
        print(f"\n{'=' * 60}")
        print("All test prompts completed.")
    else:
        print("\n" + "=" * 60)
        print("Interactive mode — type a text prefix and the model will")
        print("continue it. Type 'quit' or Ctrl-C to exit.")
        print("=" * 60)
        with torch.no_grad():
            while True:
                try:
                    prompt = input("\nPrompt> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye.")
                    break
                if prompt.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye.")
                    break
                if not prompt:
                    continue
                print(f"\n  {_torch_complete(prompt)}")

# ============================================================================
# PATH B: Pure Python fallback — no dependencies, scalar autograd
# ============================================================================
else:
    # Autograd engine: recursively apply the chain rule through a computation graph
    class Value:
        __slots__ = ('data', 'grad', '_children', '_local_grads')

        def __init__(self, data, children=(), local_grads=()):
            self.data = data
            self.grad = 0
            self._children = children
            self._local_grads = local_grads

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(self.data + other.data, (self, other), (1, 1))

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(self.data * other.data, (self, other), (other.data, self.data))

        def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
        def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
        def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
        def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
        def __neg__(self): return self * -1
        def __radd__(self, other): return self + other
        def __sub__(self, other): return self + (-other)
        def __rsub__(self, other): return other + (-self)
        def __rmul__(self, other): return self * other
        def __truediv__(self, other): return self * other**-1
        def __rtruediv__(self, other): return other * self**-1

        def backward(self):
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._children:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            self.grad = 1
            for v in reversed(topo):
                for child, local_grad in zip(v._children, v._local_grads):
                    child.grad += local_grad * v.grad

    # --- Model functions (scalar, GPT-2 style) ---
    def linear(x, w):
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

    def softmax(logits):
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def rmsnorm(x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    # --- Initialize parameters ---
    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"num params: {len(params)}")

    def gpt(token_id, pos_id, keys, values):
        tok_emb = state_dict['wte'][token_id]
        pos_emb = state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)
        for li in range(n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f'layer{li}.attn_wq'])
            k = linear(x, state_dict[f'layer{li}.attn_wk'])
            v = linear(x, state_dict[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs+head_dim]
                k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+head_dim] for vi in values[li]]
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
                attn_weights = softmax(attn_logits)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]
        return linear(x, state_dict['lm_head'])

    # --- Training (single-doc, scalar autograd) ---
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)
    print(f"\nTraining for {num_steps} steps (pure Python, single-doc per step)...")
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = encode_doc(doc)
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)

        loss.backward()

        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    # --- Inference: random samples ---
    print("\n--- inference (hallucinated text from the learned corpus) ---")
    for sample_idx in range(20):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        generated_ids = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            generated_ids.append(token_id)
        print(f"sample {sample_idx+1:2d}: {decode_tokens(generated_ids)}")

    # --- Prompt completion (file-driven or interactive REPL) ---
    def _scalar_complete(prompt):
        """Run a single prompt through the scalar model and return the result."""
        prompt_tokens, skipped = encode_prompt(prompt)
        if skipped:
            print(f"  (skipped unknown tokens: {skipped})")
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        all_tokens = [BOS] + prompt_tokens
        for pos_id in range(len(all_tokens) - 1):
            logits = gpt(all_tokens[pos_id], pos_id, keys, values)
        token_id = all_tokens[-1]
        pos_id = len(all_tokens) - 1
        generated = list(prompt_tokens)
        for _ in range(block_size - len(all_tokens)):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            generated.append(token_id)
            pos_id += 1
            if pos_id >= block_size:
                break
        return decode_tokens(generated)

    if args.prompt_file:
        with open(args.prompt_file) as f:
            test_prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"\n{'=' * 60}")
        print(f"Running {len(test_prompts)} test prompts from: {args.prompt_file}")
        print(f"{'=' * 60}")
        for i, prompt in enumerate(test_prompts, 1):
            result = _scalar_complete(prompt)
            print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt}")
            print(f"  Output: {result}")
        print(f"\n{'=' * 60}")
        print("All test prompts completed.")
    else:
        print("\n" + "=" * 60)
        print("Interactive mode — type a text prefix and the model will")
        print("continue it. Type 'quit' or Ctrl-C to exit.")
        print("=" * 60)
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Goodbye.")
                break
            if not prompt:
                continue
            print(f"\n  {_scalar_complete(prompt)}")