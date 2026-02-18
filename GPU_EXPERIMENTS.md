# Maximizing GPU Utilization — Parallel Experiment Guide

Your Tesla P40 has 24GB VRAM and 3,840 CUDA cores. A single microgpt-sh.py run
with default parameters uses a fraction of that. This guide shows how to run
multiple experiments concurrently to keep the GPU busy and compare results faster.

---

## How It Works

NVIDIA GPUs support **concurrent kernels** from different processes. When you
launch multiple training runs, the GPU time-slices between them. Each process
gets its own CUDA context and memory allocation. As long as total VRAM usage
stays under 24GB, all runs proceed in parallel.

### VRAM Budget (Tesla P40 — 24GB)

Approximate VRAM per run (model + optimizer + activations):

| Config | Batch 32 | Batch 64 | Batch 128 |
|--------|----------|----------|-----------|
| n_embd=64, n_layer=2, block=128 | ~200MB | ~350MB | ~650MB |
| n_embd=64, n_layer=3, block=128 | ~250MB | ~450MB | ~800MB |
| n_embd=128, n_layer=4, block=128 | ~500MB | ~900MB | ~1.6GB |
| n_embd=128, n_layer=4, block=256 | ~800MB | ~1.4GB | ~2.5GB |
| n_embd=256, n_layer=6, block=256 | ~2GB | ~3.5GB | ~6GB |

**Rule of thumb**: You can comfortably run 4–8 small experiments or 2–3 large
ones simultaneously on the P40.

---

## Monitor GPU Usage

Keep a terminal open with:

```bash
# Live GPU monitoring (updates every 1 second)
watch -n 1 nvidia-smi
```

Key columns to watch:
- **GPU-Util**: percentage of GPU compute in use (target: 80%+)
- **Memory-Usage**: total VRAM consumed across all processes
- **Processes**: list of running training jobs and their memory

---

## Experiment Set 1: Tokenizer Comparison

Run all three tokenizer modes on the same data simultaneously.
Open three terminals side by side.

### Terminal 1 — Character Level

```bash
python3 microgpt-sh.py \
  --tokenizer char \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 256 --n-embd 64 --n-head 8 --n-layer 2 \
  --num-steps 10000 --batch-size 32 \
  2>&1 | tee results_char.log
```

### Terminal 2 — Word Level

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  2>&1 | tee results_word.log
```

### Terminal 3 — BPE Subword

```bash
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 1000 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  2>&1 | tee results_bpe.log
```

**Total VRAM**: ~700MB (3 × ~250MB). Leaves 23GB headroom.

**What to compare**: Final loss, sample quality, interactive prompt responses.
After all three finish, grep for the final results:

```bash
echo "=== CHAR ===" && tail -25 results_char.log
echo "=== WORD ===" && tail -25 results_word.log
echo "=== BPE ===" && tail -25 results_bpe.log
```

---

## Experiment Set 2: Model Depth Sweep

Test how adding layers affects quality. All use word tokenizer on the Bible.

### Terminal 1 — 1 Layer (baseline)

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 1 \
  --num-steps 10000 --batch-size 64 \
  2>&1 | tee results_layer1.log
```

### Terminal 2 — 2 Layers

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 2 \
  --num-steps 10000 --batch-size 64 \
  2>&1 | tee results_layer2.log
```

### Terminal 3 — 3 Layers

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 64 \
  2>&1 | tee results_layer3.log
```

### Terminal 4 — 4 Layers

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 4 \
  --num-steps 10000 --batch-size 64 \
  2>&1 | tee results_layer4.log
```

**Total VRAM**: ~1.2GB (4 × ~300MB). No problem.

**What to compare**: Loss convergence speed and final loss across depths.

```bash
for f in results_layer*.log; do
  echo "--- $f ---"
  grep "step.*10000" "$f" | tail -1
done
```

---

## Experiment Set 3: Embedding Width Sweep

Test how embedding size affects quality. Same depth (2 layers), vary width.

### Terminal 1 — n_embd=32 (narrow)

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 32 --n-head 4 --n-layer 2 \
  --num-steps 10000 --batch-size 64 \
  2>&1 | tee results_embd32.log
```

### Terminal 2 — n_embd=64 (medium)

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 2 \
  --num-steps 10000 --batch-size 64 \
  2>&1 | tee results_embd64.log
```

### Terminal 3 — n_embd=128 (wide)

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 128 --n-head 8 --n-layer 2 \
  --num-steps 10000 --batch-size 32 \
  2>&1 | tee results_embd128.log
```

### Terminal 4 — n_embd=256 (extra wide)

```bash
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 256 --n-head 16 --n-layer 2 \
  --num-steps 10000 --batch-size 16 \
  2>&1 | tee results_embd256.log
```

**Note**: Larger embeddings need more parameters, so batch size is reduced
to stay within VRAM. The `n_head` is also scaled (must divide `n_embd` evenly).

**Total VRAM**: ~3GB. Comfortable.

---

## Experiment Set 4: BPE Merge Count Sweep

Find the sweet spot for BPE vocabulary size.

### Terminal 1 — 250 merges (aggressive splitting)

```bash
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 250 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  2>&1 | tee results_bpe250.log
```

### Terminal 2 — 1000 merges (balanced)

```bash
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 1000 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  2>&1 | tee results_bpe1000.log
```

### Terminal 3 — 3000 merges (most words stay whole)

```bash
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 3000 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  2>&1 | tee results_bpe3000.log
```

### Terminal 4 — 5000 merges (approaching word-level)

```bash
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 5000 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  2>&1 | tee results_bpe5000.log
```

**Note**: BPE training runs once per merge count and is cached. If running
all four simultaneously, the first-time BPE training will compete for CPU.
Consider staggering the starts by 2–3 minutes, or run one first to warm the
cache, then launch the rest.

---

## Experiment Set 5: Maximum GPU Utilization

Push the P40 to its limits with one large experiment.

```bash
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 2000 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 512 --n-embd 256 --n-head 16 --n-layer 6 \
  --num-steps 20000 --batch-size 16 \
  2>&1 | tee results_max.log
```

This configuration:
- ~2.5M parameters
- block_size=512 means attention is O(512^2) per head — significant compute
- Should use 8–12GB VRAM
- Will take several hours but produce the best quality output

Monitor with `nvidia-smi` — you should see GPU-Util at 60–90%.

---

## Batch Launching with tmux

For running many experiments without keeping terminals open:

```bash
# Install tmux if needed
sudo apt install tmux

# Launch experiment set 2 (layer sweep) in tmux sessions
# Using --prompt-file so they run to completion unattended
tmux new-session -d -s layer1 'cd ~/microgpt && python3 microgpt-sh.py --tokenizer word --urls "https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt" --block-size 128 --n-embd 64 --n-head 8 --n-layer 1 --num-steps 10000 --batch-size 64 --prompt-file test_prompts.txt 2>&1 | tee results_layer1.log'

tmux new-session -d -s layer2 'cd ~/microgpt && python3 microgpt-sh.py --tokenizer word --urls "https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt" --block-size 128 --n-embd 64 --n-head 8 --n-layer 2 --num-steps 10000 --batch-size 64 --prompt-file test_prompts.txt 2>&1 | tee results_layer2.log'

tmux new-session -d -s layer3 'cd ~/microgpt && python3 microgpt-sh.py --tokenizer word --urls "https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt" --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 --num-steps 10000 --batch-size 64 --prompt-file test_prompts.txt 2>&1 | tee results_layer3.log'

tmux new-session -d -s layer4 'cd ~/microgpt && python3 microgpt-sh.py --tokenizer word --urls "https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt" --block-size 128 --n-embd 64 --n-head 8 --n-layer 4 --num-steps 10000 --batch-size 64 --prompt-file test_prompts.txt 2>&1 | tee results_layer4.log'

# Check status
tmux ls

# Attach to a session to watch progress
tmux attach -t layer3

# Detach with Ctrl-B then D
```

---

## Automated Test Prompts

Use `--prompt-file` to run the same prompts against every experiment automatically.
The script runs all prompts after training, prints results, and exits cleanly —
no interactive REPL blocking.

A sample file `test_prompts.txt` is included:

```
# Test prompts for comparing experiment results
And God said
In the beginning
The Lord is my
Thou shalt not
And he said unto
Blessed are the
The king of
For the wages of sin
And it came to pass
Love thy
```

### Using it with parallel experiments

Just add `--prompt-file test_prompts.txt` to every run:

```bash
# Terminal 1 — char
python3 microgpt-sh.py \
  --tokenizer char \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 256 --n-embd 64 --n-head 8 --n-layer 2 \
  --num-steps 10000 --batch-size 32 \
  --prompt-file test_prompts.txt \
  2>&1 | tee results_char.log

# Terminal 2 — word
python3 microgpt-sh.py \
  --tokenizer word \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  --prompt-file test_prompts.txt \
  2>&1 | tee results_word.log

# Terminal 3 — bpe
python3 microgpt-sh.py \
  --tokenizer bpe --bpe-merges 1000 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  --prompt-file test_prompts.txt \
  2>&1 | tee results_bpe.log
```

All three runs will:
1. Train on the Bible
2. Generate 20 random samples
3. Run all 10 test prompts identically
4. Exit cleanly (no hanging at `Prompt>`)

### Custom prompt files

Create your own for different corpora:

```bash
cat > my_prompts.txt << 'EOF'
# Shakespeare prompts
To be or not
Once upon a time
The quality of mercy
EOF

python3 microgpt-sh.py --prompt-file my_prompts.txt [other options...]
```

---

## Comparing Results

After experiments complete, a quick summary script:

```bash
#!/bin/bash
echo "============================================"
echo "Experiment Results Summary"
echo "============================================"
for f in results_*.log; do
  name=$(basename "$f" .log)
  final=$(grep "Final loss" "$f" 2>/dev/null | tail -1)
  params=$(grep "num params" "$f" 2>/dev/null | tail -1)
  vocab=$(grep "vocab size" "$f" 2>/dev/null | tail -1)
  echo ""
  echo "--- $name ---"
  echo "  $params"
  echo "  $vocab"
  echo "  $final"
  echo "  Random samples:"
  grep "^sample" "$f" 2>/dev/null | head -3
  echo "  Test prompts:"
  grep -A1 "Prompt:" "$f" 2>/dev/null | head -10
done
```

Save as `compare_results.sh` and run with `bash compare_results.sh`.

### Side-by-Side Prompt Comparison

To see how each experiment completed the same prompt:

```bash
#!/bin/bash
PROMPT="${1:-And God said}"
echo "Comparing completions for: \"$PROMPT\""
echo "============================================"
for f in results_*.log; do
  name=$(basename "$f" .log)
  output=$(grep -A1 "Prompt: $PROMPT" "$f" 2>/dev/null | grep "Output:" | sed 's/.*Output: //')
  printf "  %-20s %s\n" "$name:" "$output"
done
```

Save as `compare_prompt.sh` and run:

```bash
bash compare_prompt.sh "And God said"
bash compare_prompt.sh "Thou shalt not"
bash compare_prompt.sh "The Lord is my"
```

---

## Tips

- **Stagger BPE starts**: If multiple runs need BPE training (different merge
  counts), start them 2–3 minutes apart so BPE training doesn't bottleneck the CPU.
  After the first run, BPE merges are cached and subsequent runs start instantly.

- **Watch for OOM**: If you see `CUDA out of memory`, reduce `--batch-size` first
  (halve it), then `--block-size` if needed. The error message includes how much
  memory was requested vs available.

- **Log everything**: The `| tee results_X.log` pattern lets you watch output live
  while saving it to a file for later comparison.

- **Use `--prompt-file` for unattended runs**: Without it, the process hangs at
  the interactive `Prompt>`. With `--prompt-file test_prompts.txt`, it runs all
  prompts and exits cleanly — perfect for tmux and background jobs.

- **Check GPU before launching**: Always run `nvidia-smi` first to see if other
  experiments are still running and how much VRAM is free.

---

## Quick-Launch Example

Launch 3 tokenizer experiments in parallel, all using the same test prompts,
logging everything, exiting cleanly:

```bash
python3 microgpt-sh.py --tokenizer char  \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 256 --n-embd 64 --n-head 8 --n-layer 2 \
  --num-steps 10000 --batch-size 32 \
  --prompt-file test_prompts.txt 2>&1 | tee results_char.log &

python3 microgpt-sh.py --tokenizer word  \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  --prompt-file test_prompts.txt 2>&1 | tee results_word.log &

python3 microgpt-sh.py --tokenizer bpe --bpe-merges 1000 \
  --urls 'https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt' \
  --block-size 128 --n-embd 64 --n-head 8 --n-layer 3 \
  --num-steps 10000 --batch-size 32 \
  --prompt-file test_prompts.txt 2>&1 | tee results_bpe.log &

# Wait for all to finish
wait
echo "All experiments complete."

# Compare how each completed a specific prompt
bash compare_prompt.sh "And God said"
```
