# minGPT-go

A pure-Go port of Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT):
a small, educational GPT implementation with a tape-based autograd engine, a
GPT-2 compatible BPE tokenizer, and a character-level training demo. **No CGo,
no external numeric libraries** — everything is stdlib Go and `float32` slices.

## Features

- **Tape-based autograd** (`autograd/`) with topological-sort reverse mode and
  gradient accumulation. All ops validated by finite-difference gradient checks.
- **Tensor** primitive (`tensor/`) with NumPy-style broadcasting, batched
  matmul (2D/3D/4D), and the usual element-wise ops.
- **Transformer model** (`model/`): embeddings, causal self-attention with
  pre-allocated causal mask, LayerNorm, GELU, residual stream, LM head.
- **AdamW optimizer** (`optim/`) with decoupled weight decay and global gradient
  clipping.
- **Trainer** (`trainer/`) with a `Dataset` interface and named callbacks.
- **Tokenizers** (`tokenizer/`): character-level, and a GPT-2 compatible BPE
  tokenizer (loads OpenAI's `vocab.json` / `merges.txt`).
- **Presets** matching karpathy's sizes: `gpt-nano`, `gpt-micro`, `gpt-mini`,
  `gpt1`.

## Quick start

```bash
# Grab Tiny Shakespeare
curl -LO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train a nano-GPT for 2000 iters, generating samples every 500, and save the
# final weights to disk
go run ./cmd/chargpt \
  -input input.txt \
  -model gpt-nano \
  -block_size 64 \
  -max_iters 2000 \
  -batch_size 32 \
  -lr 5e-4 \
  -gen_every 500 \
  -prompt "O God, O God!" \
  -save ckpt.gob

# Resume from the saved checkpoint and train for another 2000 iters
go run ./cmd/chargpt \
  -input input.txt \
  -load ckpt.gob \
  -max_iters 2000 \
  -save ckpt.gob
```

Expected: loss drops from ~4.2 → ~2.5 on `gpt-nano` (94k params) in a couple
thousand iters; samples start producing plausible Shakespearean character
distributions (capital letters, line breaks, punctuation rhythm) well before
they become grammatical.

## Project layout

```
github.com/fieldingj/mingpt/
├── tensor/          # Tensor type + non-differentiable ops
├── autograd/        # Node, Backward(), differentiable ops
├── nn/              # Linear, Embedding, LayerNorm, Dropout
├── optim/           # AdamW + ClipGradNorm
├── model/           # GPT config, attention, block, full model
├── trainer/         # Training loop + Dataset interface
├── tokenizer/       # CharTokenizer + BPETokenizer
└── cmd/chargpt/     # CLI demo
```

## Correctness

Every differentiable op has a finite-difference gradient check
(`autograd/engine_test.go`). Run:

```bash
go test ./...
```

The autograd suite covers: `Add`, `MatMul`, `GELU`, `Softmax`, `LayerNorm`,
`Embedding`, `CrossEntropy`, `Dropout`, and `MaskedFillNegInf`. Relative errors
stay below `1e-3`.

There's also a convergence test — `trainer.TestTrainerConverges` runs
`gpt-nano` on the digit-sort toy task from karpathy's demo notebook and fails
the build if loss hasn't dropped well below the random-uniform baseline.

Benchmarks for matmul and full GPT forward/backward live alongside the tests:

```bash
go test -bench=. -benchtime=3x -run=^$ ./tensor ./model
```

## Presets

| Name | Layers | Heads | Embed |
|------|--------|-------|-------|
| gpt-nano | 3 | 3 | 48 |
| gpt-micro | 4 | 4 | 128 |
| gpt-mini | 6 | 6 | 192 |
| gpt1 | 12 | 12 | 768 |

## Using the BPE tokenizer

```go
import "github.com/fieldingj/mingpt/tokenizer"

tok, err := tokenizer.NewBPETokenizer("vocab.json", "merges.txt")
if err != nil { log.Fatal(err) }
ids := tok.Encode("Hello, world!")
text := tok.Decode(ids)
```

Download GPT-2's BPE data from OpenAI's public CDN:
```
https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
```
(rename to `merges.txt` and `vocab.json`).

## License

MIT — same as the original minGPT. This is a clean-room Go port for educational
use.

## Credits

- Andrej Karpathy for the original [minGPT](https://github.com/karpathy/minGPT).
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2).
