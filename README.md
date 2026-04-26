# minGPT-go

[![ci](https://github.com/jpfielding/mingpt-go/actions/workflows/ci.yml/badge.svg)](https://github.com/jpfielding/mingpt-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/fieldingj/mingpt.svg)](https://pkg.go.dev/github.com/fieldingj/mingpt)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Go port of Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT).
Trains a real transformer language model end-to-end — autograd, attention,
AdamW, BPE tokenizer, and all — with nothing but the Go standard library.

**No CGo. No external numeric libraries. No Python.** `float32` slices and a
tape-based autograd engine, start to finish.

```
go get github.com/fieldingj/mingpt
```

---

## Why this exists

The Go ecosystem has ML inference libraries (gorgonia, ONNX bindings, TF-Lite
wrappers), but almost no pure-Go code that trains a modern transformer from
first principles. This repo is that code. It's meant to be:

- **Readable.** Every op is ~20 lines. The full autograd engine is ~100 lines.
  You can follow the backward pass of attention by reading the file.
- **Correct.** Every differentiable op has a finite-difference gradient check,
  and a sort-task convergence test fails the build if training regresses.
- **Self-contained.** One `go test ./...` runs the whole suite. No Python, no
  CUDA, no LAPACK.

It is *not* fast. A tuned tensor library this is not — if you want to train
GPT-2 scale models, reach for llama.cpp or PyTorch. Use this when you want to
*understand* what `loss.backward()` actually does.

---

## Features

| | |
|---|---|
| **Tensor** (`tensor/`) | Row-major `float32` slices, NumPy-style broadcasting, batched matmul (2D/3D/4D), transpose, reductions |
| **Autograd** (`autograd/`) | Tape-based reverse-mode with topological sort. Ops: `Add`, `MatMul`, `GELU`, `Softmax`, `LayerNorm`, `Embedding`, `Dropout`, `CrossEntropyLoss`, `MaskedFillNegInf`, `Transpose` |
| **Modules** (`nn/`) | `Linear`, `Embedding`, `LayerNorm`, `Dropout` |
| **Optimizer** (`optim/`) | `AdamW` with decoupled weight decay + parameter groups, `ClipGradNorm` |
| **Model** (`model/`) | Causal self-attention, transformer block, full GPT; presets `gpt-nano`, `gpt-micro`, `gpt-mini`, `gpt1` |
| **Trainer** (`trainer/`) | Training loop, `Dataset` interface, named callbacks |
| **Tokenizers** (`tokenizer/`) | Character-level; GPT-2 compatible BPE (loads OpenAI's `vocab.json` / `merges.txt`) |
| **Checkpoints** | `(*GPT).Save(path)` / `LoadGPT(path)` via `encoding/gob` |
| **CLI** (`cmd/chargpt`) | Train a char-level model from a text file, with `-save` / `-load` |

---

## Quick start

```bash
git clone https://github.com/jpfielding/mingpt-go
cd mingpt-go

# Grab Tiny Shakespeare (~1 MB of Shakespeare plays)
curl -LO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train a nano-GPT for 2000 iters, sample every 500, save at the end
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

# Resume from the checkpoint
go run ./cmd/chargpt \
  -input input.txt \
  -load ckpt.gob \
  -max_iters 2000 \
  -save ckpt.gob
```

Expected on `gpt-nano` (~94k params): loss drops from ~4.2 → ~2.5 within a
couple thousand iters. Samples get the *shape* of Shakespeare (capitals, line
breaks, rhythmic punctuation, character name prefixes) well before they become
grammatical. That's the point — you're watching a language model learn.

---

## Architecture

Package dependency graph (each package only imports the ones below it):

```
cmd/chargpt
    │
    ├── trainer ──────────────── optim
    │                              │
    ├── model ──── nn ──── autograd
    │                         │
    └── tokenizer          tensor
```

- `tensor` is the bottom layer — no autograd awareness, just float32 math.
- `autograd` wraps tensors in `Node`s that record their own backward closures.
- `nn` layers hold `Param(...)` nodes as trainable weights.
- `model` composes `nn` modules into a GPT.
- `optim` updates parameter values in place using `.Grad` accumulated by
  `autograd.Backward`.
- `trainer` orchestrates the loop; `tokenizer` is independent of everything.

---

## API tour

### Tensors

```go
import "github.com/fieldingj/mingpt/tensor"

a := tensor.RandNorm(rng, 0.02, 4, 8)     // [4,8] Gaussian
b := tensor.Ones(8, 3)                    // [8,3] all-ones
c := tensor.MatMul(a, b)                  // [4,3]

// Broadcasting works NumPy-style.
bias := tensor.Zeros(1, 3)
d := tensor.ElemAdd(c, bias)              // [4,3] + [1,3] -> [4,3]
```

### Autograd

```go
import "github.com/fieldingj/mingpt/autograd"

// Parameters are leaf nodes.
w := autograd.Param(tensor.RandNorm(rng, 0.02, 4, 4))
x := autograd.Const(tensor.Ones(4, 4))

y := autograd.MatMul(x, w)
loss := autograd.CrossEntropyLoss(
    reshapeTo3D(y, 1, 4, 4),              // [B=1, T=4, V=4]
    []int{0, 1, 2, 3},                    // targets
)

autograd.Backward(loss)                   // fills in w.Grad
fmt.Println(w.Grad.Shape)                 // [4, 4]
```

Every op records a backward closure when its forward runs. `Backward` does a
topological sort from the loss node, seeds `grad=1`, and walks the graph in
reverse calling each closure.

### Modules

```go
import "github.com/fieldingj/mingpt/nn"

nn.RNG = rand.New(rand.NewSource(42))     // set before constructing layers
lin := nn.NewLinear(128, 64, true, 0.02)  // 128->64 with bias, std=0.02
emb := nn.NewEmbedding(50257, 768, 0.02)
ln  := nn.NewLayerNorm(768, 1e-5)
dp  := nn.NewDropout(0.1)

params := lin.Parameters()                // []*autograd.Node
```

### A full GPT

```go
import "github.com/fieldingj/mingpt/model"

cfg := model.GPTMini                      // 6 layers, 6 heads, embd=192
cfg.VocabSize = 50257
cfg.BlockSize = 256
gpt := model.NewGPT(cfg)

logits, loss := gpt.Forward(tokens, targets)
autograd.Backward(loss)

// Inference:
out := gpt.Generate(promptIDs, 200, 1.0, 40, rng)  // temp=1, top-k=40
```

### Training loop

```go
import "github.com/fieldingj/mingpt/trainer"

trCfg := trainer.DefaultConfig()
trCfg.MaxIters  = 5000
trCfg.BatchSize = 64
trCfg.LR        = 5e-4

tr := trainer.New(trCfg, gpt, dataset)

tr.AddCallback("on_eval", func(t *trainer.Trainer) {
    fmt.Printf("iter %d: %.4f\n", t.Iter, t.LastLoss)
})

tr.Run()
```

`Dataset` is a one-method interface:

```go
type Dataset interface {
    Len() int
    GetBatch(batchSize int, rng *rand.Rand) (inputs, targets []int)
}
```

Both `tokenizer.CharDataset` and `tokenizer.BPEDataset` satisfy it, and so
does whatever you write yourself.

### Checkpointing

```go
gpt.Save("ckpt.gob")                      // writes config + params via gob

restored, err := model.LoadGPT("ckpt.gob")
```

The gob blob stores the `GPTConfig` plus every parameter tensor in the fixed
order produced by `(*GPT).Parameters()`. `LoadGPT` rebuilds the model from the
saved config and copies weights in. A `version` field guards against format
drift.

### BPE tokenizer (GPT-2 compatible)

```go
tok, err := tokenizer.NewBPETokenizer("vocab.json", "merges.txt")
if err != nil { log.Fatal(err) }

ids := tok.Encode("Hello, world!")
text := tok.Decode(ids)                   // round-trips losslessly
```

Download GPT-2's BPE data from OpenAI's public CDN:

```bash
curl -LO https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
curl -LO https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
mv vocab.bpe merges.txt
mv encoder.json vocab.json
```

The implementation includes GPT-2's byte-level fallback (all 256 bytes map to
printable Unicode so arbitrary binary is encodable) and a hand-rolled
pre-tokenizer state machine — Go's `regexp` doesn't support `\p{L}` so the
original GPT-2 regex had to be reimplemented as explicit branching over runes.

---

## Presets

Matches karpathy's `model_type` shortcuts:

| Name | Layers | Heads | Embed | Params (vocab=65, block=128) |
|------|--------|-------|-------|------------------------------|
| `gpt-nano`  | 3  | 3  | 48  | ~100 k |
| `gpt-micro` | 4  | 4  | 128 | ~800 k |
| `gpt-mini`  | 6  | 6  | 192 | ~3 M   |
| `gpt1`      | 12 | 12 | 768 | ~125 M |

`gpt1` is a reference size — you probably don't want to train it on CPU in
pure Go.

---

## Correctness

```bash
go test ./...
```

Three layers of verification:

1. **Finite-difference gradient checks** (`autograd/engine_test.go`). Every
   differentiable op is exercised: a numerical gradient via central difference
   `(f(x+ε)-f(x-ε))/(2ε)` is compared to the analytical gradient returned by
   the backward closure. Max relative error stays below `1e-3` across `Add`,
   `MatMul`, `GELU`, `Softmax`, `LayerNorm`, `Embedding`, `Dropout`,
   `CrossEntropyLoss`, and `MaskedFillNegInf`.

2. **Convergence test** (`trainer/trainer_test.go`). Runs `gpt-nano` on the
   digit-sort toy task from karpathy's demo notebook and fails the build if
   loss hasn't dropped well below the `ln(3) ≈ 1.1` uniform baseline after
   500 iterations. Current run: 0.0004 final loss in 43 seconds.

3. **Checkpoint equivalence** (`model/checkpoint_test.go`). A round-tripped
   model produces byte-identical logits for the same input, proving the
   serialization is lossless.

---

## Benchmarks

Apple M2 Max, single-threaded, `go1.26.1`:

| Benchmark | Time | Allocs |
|---|---:|---:|
| `MatMul2D` (64×128 @ 128×256) | 2.0 ms | 3 |
| `MatMul3D` (batch=8, 64×128 @ 128×256) | 15 ms | 3 |
| `ElemAdd` (1024×1024) | 4.6 ms | 6 |
| `GPTForward` (2 layers, embd=64, batch=4×32) | 12 ms | 492 |
| `GPTForwardBackward` (same config) | 40 ms | 1 240 |

```bash
go test -bench=. -benchtime=3x -run=^$ ./tensor ./model
```

Obvious optimizations left on the table: tiled matmul, SIMD via
`golang.org/x/sys`, op fusion (GELU + Linear), buffer pools to cut the
allocations per backward pass. The goal was clarity, not throughput — PRs
welcome if you want to dig in.

---

## Project layout

```
github.com/fieldingj/mingpt/
├── tensor/          Tensor type + non-differentiable ops
│   ├── tensor.go    Tensor, constructors, reshape, transpose
│   └── ops.go       ElemAdd/Sub/Mul/Div, MatMul, SumAxis, broadcasting
├── autograd/        Reverse-mode autograd
│   ├── engine.go    Node, Backward(), topological sort
│   └── ops.go       Differentiable ops with backward closures
├── nn/              Neural net building blocks
│   ├── linear.go    Linear (y = xWᵀ + b)
│   ├── embedding.go Embedding (gather + scatter-add backward)
│   ├── layernorm.go LayerNorm
│   ├── dropout.go   Inverted dropout
│   └── module.go    Module interface, shared RNG
├── optim/           Optimization
│   └── adamw.go     AdamW with decoupled weight decay, ClipGradNorm
├── model/           Transformer assembly
│   ├── config.go    GPTConfig + presets
│   ├── attention.go CausalSelfAttention (cached causal mask)
│   ├── block.go     LN → Attn → residual → LN → FFN → residual
│   ├── gpt.go       Full model + Generate + ConfigureOptimizer
│   └── checkpoint.go Save / LoadGPT via gob
├── trainer/         Training loop
│   └── trainer.go   Trainer.Run(), Dataset, Callback
├── tokenizer/       Text ↔ token IDs
│   ├── char.go      Character-level
│   └── bpe.go       GPT-2 BPE with byte-level fallback
└── cmd/chargpt/     CLI: train a char-level GPT on any text file
    └── main.go
```

---

## What's *not* here

Honest list of things deliberately skipped:

- **GPU / SIMD / parallelism.** All ops are single-threaded scalar Go.
- **Op fusion.** Every node allocates its own output tensor.
- **Gradient checkpointing.** The whole computation graph lives in memory.
- **Mixed precision.** `float32` only.
- **Pretrained weights.** You can't load GPT-2 checkpoints from Hugging Face;
  the checkpoint format is native gob, not safetensors.
- **PyTorch numerical-parity harness.** Gradient checks are analytical-vs-
  numerical only. A `cmd/verify/` that compares against PyTorch reference
  outputs is on the list but not done.
- **RoPE / ALiBi / grouped-query attention.** This is minGPT — vanilla causal
  self-attention with learned position embeddings, nothing fancier.

---

## Go version policy

Scripts and commands pin the latest stable Go (`go 1.26` in `go.mod`). The
library packages use only stdlib features compatible with the same version.
No external dependencies, no `go.sum`.

---

## License

MIT. See [LICENSE](LICENSE). Derivative of karpathy/minGPT, also MIT.

## Credits

- Andrej Karpathy for the original [minGPT](https://github.com/karpathy/minGPT).
- Radford et al., *[Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models)* (GPT-2).
- Vaswani et al., *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*.
