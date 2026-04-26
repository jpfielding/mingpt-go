package model

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/fieldingj/mingpt/autograd"
	"github.com/fieldingj/mingpt/nn"
	"github.com/fieldingj/mingpt/optim"
)

// GPT is the complete transformer language model.
type GPT struct {
	cfg      GPTConfig
	training bool

	TokEmb *nn.Embedding
	PosEmb *nn.Embedding
	Drop   *nn.Dropout
	Blocks []*Block
	LNF    *nn.LayerNorm
	Head   *nn.Linear // no bias, NEmbd → VocabSize
}

// NewGPT creates a GPT model and initialises all parameters.
func NewGPT(cfg GPTConfig) *GPT {
	residStd := defaultResidStd(cfg.NLayer)
	g := &GPT{
		cfg:      cfg,
		training: true,
		TokEmb:   nn.NewEmbedding(cfg.VocabSize, cfg.NEmbd, 0.02),
		PosEmb:   nn.NewEmbedding(cfg.BlockSize, cfg.NEmbd, 0.02),
		Drop:     nn.NewDropout(cfg.EmbdDrop),
		LNF:      nn.NewLayerNorm(cfg.NEmbd, 1e-5),
		Head:     nn.NewLinear(cfg.NEmbd, cfg.VocabSize, false, 0.02),
	}
	g.Blocks = make([]*Block, cfg.NLayer)
	for i := range g.Blocks {
		g.Blocks[i] = NewBlock(cfg, residStd)
	}
	return g
}

// NumParams returns the total number of trainable parameters.
func (g *GPT) NumParams() int {
	n := 0
	for _, p := range g.Parameters() {
		n += p.Value.Numel()
	}
	return n
}

// Parameters returns all leaf parameter nodes.
func (g *GPT) Parameters() []*autograd.Node {
	var params []*autograd.Node
	params = append(params, g.TokEmb.Parameters()...)
	params = append(params, g.PosEmb.Parameters()...)
	params = append(params, g.LNF.Parameters()...)
	params = append(params, g.Head.Parameters()...)
	for _, b := range g.Blocks {
		params = append(params, b.Parameters()...)
	}
	return params
}

// Train sets all layers to training mode.
func (g *GPT) Train() {
	g.training = true
	g.Drop.SetTraining(true)
	for _, b := range g.Blocks {
		b.SetTraining(true)
	}
}

// Eval sets all layers to inference mode (dropout disabled).
func (g *GPT) Eval() {
	g.training = false
	g.Drop.SetTraining(false)
	for _, b := range g.Blocks {
		b.SetTraining(false)
	}
}

// Forward runs the model.
// tokenIDs: []int of length B*T.
// targets: []int of length B*T, or nil for inference.
// Returns (logits, loss). loss is nil if targets is nil.
func (g *GPT) Forward(tokenIDs []int, targets []int) (*autograd.Node, *autograd.Node) {
	N := len(tokenIDs)
	T := g.cfg.BlockSize
	if N < T {
		T = N
	}
	B := N / T
	if B == 0 {
		B = 1
		T = N
	}

	// Positional indices 0..T-1, same for all batches.
	posIDs := make([]int, T)
	for i := range posIDs {
		posIDs[i] = i
	}

	// Token + position embeddings.
	tok := g.TokEmb.Lookup(tokenIDs, B, T)     // [B, T, C]
	pos := g.PosEmb.Lookup(posIDs, 1, T)        // [1, T, C] → broadcast
	x := autograd.Add(tok, pos)
	x = g.Drop.Forward(x)

	// Transformer blocks.
	for _, block := range g.Blocks {
		x = block.Forward(x)
	}
	x = g.LNF.Forward(x)

	// Language modelling head.
	logits := g.Head.Forward(x) // [B, T, V]

	var loss *autograd.Node
	if targets != nil {
		loss = autograd.CrossEntropyLoss(logits, targets)
	}
	return logits, loss
}

// Generate autoregressively extends ctx by nTokens.
// topK=0 disables top-k filtering (pure temperature sampling).
func (g *GPT) Generate(ctx []int, nTokens int, temperature float32, topK int, rng *rand.Rand) []int {
	g.Eval()
	defer g.Train()

	result := make([]int, len(ctx))
	copy(result, ctx)

	for range nTokens {
		input := result
		if len(input) > g.cfg.BlockSize {
			input = input[len(input)-g.cfg.BlockSize:]
		}

		logits, _ := g.Forward(input, nil)

		// logits has shape [1, T, V]; take the last position.
		V := g.cfg.VocabSize
		T := len(input)
		lastLogits := make([]float32, V)
		copy(lastLogits, logits.Value.Data[(T-1)*V:T*V])

		// Temperature scaling.
		if temperature != 1.0 {
			for i := range lastLogits {
				lastLogits[i] /= temperature
			}
		}

		// Top-k filtering.
		if topK > 0 && topK < V {
			topKFilter(lastLogits, topK)
		}

		// Sample from softmax distribution.
		probs := softmax1D(lastLogits)
		next := sampleMultinomial(probs, rng)
		result = append(result, next)
	}
	return result
}

// ConfigureOptimizer builds an AdamW optimizer with the standard GPT weight decay scheme:
// 2D+ parameters (matmul weights, embeddings) get weight decay; 1D get none.
func (g *GPT) ConfigureOptimizer(lr, weightDecay, beta1, beta2 float32) *optim.AdamW {
	var decay, noDecay []*autograd.Node
	for _, p := range g.Parameters() {
		if len(p.Value.Shape) >= 2 {
			decay = append(decay, p)
		} else {
			noDecay = append(noDecay, p)
		}
	}
	fmt.Printf("optimizer: %d decay params, %d no-decay params\n", len(decay), len(noDecay))
	return optim.NewAdamW(
		optim.ParamGroup{Params: decay, LR: lr, WeightDecay: weightDecay, Beta1: beta1, Beta2: beta2, Eps: 1e-8},
		optim.ParamGroup{Params: noDecay, LR: lr, WeightDecay: 0, Beta1: beta1, Beta2: beta2, Eps: 1e-8},
	)
}

// ---- generation helpers ----

func topKFilter(logits []float32, k int) {
	// Find the k-th largest value and set all smaller values to -inf.
	tmp := make([]float32, len(logits))
	copy(tmp, logits)
	// Partial sort: find threshold.
	threshold := kthLargest(tmp, k)
	negInf := float32(math.Inf(-1))
	for i, v := range logits {
		if v < threshold {
			logits[i] = negInf
		}
	}
}

func kthLargest(s []float32, k int) float32 {
	// Simple selection: O(n*k) but k is small.
	used := make([]bool, len(s))
	var last float32
	for i := 0; i < k; i++ {
		max := float32(math.Inf(-1))
		maxIdx := -1
		for j, v := range s {
			if !used[j] && v > max {
				max = v
				maxIdx = j
			}
		}
		used[maxIdx] = true
		last = max
	}
	return last
}

func softmax1D(logits []float32) []float32 {
	maxV := float32(math.Inf(-1))
	for _, v := range logits {
		if v > maxV {
			maxV = v
		}
	}
	probs := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		e := float32(math.Exp(float64(v - maxV)))
		probs[i] = e
		sum += float64(e)
	}
	for i := range probs {
		probs[i] /= float32(sum)
	}
	return probs
}

func sampleMultinomial(probs []float32, rng *rand.Rand) int {
	r := rng.Float32()
	var cum float32
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(probs) - 1
}

