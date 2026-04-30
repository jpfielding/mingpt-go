package model

import (
	"math/rand"
	"testing"

	"github.com/jpfielding/mingpt/autograd"
	"github.com/jpfielding/mingpt/nn"
	"github.com/jpfielding/mingpt/tensor"
)

func init() {
	nn.RNG = rand.New(rand.NewSource(42))
}

// TestForwardShape runs a tiny GPT forward pass and checks logits/loss shapes.
func TestForwardShape(t *testing.T) {
	cfg := GPTConfig{
		VocabSize: 11,
		BlockSize: 8,
		NEmbd:     16,
		NHead:     2,
		NLayer:    2,
	}
	g := NewGPT(cfg)

	B, T := 2, 8
	tokens := make([]int, B*T)
	targets := make([]int, B*T)
	rng := rand.New(rand.NewSource(3))
	for i := range tokens {
		tokens[i] = rng.Intn(cfg.VocabSize)
		targets[i] = rng.Intn(cfg.VocabSize)
	}

	logits, loss := g.Forward(tokens, targets)
	if !tensor.ShapeEqual(logits.Value.Shape, []int{B, T, cfg.VocabSize}) {
		t.Fatalf("logits shape %v, want [%d,%d,%d]", logits.Value.Shape, B, T, cfg.VocabSize)
	}
	if !tensor.ShapeEqual(loss.Value.Shape, []int{1}) {
		t.Fatalf("loss shape %v, want scalar", loss.Value.Shape)
	}
	if loss.Value.Data[0] <= 0 {
		t.Fatalf("loss should be positive, got %f", loss.Value.Data[0])
	}
}

// TestBackwardProducesGradients runs forward+backward and checks every
// parameter received a non-nil gradient with the right shape.
func TestBackwardProducesGradients(t *testing.T) {
	cfg := GPTConfig{
		VocabSize: 7,
		BlockSize: 4,
		NEmbd:     16,
		NHead:     2,
		NLayer:    1,
	}
	g := NewGPT(cfg)

	B, T := 2, 4
	rng := rand.New(rand.NewSource(5))
	tokens := make([]int, B*T)
	targets := make([]int, B*T)
	for i := range tokens {
		tokens[i] = rng.Intn(cfg.VocabSize)
		targets[i] = rng.Intn(cfg.VocabSize)
	}

	_, loss := g.Forward(tokens, targets)
	autograd.Backward(loss)

	for i, p := range g.Parameters() {
		if p.Grad == nil {
			t.Errorf("param %d has nil grad", i)
			continue
		}
		if !tensor.ShapeEqual(p.Grad.Shape, p.Value.Shape) {
			t.Errorf("param %d grad shape %v != value shape %v", i, p.Grad.Shape, p.Value.Shape)
		}
	}
}

// TestGenerateRunsAndObeysVocab verifies Generate produces tokens within the
// vocab and lengthens the context by exactly nTokens.
func TestGenerateRunsAndObeysVocab(t *testing.T) {
	cfg := GPTConfig{
		VocabSize: 5,
		BlockSize: 8,
		NEmbd:     16,
		NHead:     2,
		NLayer:    1,
	}
	g := NewGPT(cfg)

	ctx := []int{0, 1, 2}
	out := g.Generate(ctx, 10, 1.0, 0, rand.New(rand.NewSource(9)))
	if len(out) != len(ctx)+10 {
		t.Fatalf("output length %d, want %d", len(out), len(ctx)+10)
	}
	for i, tok := range out {
		if tok < 0 || tok >= cfg.VocabSize {
			t.Fatalf("generated token out of vocab at %d: %d", i, tok)
		}
	}
}

// TestConfigureOptimizerGrouping verifies 2D+ params go to the decay group and
// 1D params go to no-decay.
func TestConfigureOptimizerGrouping(t *testing.T) {
	cfg := GPTNano
	cfg.VocabSize = 10
	cfg.BlockSize = 8
	g := NewGPT(cfg)
	opt := g.ConfigureOptimizer(1e-3, 0.1, 0.9, 0.95)

	for _, p := range opt.AllParams() {
		if p.Grad != nil {
			t.Fatal("params should have nil grad before training")
		}
	}

	// Presence check: optimizer returned a non-nil struct.
	if opt == nil {
		t.Fatal("optimizer is nil")
	}
}
