package model

import (
	"math/rand"
	"testing"

	"github.com/fieldingj/mingpt/autograd"
	"github.com/fieldingj/mingpt/nn"
)

func benchGPT() (*GPT, []int, []int) {
	nn.RNG = rand.New(rand.NewSource(42))
	cfg := GPTConfig{
		VocabSize: 64,
		BlockSize: 32,
		NEmbd:     64,
		NHead:     4,
		NLayer:    2,
	}
	g := NewGPT(cfg)
	B, T := 4, 32
	rng := rand.New(rand.NewSource(1))
	tokens := make([]int, B*T)
	targets := make([]int, B*T)
	for i := range tokens {
		tokens[i] = rng.Intn(cfg.VocabSize)
		targets[i] = rng.Intn(cfg.VocabSize)
	}
	return g, tokens, targets
}

func BenchmarkGPTForward(b *testing.B) {
	g, tokens, targets := benchGPT()
	g.Train()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = g.Forward(tokens, targets)
	}
}

func BenchmarkGPTForwardBackward(b *testing.B) {
	g, tokens, targets := benchGPT()
	g.Train()
	opt := g.ConfigureOptimizer(1e-3, 0.1, 0.9, 0.95)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		opt.ZeroGrad()
		_, loss := g.Forward(tokens, targets)
		autograd.Backward(loss)
	}
}
