package trainer

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/jpfielding/mingpt/autograd"
	"github.com/jpfielding/mingpt/model"
	"github.com/jpfielding/mingpt/optim"
)

// Dataset provides batches of (inputs, targets) token ID slices.
type Dataset interface {
	Len() int
	GetBatch(batchSize int, rng *rand.Rand) (inputs []int, targets []int)
}

// Callback is called at eval intervals with access to the Trainer state.
type Callback func(t *Trainer)

// Config holds training hyperparameters.
type Config struct {
	MaxIters     int
	BatchSize    int
	LR           float32
	Beta1        float32
	Beta2        float32
	WeightDecay  float32
	GradNormClip float32
	LogEvery     int
	EvalEvery    int
	Seed         int64
}

// DefaultConfig returns sensible defaults matching karpathy's trainer.
func DefaultConfig() Config {
	return Config{
		MaxIters:     5000,
		BatchSize:    64,
		LR:           3e-4,
		Beta1:        0.9,
		Beta2:        0.95,
		WeightDecay:  0.1,
		GradNormClip: 1.0,
		LogEvery:     10,
		EvalEvery:    500,
		Seed:         3407,
	}
}

// Trainer orchestrates the training loop.
type Trainer struct {
	Cfg      Config
	Model    *model.GPT
	Optim    *optim.AdamW
	Dataset  Dataset
	Iter     int
	LastLoss float32

	callbacks map[string]Callback
	rng       *rand.Rand
}

// New creates a Trainer.
func New(cfg Config, m *model.GPT, ds Dataset) *Trainer {
	o := m.ConfigureOptimizer(cfg.LR, cfg.WeightDecay, cfg.Beta1, cfg.Beta2)
	return &Trainer{
		Cfg:       cfg,
		Model:     m,
		Optim:     o,
		Dataset:   ds,
		rng:       rand.New(rand.NewSource(cfg.Seed)),
		callbacks: make(map[string]Callback),
	}
}

// AddCallback registers a named callback.
func (tr *Trainer) AddCallback(name string, fn Callback) {
	tr.callbacks[name] = fn
}

// Run executes the training loop.
func (tr *Trainer) Run() {
	tr.Model.Train()
	start := time.Now()

	for tr.Iter = 0; tr.Iter < tr.Cfg.MaxIters; tr.Iter++ {
		inputs, targets := tr.Dataset.GetBatch(tr.Cfg.BatchSize, tr.rng)

		tr.Optim.ZeroGrad()

		_, loss := tr.Model.Forward(inputs, targets)

		autograd.Backward(loss)

		allParams := tr.Optim.AllParams()
		optim.ClipGradNorm(allParams, tr.Cfg.GradNormClip)

		tr.Optim.Step()

		tr.LastLoss = loss.Value.Data[0]

		if tr.Cfg.LogEvery > 0 && tr.Iter%tr.Cfg.LogEvery == 0 {
			elapsed := time.Since(start)
			fmt.Printf("iter %5d | loss %.4f | %v\n", tr.Iter, tr.LastLoss, elapsed.Round(time.Millisecond))
		}

		if tr.Cfg.EvalEvery > 0 && tr.Iter > 0 && tr.Iter%tr.Cfg.EvalEvery == 0 {
			if cb, ok := tr.callbacks["on_eval"]; ok {
				tr.Model.Eval()
				cb(tr)
				tr.Model.Train()
			}
		}
	}
}
