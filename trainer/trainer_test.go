package trainer

import (
	"math/rand"
	"testing"

	"github.com/fieldingj/mingpt/model"
	"github.com/fieldingj/mingpt/nn"
)

// sortDataset implements the classic "sort digits" toy task from
// karpathy/minGPT's demo notebook. Each sample concatenates the unsorted
// digits with their sorted version; the target is the same sequence shifted
// by one so the model learns to predict the sorted suffix.
type sortDataset struct {
	length    int // number of digits
	numDigits int // vocabulary size
}

func (d *sortDataset) Len() int       { return 1_000_000 }
func (d *sortDataset) VocabSize() int { return d.numDigits }
func (d *sortDataset) BlockSize() int { return 2*d.length - 1 }

func (d *sortDataset) GetBatch(batchSize int, rng *rand.Rand) (inputs, targets []int) {
	T := d.BlockSize()
	inputs = make([]int, batchSize*T)
	targets = make([]int, batchSize*T)
	for b := 0; b < batchSize; b++ {
		digits := make([]int, d.length)
		for i := range digits {
			digits[i] = rng.Intn(d.numDigits)
		}
		sorted := make([]int, d.length)
		copy(sorted, digits)
		for i := 0; i < len(sorted); i++ {
			for j := i + 1; j < len(sorted); j++ {
				if sorted[j] < sorted[i] {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}
		full := append(append([]int{}, digits...), sorted...)
		copy(inputs[b*T:(b+1)*T], full[:T])
		copy(targets[b*T:(b+1)*T], full[1:T+1])
	}
	return
}

// TestTrainerConverges runs a tiny GPT on the sort task and verifies that
// loss drops well below the uniform-random baseline of ln(numDigits).
func TestTrainerConverges(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping convergence test in -short mode")
	}

	nn.RNG = rand.New(rand.NewSource(42))

	ds := &sortDataset{length: 6, numDigits: 3}

	cfg := model.GPTNano
	cfg.VocabSize = ds.VocabSize()
	cfg.BlockSize = ds.BlockSize()
	cfg.EmbdDrop = 0
	cfg.ResidDrop = 0
	cfg.AttnDrop = 0
	gpt := model.NewGPT(cfg)

	trCfg := DefaultConfig()
	trCfg.MaxIters = 500
	trCfg.BatchSize = 32
	trCfg.LR = 1e-3
	trCfg.LogEvery = 0
	trCfg.EvalEvery = 0
	trCfg.Seed = 42

	tr := New(trCfg, gpt, ds)
	tr.Run()

	// Uniform-over-3 baseline loss is ln(3) ≈ 1.0986. A learning model on this
	// task should reach well below 0.85 within a few hundred steps; the prefix
	// positions are inherently unpredictable so the asymptote isn't 0.
	const threshold = 0.85
	if tr.LastLoss > threshold {
		t.Fatalf("trainer did not converge: final loss %.4f > %.2f", tr.LastLoss, threshold)
	}
	t.Logf("sort-task final loss after %d iters: %.4f", trCfg.MaxIters, tr.LastLoss)
}
