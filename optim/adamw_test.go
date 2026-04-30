package optim

import (
	"math"
	"testing"

	"github.com/jpfielding/mingpt/autograd"
	"github.com/jpfielding/mingpt/tensor"
)

// TestAdamWMinimisesQuadratic descends on f(x) = (x-3)^2; the minimum is at x=3.
// AdamW without weight decay should reach within 1e-3 of 3 given enough steps.
func TestAdamWMinimisesQuadratic(t *testing.T) {
	x := autograd.Param(tensor.FromSlice([]float32{0}, 1))

	opt := NewAdamW(ParamGroup{
		Params: []*autograd.Node{x},
		LR:     0.1,
		Beta1:  0.9,
		Beta2:  0.999,
		Eps:    1e-8,
	})

	for i := 0; i < 500; i++ {
		opt.ZeroGrad()
		// Forward: f = (x-3)^2. Analytical gradient: 2*(x-3).
		xi := x.Value.Data[0]
		x.Grad = tensor.FromSlice([]float32{2 * (xi - 3)}, 1)
		opt.Step()
	}

	got := x.Value.Data[0]
	if math.Abs(float64(got-3)) > 1e-2 {
		t.Fatalf("AdamW did not converge: x=%.6f (want ≈3)", got)
	}
}

// TestAdamWWeightDecayPullsToZero: with no gradient and only weight decay,
// parameter must shrink monotonically toward zero.
func TestAdamWWeightDecayPullsToZero(t *testing.T) {
	x := autograd.Param(tensor.FromSlice([]float32{1.0}, 1))

	opt := NewAdamW(ParamGroup{
		Params:      []*autograd.Node{x},
		LR:          0.1,
		WeightDecay: 0.5,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
	})

	prev := x.Value.Data[0]
	for i := 0; i < 10; i++ {
		opt.ZeroGrad()
		x.Grad = tensor.Zeros(1)
		opt.Step()
		cur := x.Value.Data[0]
		if cur >= prev {
			t.Fatalf("weight decay did not shrink x: step %d: %f → %f", i, prev, cur)
		}
		prev = cur
	}
}

func TestClipGradNorm(t *testing.T) {
	x := autograd.Param(tensor.FromSlice([]float32{1, 2, 3, 4}, 4))
	x.Grad = tensor.FromSlice([]float32{3, 4, 0, 0}, 4) // ||g|| = 5

	norm := ClipGradNorm([]*autograd.Node{x}, 1.0)
	if math.Abs(float64(norm-5)) > 1e-5 {
		t.Fatalf("pre-clip norm %f, want 5", norm)
	}
	// After clip, ||g|| should be 1, and g is scaled by 1/5.
	var sq float64
	for _, v := range x.Grad.Data {
		sq += float64(v) * float64(v)
	}
	if math.Abs(math.Sqrt(sq)-1.0) > 1e-5 {
		t.Fatalf("post-clip norm = %f, want 1", math.Sqrt(sq))
	}
	if math.Abs(float64(x.Grad.Data[0]-0.6)) > 1e-5 || math.Abs(float64(x.Grad.Data[1]-0.8)) > 1e-5 {
		t.Fatalf("post-clip grads %v", x.Grad.Data)
	}
}

func TestClipGradNormBelowThreshold(t *testing.T) {
	// Norm = 0.5 < maxNorm=1, so no change.
	x := autograd.Param(tensor.FromSlice([]float32{0}, 1))
	x.Grad = tensor.FromSlice([]float32{0.3, 0.4}, 2)
	before := []float32{0.3, 0.4}
	x.Grad = tensor.FromSlice(before, 2)

	ClipGradNorm([]*autograd.Node{x}, 1.0)
	for i, v := range x.Grad.Data {
		if v != before[i] {
			t.Fatalf("ClipGradNorm modified grad when norm < maxNorm")
		}
	}
}
