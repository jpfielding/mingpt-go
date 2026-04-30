package nn

import (
	"math/rand"
	"testing"

	"github.com/jpfielding/mingpt/autograd"
	"github.com/jpfielding/mingpt/tensor"
)

func init() {
	RNG = rand.New(rand.NewSource(42))
}

func TestLinearForwardShape(t *testing.T) {
	l := NewLinear(4, 8, true, 0.02)
	x := autograd.Param(tensor.RandNorm(rand.New(rand.NewSource(1)), 1, 2, 3, 4))
	y := l.Forward(x)
	if !tensor.ShapeEqual(y.Value.Shape, []int{2, 3, 8}) {
		t.Fatalf("output shape %v, want [2,3,8]", y.Value.Shape)
	}
}

func TestLinearBackward(t *testing.T) {
	// Build simple: y = Linear(x), loss = sum(y). Verify gradients are non-zero
	// and shapes match.
	l := NewLinear(3, 5, true, 0.1)
	x := autograd.Param(tensor.RandNorm(rand.New(rand.NewSource(2)), 1, 2, 3))
	y := l.Forward(x)

	loss := autograd.MulScalar(sumAll(y), 1.0)
	autograd.Backward(loss)

	if l.Weight.Grad == nil || !tensor.ShapeEqual(l.Weight.Grad.Shape, l.Weight.Value.Shape) {
		t.Fatalf("weight grad wrong: %v", l.Weight.Grad)
	}
	if l.Bias.Grad == nil || !tensor.ShapeEqual(l.Bias.Grad.Shape, l.Bias.Value.Shape) {
		t.Fatalf("bias grad wrong: %v", l.Bias.Grad)
	}
	// Bias grad must equal 2 (= batch size) for every entry since loss = sum(y) and
	// every output position adds 1 to each bias.
	for _, g := range l.Bias.Grad.Data {
		if g != 2 {
			t.Fatalf("bias grad entries = %v, expected all 2", l.Bias.Grad.Data)
		}
	}
}

func TestEmbeddingLookupAndGrad(t *testing.T) {
	e := NewEmbedding(10, 4, 0.1)
	idx := []int{1, 3, 5, 7}
	y := e.Lookup(idx, 2, 2)
	if !tensor.ShapeEqual(y.Value.Shape, []int{2, 2, 4}) {
		t.Fatalf("shape %v", y.Value.Shape)
	}

	loss := sumAll(y)
	autograd.Backward(loss)

	// Gradient rows should be 1 exactly for idx entries, 0 elsewhere.
	for i := 0; i < 10; i++ {
		for c := 0; c < 4; c++ {
			got := e.Weight.Grad.Data[i*4+c]
			want := float32(0)
			for _, id := range idx {
				if id == i {
					want++
				}
			}
			if got != want {
				t.Fatalf("weight.Grad[%d,%d]=%f, want %f", i, c, got, want)
			}
		}
	}
}

func TestLayerNormIdentityWhenZeroMeanUnitVar(t *testing.T) {
	ln := NewLayerNorm(4, 1e-5)
	// Input already has zero mean / unit variance along last dim.
	x := autograd.Const(tensor.FromSlice([]float32{
		-1.5, -0.5, 0.5, 1.5,
		2, 0, -2, 0,
	}, 2, 4))
	y := ln.Forward(x)
	// With weight=1, bias=0, output should equal (x-mean)/std ≈ input itself
	// (because input is already standardised after accounting for variance).
	// For first row: mean=0, var=(1.5^2 + 0.5^2 + 0.5^2 + 1.5^2)/4 = 5/4 = 1.25;
	// std ≈ 1.118; output ≈ x / 1.118.
	got0 := y.Value.Data[0]
	want0 := float32(-1.5 / 1.118033988749895)
	if absf(got0-want0) > 1e-3 {
		t.Fatalf("LN out[0] = %f, want ≈ %f", got0, want0)
	}
}

func TestDropoutTrainingVsEval(t *testing.T) {
	d := NewDropout(0.5)
	x := autograd.Const(tensor.Ones(100))

	// Eval mode: output == input (no new node, identity).
	d.SetTraining(false)
	yEval := d.Forward(x)
	for _, v := range yEval.Value.Data {
		if v != 1 {
			t.Fatalf("eval dropout changed data: %f", v)
		}
	}

	// Training mode: expected mean ≈ 1 (inverted dropout preserves mean).
	d.SetTraining(true)
	yTrain := d.Forward(x)
	var sum float32
	for _, v := range yTrain.Value.Data {
		sum += v
	}
	mean := sum / float32(len(yTrain.Value.Data))
	// With p=0.5 and 100 elements, mean should be in a wide band around 1.
	if mean < 0.5 || mean > 1.5 {
		t.Fatalf("train dropout mean %f not near 1", mean)
	}
}

// ---- helpers ----

func sumAll(n *autograd.Node) *autograd.Node {
	// Returns a scalar = sum(n) by reducing via CrossEntropy-like mechanics is
	// overkill; instead use MulScalar(n, 1) and a fake reduction.
	// We implement a simple sum op via a custom Node.
	val := float32(0)
	for _, v := range n.Value.Data {
		val += v
	}
	out := autograd.NewNode(tensor.Scalar(val), "sumall", []*autograd.Node{n}, nil)
	out.SetBackwardFn(func() {
		// dL/dn = grad_scalar * ones_like(n)
		g := tensor.Ones(n.Value.Shape...)
		// Multiply by out.Grad scalar.
		if out.Grad != nil {
			s := out.Grad.Data[0]
			for i := range g.Data {
				g.Data[i] *= s
			}
		}
		autograd.AccumulateGrad(n, g)
	})
	return out
}

func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
