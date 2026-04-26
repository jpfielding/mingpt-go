package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/fieldingj/mingpt/tensor"
)

// gradCheck verifies analytical gradient against numerical central differences.
// f should take a single parameter node and return a scalar node.
func gradCheck(t *testing.T, name string, f func(p *Node) *Node, p *Node, eps float32) {
	t.Helper()

	// Analytical.
	out := f(p)
	Backward(out)
	analytical := make([]float32, len(p.Grad.Data))
	copy(analytical, p.Grad.Data)

	// Numerical.
	for i := range p.Value.Data {
		orig := p.Value.Data[i]

		p.Value.Data[i] = orig + eps
		p.Grad = nil
		fPlus := f(p).Value.Data[0]

		p.Value.Data[i] = orig - eps
		p.Grad = nil
		fMinus := f(p).Value.Data[0]

		p.Value.Data[i] = orig

		num := (fPlus - fMinus) / (2 * eps)
		ana := analytical[i]

		absErr := abs32(ana - num)
		// Skip near-zero gradients — both analytical and numerical should be negligible.
		if abs32(ana)+abs32(num) < 1e-4 {
			continue
		}
		relErr := absErr / (abs32(num) + abs32(ana) + 1e-7)
		if relErr > 1e-2 {
			t.Errorf("%s: param[%d] analytical=%.6f numerical=%.6f rel_err=%.4f",
				name, i, ana, num, relErr)
		}
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func makeTensor(shape []int, rng *rand.Rand) *tensor.Tensor {
	t := tensor.Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = float32(rng.NormFloat64()) * 0.5
	}
	return t
}

func TestGELUBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	p := Param(makeTensor([]int{3, 4}, rng))
	gradCheck(t, "gelu", func(p *Node) *Node {
		out := GELU(p)
		// Return mean as scalar.
		s := float32(0)
		for _, v := range out.Value.Data {
			s += v
		}
		res := &Node{Value: tensor.Scalar(s / float32(out.Value.Numel())), inputs: []*Node{out}, op: "mean"}
		res.backward = func() {
			if res.Grad == nil {
				return
			}
			scale := res.Grad.Data[0] / float32(out.Value.Numel())
			accumulateGrad(out, tensor.Full(scale, out.Value.Shape...))
		}
		return res
	}, p, 1e-3)
}

func TestSoftmaxBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	p := Param(makeTensor([]int{2, 4}, rng))
	gradCheck(t, "softmax", func(p *Node) *Node {
		out := Softmax(p)
		s := float32(0)
		for _, v := range out.Value.Data {
			s += v
		}
		res := &Node{Value: tensor.Scalar(s / float32(out.Value.Numel())), inputs: []*Node{out}, op: "mean"}
		res.backward = func() {
			if res.Grad == nil {
				return
			}
			scale := res.Grad.Data[0] / float32(out.Value.Numel())
			accumulateGrad(out, tensor.Full(scale, out.Value.Shape...))
		}
		return res
	}, p, 1e-3)
}

func TestMatMulBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	a := Param(makeTensor([]int{2, 3, 4}, rng))
	b := Param(makeTensor([]int{4, 5}, rng))

	// Test gradient for a.
	gradCheck(t, "matmul/a", func(a *Node) *Node {
		out := MatMul(a, b)
		s := float32(0)
		for _, v := range out.Value.Data {
			s += v
		}
		res := scalarWrap(out, s)
		return res
	}, a, 1e-3)

	// Test gradient for b.
	a.Grad = nil
	b.Grad = nil
	gradCheck(t, "matmul/b", func(b *Node) *Node {
		out := MatMul(a, b)
		s := float32(0)
		for _, v := range out.Value.Data {
			s += v
		}
		return scalarWrap(out, s)
	}, b, 1e-3)
}

func TestLayerNormBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	C := 8
	x := Param(makeTensor([]int{2, 4, C}, rng))
	w := Param(tensor.Ones(C))
	b := Param(tensor.Zeros(C))

	gradCheck(t, "layernorm/x", func(x *Node) *Node {
		out := LayerNorm(x, w, b, 1e-5)
		return sumScalar(out)
	}, x, 1e-3)
}

func TestEmbeddingBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	vocab, dim := 5, 4
	w := Param(makeTensor([]int{vocab, dim}, rng))
	idx := []int{0, 2, 1, 2} // B=1, T=4

	gradCheck(t, "embedding/w", func(w *Node) *Node {
		out := Embedding(w, idx, 1, 4)
		return sumScalar(out)
	}, w, 1e-3)
}

func TestCrossEntropyBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	B, T, V := 2, 3, 5
	logits := Param(makeTensor([]int{B, T, V}, rng))
	targets := []int{1, 2, 0, 3, 1, 4}

	gradCheck(t, "xent/logits", func(logits *Node) *Node {
		return CrossEntropyLoss(logits, targets)
	}, logits, 1e-3)
}

func TestAddBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	x := Param(makeTensor([]int{2, 3}, rng))
	y := Param(makeTensor([]int{2, 3}, rng))

	gradCheck(t, "add/x", func(x *Node) *Node {
		out := Add(x, y)
		return sumScalar(out)
	}, x, 1e-3)
}

func TestDropoutBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	x := Param(makeTensor([]int{4, 4}, rng))
	dropRng := rand.New(rand.NewSource(1))

	gradCheck(t, "dropout/x", func(x *Node) *Node {
		dropRng.Seed(99) // same mask each call
		out := Dropout(x, 0.3, true, dropRng)
		return sumScalar(out)
	}, x, 1e-3)
}

func TestMaskedFillNegInfForward(t *testing.T) {
	// Verify that masked positions produce ~0 probability after softmax.
	rng := rand.New(rand.NewSource(42))
	T := 4
	x := Param(makeTensor([]int{1, 1, T, T}, rng))

	// Build lower-triangular mask.
	mask := tensor.Zeros(T, T)
	for i := 0; i < T; i++ {
		for j := 0; j <= i; j++ {
			mask.Data[i*T+j] = 1
		}
	}

	out := MaskedFillNegInf(x, mask)
	probs := Softmax(out)

	for i := 0; i < T; i++ {
		for j := i + 1; j < T; j++ {
			v := probs.Value.Data[i*T+j]
			if math.Abs(float64(v)) > 1e-5 {
				t.Errorf("masked position (%d,%d) has non-zero prob %.6f", i, j, v)
			}
		}
	}
}

// helpers

func scalarWrap(n *Node, s float32) *Node {
	res := &Node{Value: tensor.Scalar(s / float32(n.Value.Numel())), inputs: []*Node{n}, op: "mean"}
	res.backward = func() {
		if res.Grad == nil {
			return
		}
		scale := res.Grad.Data[0] / float32(n.Value.Numel())
		accumulateGrad(n, tensor.Full(scale, n.Value.Shape...))
	}
	return res
}

func sumScalar(n *Node) *Node {
	s := float32(0)
	for _, v := range n.Value.Data {
		s += v
	}
	return scalarWrap(n, s)
}
