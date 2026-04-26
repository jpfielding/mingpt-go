package autograd

import (
	"math"
	"math/rand"

	"github.com/fieldingj/mingpt/tensor"
)

// ---- Transpose ----

// Transpose swaps two dimensions of a tensor, materialising the result.
func Transpose(x *Node, dim0, dim1 int) *Node {
	n := &Node{
		Value:  x.Value.Transpose(dim0, dim1),
		inputs: []*Node{x},
		op:     "transpose",
	}
	n.backward = func() {
		if n.Grad == nil {
			return
		}
		// Transpose is its own inverse.
		accumulateGrad(x, n.Grad.Transpose(dim0, dim1))
	}
	return n
}

// ---- Add ----

// Add adds two nodes element-wise with broadcasting.
func Add(x, y *Node) *Node {
	z := &Node{
		Value:  tensor.ElemAdd(x.Value, y.Value),
		inputs: []*Node{x, y},
		op:     "add",
	}
	z.backward = func() {
		if z.Grad == nil {
			return
		}
		accumulateGrad(x, tensor.SumKeepBroadcastDims(z.Grad, x.Value.Shape))
		accumulateGrad(y, tensor.SumKeepBroadcastDims(z.Grad, y.Value.Shape))
	}
	return z
}

// MulScalar multiplies a node by a scalar constant.
func MulScalar(x *Node, s float32) *Node {
	z := &Node{
		Value:  tensor.Scale(x.Value, s),
		inputs: []*Node{x},
		op:     "mulscalar",
	}
	z.backward = func() {
		if z.Grad == nil {
			return
		}
		accumulateGrad(x, tensor.Scale(z.Grad, s))
	}
	return z
}

// ---- MatMul ----

// MatMul performs batched matrix multiplication.
// Supports 2D, 3D, 4D tensors (see tensor.MatMul for rules).
func MatMul(a, b *Node) *Node {
	c := &Node{
		Value:  tensor.MatMul(a.Value, b.Value),
		inputs: []*Node{a, b},
		op:     "matmul",
	}
	c.backward = func() {
		if c.Grad == nil {
			return
		}
		// dA = dC @ B^T
		bT := transposeLastTwo(b.Value)
		dA := tensor.MatMul(c.Grad, bT)
		accumulateGrad(a, dA)

		// dB = A^T @ dC
		aT := transposeLastTwo(a.Value)
		dB := tensor.MatMul(aT, c.Grad)
		// If b has fewer dimensions (shared weight), sum and squeeze extra batch dims.
		if len(b.Value.Shape) < len(a.Value.Shape) {
			extra := len(dB.Shape) - len(b.Value.Shape)
			for i := 0; i < extra; i++ {
				dB = tensor.SumAxis(dB, 0)
			}
			dB = dB.Reshape(b.Value.Shape...)
		}
		accumulateGrad(b, dB)
	}
	return c
}

// transposeLastTwo swaps the last two dimensions of a tensor.
func transposeLastTwo(t *tensor.Tensor) *tensor.Tensor {
	n := len(t.Shape)
	return t.Transpose(n-2, n-1)
}

// ---- GELU ----

const (
	geluC = float32(0.7978845608028654) // sqrt(2/pi)
	geluA = float32(0.044715)
)

// GELU applies the Gaussian Error Linear Unit activation:
// y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
func GELU(x *Node) *Node {
	// Forward: save intermediate values for backward.
	u := make([]float32, len(x.Value.Data))
	tanhU := make([]float32, len(x.Value.Data))
	y := tensor.Zeros(x.Value.Shape...)

	for i, xi := range x.Value.Data {
		ui := geluC * (xi + geluA*xi*xi*xi)
		ti := float32(math.Tanh(float64(ui)))
		u[i] = ui
		tanhU[i] = ti
		y.Data[i] = 0.5 * xi * (1 + ti)
	}
	_ = u // captured in closure

	n := &Node{Value: y, inputs: []*Node{x}, op: "gelu"}
	n.backward = func() {
		if n.Grad == nil {
			return
		}
		dx := tensor.Zeros(x.Value.Shape...)
		for i, xi := range x.Value.Data {
			ti := tanhU[i]
			// dy/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*c*(1 + 3*a*x^2)
			dydx := 0.5*(1+ti) + 0.5*xi*(1-ti*ti)*geluC*(1+3*geluA*xi*xi)
			dx.Data[i] = n.Grad.Data[i] * dydx
		}
		accumulateGrad(x, dx)
	}
	return n
}

// ---- Softmax ----

// Softmax applies softmax over the last dimension.
func Softmax(x *Node) *Node {
	p := softmaxForward(x.Value)
	n := &Node{Value: p, inputs: []*Node{x}, op: "softmax"}
	n.backward = func() {
		if n.Grad == nil {
			return
		}
		// dx = p * (dp - sum(dp*p, lastDim))
		dpP := tensor.ElemMul(n.Grad, p)
		dot := tensor.SumAxis(dpP, len(p.Shape)-1) // [..., 1]
		sub := tensor.ElemSub(n.Grad, dot)          // broadcast
		dx := tensor.ElemMul(p, sub)
		accumulateGrad(x, dx)
	}
	return n
}

func softmaxForward(x *tensor.Tensor) *tensor.Tensor {
	out := x.Clone()
	lastDim := len(x.Shape) - 1
	stride := x.Shape[lastDim]
	nVecs := x.Numel() / stride

	for v := 0; v < nVecs; v++ {
		base := v * stride
		// Stable: subtract max.
		maxVal := float32(math.Inf(-1))
		for i := 0; i < stride; i++ {
			if out.Data[base+i] > maxVal {
				maxVal = out.Data[base+i]
			}
		}
		var sum float64
		for i := 0; i < stride; i++ {
			e := float32(math.Exp(float64(out.Data[base+i] - maxVal)))
			out.Data[base+i] = e
			sum += float64(e)
		}
		for i := 0; i < stride; i++ {
			out.Data[base+i] /= float32(sum)
		}
	}
	return out
}

// ---- MaskedFillNegInf ----

// MaskedFillNegInf sets positions where mask==0 to -inf.
// mask shape must be [1,1,T,T] and x shape must be [B,H,T,T] (4D).
// The mask is treated as a constant (no gradient).
func MaskedFillNegInf(x *Node, mask *tensor.Tensor) *Node {
	y := x.Value.Clone()
	negInf := float32(math.Inf(-1))

	B := x.Value.Shape[0]
	H := x.Value.Shape[1]
	T := x.Value.Shape[2]

	for b := 0; b < B; b++ {
		for h := 0; h < H; h++ {
			for i := 0; i < T; i++ {
				for j := 0; j < T; j++ {
					if mask.Data[i*T+j] == 0 {
						y.Data[b*H*T*T+h*T*T+i*T+j] = negInf
					}
				}
			}
		}
	}

	n := &Node{Value: y, inputs: []*Node{x}, op: "maskedfill"}
	n.backward = func() {
		if n.Grad == nil {
			return
		}
		// Gradient is zero at masked positions (softmax output is 0 there).
		accumulateGrad(x, n.Grad.Clone())
	}
	return n
}

// ---- LayerNorm ----

// LayerNorm applies layer normalisation over the last dimension.
// weight and bias are 1D Nodes of shape [C].
func LayerNorm(x, weight, bias *Node, eps float32) *Node {
	// Forward: compute mu, invStd, xhat, y and save for backward.
	C := x.Value.Shape[len(x.Value.Shape)-1]
	nVecs := x.Value.Numel() / C

	xhat := tensor.Zeros(x.Value.Shape...)
	mu := make([]float32, nVecs)
	invStd := make([]float32, nVecs)

	for v := 0; v < nVecs; v++ {
		base := v * C
		// Mean (float64 accumulator for stability).
		var sum float64
		for i := 0; i < C; i++ {
			sum += float64(x.Value.Data[base+i])
		}
		mean := float32(sum / float64(C))
		mu[v] = mean

		// Variance.
		var vsum float64
		for i := 0; i < C; i++ {
			d := float64(x.Value.Data[base+i] - mean)
			vsum += d * d
		}
		variance := float32(vsum / float64(C))
		is := float32(1.0 / math.Sqrt(float64(variance+eps)))
		invStd[v] = is

		for i := 0; i < C; i++ {
			xhat.Data[base+i] = (x.Value.Data[base+i] - mean) * is
		}
	}

	y := tensor.Zeros(x.Value.Shape...)
	for v := 0; v < nVecs; v++ {
		base := v * C
		for i := 0; i < C; i++ {
			y.Data[base+i] = xhat.Data[base+i]*weight.Value.Data[i] + bias.Value.Data[i]
		}
	}

	n := &Node{Value: y, inputs: []*Node{x, weight, bias}, op: "layernorm"}
	n.backward = func() {
		if n.Grad == nil {
			return
		}
		dX := tensor.Zeros(x.Value.Shape...)
		dW := tensor.Zeros(weight.Value.Shape...)
		dB := tensor.Zeros(bias.Value.Shape...)

		for v := 0; v < nVecs; v++ {
			base := v * C
			is := invStd[v]

			// dW += sum(dy * xhat)  over the positions this row contributes to.
			// dB += sum(dy)
			for i := 0; i < C; i++ {
				dy := n.Grad.Data[base+i]
				dW.Data[i] += dy * xhat.Data[base+i]
				dB.Data[i] += dy
			}

			// dx_hat = dy * w
			// dsigma2 = sum(dx_hat * (x - mu) * -0.5 * is^3)
			// dmu = sum(-dx_hat * is) + dsigma2 * sum(-2*(x-mu)) / C
			// dx = dx_hat * is + dsigma2 * 2*(x-mu)/C + dmu/C

			var dsigma2, dmu float64
			for i := 0; i < C; i++ {
				dxhat := n.Grad.Data[base+i] * weight.Value.Data[i]
				xmmu := x.Value.Data[base+i] - mu[v]
				dsigma2 += float64(dxhat) * float64(xmmu) * float64(-0.5*is*is*is)
				dmu += float64(-dxhat * is)
			}
			// Correction to dmu from dsigma2.
			var sumXmmu float64
			for i := 0; i < C; i++ {
				sumXmmu += float64(x.Value.Data[base+i] - mu[v])
			}
			dmu += dsigma2 * (-2.0 * sumXmmu / float64(C))

			for i := 0; i < C; i++ {
				dxhat := float64(n.Grad.Data[base+i] * weight.Value.Data[i])
				xmmu := float64(x.Value.Data[base+i] - mu[v])
				dxi := dxhat*float64(is) + dsigma2*2.0*xmmu/float64(C) + dmu/float64(C)
				dX.Data[base+i] = float32(dxi)
			}
		}

		accumulateGrad(x, dX)
		accumulateGrad(weight, dW)
		accumulateGrad(bias, dB)
	}
	return n
}

// ---- Embedding ----

// Embedding gathers rows from weight according to idx.
// weight: [VocabSize, C], idx: length B*T, output: [B, T, C].
func Embedding(weight *Node, idx []int, B, T int) *Node {
	C := weight.Value.Shape[1]
	y := tensor.Zeros(B, T, C)
	for i, id := range idx {
		copy(y.Data[i*C:(i+1)*C], weight.Value.Data[id*C:(id+1)*C])
	}

	n := &Node{Value: y, inputs: []*Node{weight}, op: "embedding"}
	n.backward = func() {
		if n.Grad == nil {
			return
		}
		dW := tensor.Zeros(weight.Value.Shape...)
		for i, id := range idx {
			for c := 0; c < C; c++ {
				dW.Data[id*C+c] += n.Grad.Data[i*C+c]
			}
		}
		accumulateGrad(weight, dW)
	}
	return n
}

// ---- CrossEntropy ----

// CrossEntropyLoss computes fused softmax + NLL loss.
// logits: [B, T, V], targets: []int of length B*T.
// Returns a scalar node.
func CrossEntropyLoss(logits *Node, targets []int) *Node {
	B, T, V := logits.Value.Shape[0], logits.Value.Shape[1], logits.Value.Shape[2]
	N := B * T

	// Compute probabilities and loss simultaneously.
	p := make([]float32, N*V)
	var totalLoss float64

	for n := 0; n < N; n++ {
		base := n * V
		// Stable softmax: subtract max.
		maxV := float32(math.Inf(-1))
		for v := 0; v < V; v++ {
			if logits.Value.Data[base+v] > maxV {
				maxV = logits.Value.Data[base+v]
			}
		}
		var sum float64
		for v := 0; v < V; v++ {
			e := float32(math.Exp(float64(logits.Value.Data[base+v] - maxV)))
			p[base+v] = e
			sum += float64(e)
		}
		for v := 0; v < V; v++ {
			p[base+v] /= float32(sum)
		}
		totalLoss -= math.Log(float64(p[base+targets[n]]) + 1e-9)
	}

	loss := float32(totalLoss / float64(N))
	lossNode := &Node{
		Value:  tensor.Scalar(loss),
		inputs: []*Node{logits},
		op:     "xent",
	}
	lossNode.backward = func() {
		if lossNode.Grad == nil {
			return
		}
		scale := lossNode.Grad.Data[0] / float32(N)
		dLogits := tensor.Zeros(B, T, V)
		for n := 0; n < N; n++ {
			base := n * V
			for v := 0; v < V; v++ {
				dLogits.Data[base+v] = p[base+v] * scale
			}
			dLogits.Data[base+targets[n]] -= scale
		}
		accumulateGrad(logits, dLogits)
	}
	return lossNode
}

// ---- Dropout ----

// Dropout applies inverted dropout during training.
// During eval (training=false), returns x unchanged (no new node).
func Dropout(x *Node, p float32, training bool, rng *rand.Rand) *Node {
	if !training || p == 0 {
		return x
	}
	scale := float32(1.0 / float64(1-p))
	mask := tensor.Zeros(x.Value.Shape...)
	for i := range mask.Data {
		if rng.Float32() >= p {
			mask.Data[i] = scale
		}
	}

	y := tensor.ElemMul(x.Value, mask)
	n := &Node{Value: y, inputs: []*Node{x}, op: "dropout"}
	n.backward = func() {
		if n.Grad == nil {
			return
		}
		accumulateGrad(x, tensor.ElemMul(n.Grad, mask))
	}
	return n
}

