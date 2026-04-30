package model

import (
	"math"

	"github.com/jpfielding/mingpt/autograd"
	"github.com/jpfielding/mingpt/nn"
	"github.com/jpfielding/mingpt/tensor"
)

// CausalSelfAttention implements multi-head masked self-attention.
type CausalSelfAttention struct {
	cfg      GPTConfig
	training bool

	CAttn    *nn.Linear // [NEmbd → 3*NEmbd]
	CProj    *nn.Linear // [NEmbd → NEmbd]
	AttnDrop *nn.Dropout
	ResidDrop *nn.Dropout

	// Full causal mask [BlockSize, BlockSize]: 1 where attention is allowed.
	causalMask *tensor.Tensor
	// Cached per-T sub-masks, keyed by sequence length. Training uses a fixed
	// T every iteration so the cache fills immediately after the first batch.
	maskCache map[int]*tensor.Tensor
}

// NewCausalSelfAttention creates the attention layer.
// std is the weight init standard deviation (scaled for residual projections).
func NewCausalSelfAttention(cfg GPTConfig, std float32) *CausalSelfAttention {
	a := &CausalSelfAttention{
		cfg:       cfg,
		training:  true,
		CAttn:     nn.NewLinear(cfg.NEmbd, 3*cfg.NEmbd, true, 0.02),
		CProj:     nn.NewLinear(cfg.NEmbd, cfg.NEmbd, true, std),
		AttnDrop:  nn.NewDropout(cfg.AttnDrop),
		ResidDrop: nn.NewDropout(cfg.ResidDrop),
		causalMask: buildLowerTriangularMask(cfg.BlockSize),
		maskCache:  make(map[int]*tensor.Tensor),
	}
	// Seed the cache with the full-size mask so training (which uses BlockSize
	// every iter) never touches the slow path.
	a.maskCache[cfg.BlockSize] = a.causalMask
	return a
}

// buildLowerTriangularMask returns a [T,T] tensor with 1s on and below the diagonal.
func buildLowerTriangularMask(T int) *tensor.Tensor {
	m := tensor.Zeros(T, T)
	for i := 0; i < T; i++ {
		for j := 0; j <= i; j++ {
			m.Data[i*T+j] = 1
		}
	}
	return m
}

// sliceMask returns the [T,T] causal sub-mask. Cached per T so the second and
// subsequent calls for a given sequence length allocate nothing. For T less
// than BlockSize the first call materialises a copy because rows of the
// [BlockSize,BlockSize] mask are not contiguous for a smaller stride.
func (a *CausalSelfAttention) sliceMask(T int) *tensor.Tensor {
	if m, ok := a.maskCache[T]; ok {
		return m
	}
	full := a.cfg.BlockSize
	m := tensor.Zeros(T, T)
	for i := 0; i < T; i++ {
		copy(m.Data[i*T:(i+1)*T], a.causalMask.Data[i*full:i*full+T])
	}
	a.maskCache[T] = m
	return m
}

// Forward applies causal self-attention.
// x shape: [B, T, C] where C = NEmbd.
func (a *CausalSelfAttention) Forward(x *autograd.Node) *autograd.Node {
	B := x.Value.Shape[0]
	T := x.Value.Shape[1]
	C := x.Value.Shape[2]
	H := a.cfg.NHead
	hs := C / H // head size

	// Combined QKV projection: [B, T, 3C]
	qkv := a.CAttn.Forward(x)

	// Split and reshape Q, K, V to [B, H, T, hs].
	q := reshapeWithGrad(splitLastDimRaw(qkv, 0, C), B, H, T, hs)
	k := reshapeWithGrad(splitLastDimRaw(qkv, C, 2*C), B, H, T, hs)
	v := reshapeWithGrad(splitLastDimRaw(qkv, 2*C, 3*C), B, H, T, hs)

	// Attention scores: [B, H, T, T] = Q @ K^T / sqrt(hs)
	scale := float32(1.0 / math.Sqrt(float64(hs)))
	kT := autograd.Transpose(k, 2, 3)               // [B, H, hs, T]
	att := autograd.MatMul(q, kT)                    // [B, H, T, T]
	att = autograd.MulScalar(att, scale)

	// Apply causal mask.
	mask := a.sliceMask(T) // [T, T]
	att = autograd.MaskedFillNegInf(att, mask)

	// Softmax + attention dropout.
	att = autograd.Softmax(att)
	att = a.AttnDrop.Forward(att)

	// Weighted sum: [B, H, T, T] @ [B, H, T, hs] → [B, H, T, hs]
	y := autograd.MatMul(att, v)

	// Merge heads: [B, H, T, hs] → [B, T, C]
	y = mergeHeadsWithGrad(y, B, H, T, C)

	// Output projection + residual dropout.
	y = a.CProj.Forward(y)
	y = a.ResidDrop.Forward(y)
	return y
}

// splitLastDimRaw extracts a slice along the last dimension [B,T,start:end].
// Backward accumulates gradient into the appropriate slice of qkv.
func splitLastDimRaw(qkv *autograd.Node, start, end int) *autograd.Node {
	B := qkv.Value.Shape[0]
	T := qkv.Value.Shape[1]
	C3 := qkv.Value.Shape[2]
	C := end - start

	// Forward: extract slice.
	out := tensor.Zeros(B, T, C)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			copy(out.Data[(b*T+t)*C:], qkv.Value.Data[(b*T+t)*C3+start:(b*T+t)*C3+end])
		}
	}

	n := autograd.NewNode(out, "split", []*autograd.Node{qkv}, nil)
	n.SetBackwardFn(func() {
		if n.Grad == nil {
			return
		}
		dQKV := tensor.Zeros(B, T, C3)
		for b := 0; b < B; b++ {
			for t := 0; t < T; t++ {
				copy(dQKV.Data[(b*T+t)*C3+start:(b*T+t)*C3+end], n.Grad.Data[(b*T+t)*C:])
			}
		}
		autograd.AccumulateGrad(qkv, dQKV)
	})
	return n
}

// reshapeWithGrad wraps a reshape in a grad-aware node.
func reshapeWithGrad(x *autograd.Node, newShape ...int) *autograd.Node {
	origShape := make([]int, len(x.Value.Shape))
	copy(origShape, x.Value.Shape)
	out := x.Value.Reshape(newShape...)

	n := autograd.NewNode(out, "reshape", []*autograd.Node{x}, nil)
	n.SetBackwardFn(func() {
		if n.Grad == nil {
			return
		}
		autograd.AccumulateGrad(x, n.Grad.Reshape(origShape...))
	})
	return n
}

// mergeHeadsWithGrad converts [B, H, T, hs] → [B, T, C] with gradient.
func mergeHeadsWithGrad(x *autograd.Node, B, H, T, C int) *autograd.Node {
	// Step 1: transpose [B, H, T, hs] → [B, T, H, hs]
	xT := autograd.Transpose(x, 1, 2)
	// Step 2: reshape [B, T, H, hs] → [B, T, C]
	return reshapeWithGrad(xT, B, T, C)
}

func (a *CausalSelfAttention) Parameters() []*autograd.Node {
	var params []*autograd.Node
	params = append(params, a.CAttn.Parameters()...)
	params = append(params, a.CProj.Parameters()...)
	return params
}

func (a *CausalSelfAttention) SetTraining(training bool) {
	a.training = training
	a.AttnDrop.SetTraining(training)
	a.ResidDrop.SetTraining(training)
}
