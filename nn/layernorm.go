package nn

import (
	"github.com/jpfielding/mingpt/autograd"
	"github.com/jpfielding/mingpt/tensor"
)

// LayerNorm applies layer normalisation over the last dimension.
type LayerNorm struct {
	Weight   *autograd.Node // [C], initialised to ones
	Bias     *autograd.Node // [C], initialised to zeros
	Eps      float32
	training bool
}

// NewLayerNorm creates a LayerNorm with the given normalised dimension.
func NewLayerNorm(normalizedShape int, eps float32) *LayerNorm {
	return &LayerNorm{
		Weight:   autograd.Param(tensor.Ones(normalizedShape)),
		Bias:     autograd.Param(tensor.Zeros(normalizedShape)),
		Eps:      eps,
		training: true,
	}
}

func (ln *LayerNorm) Forward(inputs ...*autograd.Node) *autograd.Node {
	return autograd.LayerNorm(inputs[0], ln.Weight, ln.Bias, ln.Eps)
}

func (ln *LayerNorm) Parameters() []*autograd.Node {
	return []*autograd.Node{ln.Weight, ln.Bias}
}

func (ln *LayerNorm) SetTraining(training bool) { ln.training = training }
