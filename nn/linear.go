package nn

import (
	"github.com/fieldingj/mingpt/autograd"
	"github.com/fieldingj/mingpt/tensor"
)

// Linear applies y = x @ weight^T + bias.
type Linear struct {
	Weight   *autograd.Node // [OutFeatures, InFeatures]
	Bias     *autograd.Node // [OutFeatures] or nil
	training bool
}

// NewLinear creates a Linear layer.
// std is the standard deviation for normal weight initialisation.
// Pass std=0 to get zeros (e.g. for the final LM head bias).
func NewLinear(inFeatures, outFeatures int, bias bool, std float32) *Linear {
	var w *tensor.Tensor
	if std > 0 {
		w = tensor.RandNorm(RNG, std, outFeatures, inFeatures)
	} else {
		w = tensor.Zeros(outFeatures, inFeatures)
	}
	l := &Linear{
		Weight:   autograd.Param(w),
		training: true,
	}
	if bias {
		l.Bias = autograd.Param(tensor.Zeros(outFeatures))
	}
	return l
}

// Forward computes x @ weight^T + bias.
// x shape: [..., InFeatures], output: [..., OutFeatures].
func (l *Linear) Forward(inputs ...*autograd.Node) *autograd.Node {
	x := inputs[0]
	// Transpose weight: [OutF, InF] → [InF, OutF]
	wT := autograd.Transpose(l.Weight, 0, 1)
	out := autograd.MatMul(x, wT)
	if l.Bias != nil {
		out = autograd.Add(out, l.Bias)
	}
	return out
}

func (l *Linear) Parameters() []*autograd.Node {
	params := []*autograd.Node{l.Weight}
	if l.Bias != nil {
		params = append(params, l.Bias)
	}
	return params
}

func (l *Linear) SetTraining(training bool) { l.training = training }
