package nn

import (
	"github.com/fieldingj/mingpt/autograd"
)

// Dropout applies inverted dropout during training.
type Dropout struct {
	P        float32
	training bool
}

// NewDropout creates a Dropout layer with drop probability p.
func NewDropout(p float32) *Dropout {
	return &Dropout{P: p, training: true}
}

func (d *Dropout) Forward(inputs ...*autograd.Node) *autograd.Node {
	return autograd.Dropout(inputs[0], d.P, d.training, RNG)
}

func (d *Dropout) Parameters() []*autograd.Node { return nil }
func (d *Dropout) SetTraining(training bool)     { d.training = training }
