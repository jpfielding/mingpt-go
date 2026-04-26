package nn

import (
	"math/rand"

	"github.com/fieldingj/mingpt/autograd"
)

// Module is the interface every layer implements.
type Module interface {
	Forward(inputs ...*autograd.Node) *autograd.Node
	Parameters() []*autograd.Node
	SetTraining(training bool)
}

// RNG is the shared random number generator for weight init and dropout.
// Set before building any model.
var RNG = rand.New(rand.NewSource(42))
