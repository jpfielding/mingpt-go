package nn

import (
	"github.com/jpfielding/mingpt/autograd"
	"github.com/jpfielding/mingpt/tensor"
)

// Embedding is a lookup table mapping integer IDs to dense vectors.
type Embedding struct {
	Weight   *autograd.Node // [VocabSize, Dim]
	training bool
}

// NewEmbedding creates an Embedding layer with normal(0, std) initialisation.
func NewEmbedding(vocabSize, dim int, std float32) *Embedding {
	return &Embedding{
		Weight:   autograd.Param(tensor.RandNorm(RNG, std, vocabSize, dim)),
		training: true,
	}
}

// Forward gathers rows for the given token indices.
// idx must be provided separately (not as a Node) since indices are not differentiable.
// B and T are the batch and sequence dimensions.
func (e *Embedding) Lookup(idx []int, B, T int) *autograd.Node {
	return autograd.Embedding(e.Weight, idx, B, T)
}

// Forward satisfies the Module interface but panics — use Lookup instead.
func (e *Embedding) Forward(inputs ...*autograd.Node) *autograd.Node {
	panic("Embedding.Forward: use Lookup(idx, B, T) instead")
}

func (e *Embedding) Parameters() []*autograd.Node { return []*autograd.Node{e.Weight} }
func (e *Embedding) SetTraining(training bool)     { e.training = training }
