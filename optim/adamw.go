package optim

import (
	"math"

	"github.com/jpfielding/mingpt/autograd"
)

// ParamGroup is a set of parameters with shared optimizer hyperparameters.
type ParamGroup struct {
	Params      []*autograd.Node
	LR          float32
	WeightDecay float32
	Beta1       float32
	Beta2       float32
	Eps         float32
}

// AdamW is the decoupled weight-decay variant of Adam.
type AdamW struct {
	groups []ParamGroup
	step   int
	m      map[*autograd.Node][]float32 // first moment
	v      map[*autograd.Node][]float32 // second moment
}

// NewAdamW creates an AdamW optimizer from one or more parameter groups.
func NewAdamW(groups ...ParamGroup) *AdamW {
	aw := &AdamW{
		groups: groups,
		m:      make(map[*autograd.Node][]float32),
		v:      make(map[*autograd.Node][]float32),
	}
	for _, g := range groups {
		if g.Eps == 0 {
			g.Eps = 1e-8
		}
		for _, p := range g.Params {
			n := p.Value.Numel()
			aw.m[p] = make([]float32, n)
			aw.v[p] = make([]float32, n)
		}
	}
	return aw
}

// ZeroGrad zeroes gradients for all parameters.
func (aw *AdamW) ZeroGrad() {
	for _, g := range aw.groups {
		for _, p := range g.Params {
			p.ZeroGrad()
		}
	}
}

// AllParams returns all parameter nodes across all groups.
func (aw *AdamW) AllParams() []*autograd.Node {
	var all []*autograd.Node
	for _, g := range aw.groups {
		all = append(all, g.Params...)
	}
	return all
}

// Step applies one AdamW update after gradients have been accumulated via Backward.
func (aw *AdamW) Step() {
	aw.step++
	t := float64(aw.step)

	for _, g := range aw.groups {
		eps := g.Eps
		if eps == 0 {
			eps = 1e-8
		}
		// Bias corrections.
		bc1 := float32(1 - math.Pow(float64(g.Beta1), t))
		bc2 := float32(1 - math.Pow(float64(g.Beta2), t))

		for _, p := range g.Params {
			if p.Grad == nil {
				continue
			}
			w := p.Value.Data
			grad := p.Grad.Data
			m := aw.m[p]
			v := aw.v[p]

			for i := range w {
				gi := grad[i]

				m[i] = g.Beta1*m[i] + (1-g.Beta1)*gi
				v[i] = g.Beta2*v[i] + (1-g.Beta2)*gi*gi

				mHat := m[i] / bc1
				vHat := v[i] / bc2

				// Decoupled weight decay.
				w[i] *= 1 - g.LR*g.WeightDecay

				// Adam update.
				w[i] -= g.LR * mHat / (float32(math.Sqrt(float64(vHat)))+eps)
			}
		}
	}
}

// ClipGradNorm rescales all gradients so their global L2 norm ≤ maxNorm.
// Returns the pre-clipping norm.
func ClipGradNorm(params []*autograd.Node, maxNorm float32) float32 {
	var sumSq float64
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		for _, g := range p.Grad.Data {
			sumSq += float64(g) * float64(g)
		}
	}
	norm := float32(math.Sqrt(sumSq))
	if norm > maxNorm {
		scale := maxNorm / norm
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for i := range p.Grad.Data {
				p.Grad.Data[i] *= scale
			}
		}
	}
	return norm
}
