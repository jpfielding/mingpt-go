package model

import (
	"math"

	"github.com/fieldingj/mingpt/autograd"
	"github.com/fieldingj/mingpt/nn"
)

// Block is a single transformer layer: LN → Attn → residual + LN → FFN → residual.
type Block struct {
	cfg  GPTConfig
	LN1  *nn.LayerNorm
	Attn *CausalSelfAttention
	LN2  *nn.LayerNorm
	FC   *nn.Linear // FFN expand: NEmbd → 4*NEmbd
	Proj *nn.Linear // FFN contract: 4*NEmbd → NEmbd
	Drop *nn.Dropout
}

// NewBlock creates a transformer block.
// residStd is the scaled init std for residual projections (1/sqrt(2*NLayer)).
func NewBlock(cfg GPTConfig, residStd float32) *Block {
	return &Block{
		cfg:  cfg,
		LN1:  nn.NewLayerNorm(cfg.NEmbd, 1e-5),
		Attn: NewCausalSelfAttention(cfg, residStd),
		LN2:  nn.NewLayerNorm(cfg.NEmbd, 1e-5),
		FC:   nn.NewLinear(cfg.NEmbd, 4*cfg.NEmbd, true, 0.02),
		Proj: nn.NewLinear(4*cfg.NEmbd, cfg.NEmbd, true, residStd),
		Drop: nn.NewDropout(cfg.ResidDrop),
	}
}

// defaultResidStd returns the GPT-2 paper residual init std for NLayer blocks.
func defaultResidStd(nLayer int) float32 {
	return float32(0.02 / math.Sqrt(float64(2*nLayer)))
}

// Forward applies x → x + Attn(LN1(x)) → x + FFN(LN2(x)).
func (b *Block) Forward(x *autograd.Node) *autograd.Node {
	// Attention residual.
	x = autograd.Add(x, b.Attn.Forward(b.LN1.Forward(x)))

	// FFN residual.
	h := b.FC.Forward(b.LN2.Forward(x))
	h = autograd.GELU(h)
	h = b.Proj.Forward(h)
	h = b.Drop.Forward(h)
	x = autograd.Add(x, h)
	return x
}

func (b *Block) Parameters() []*autograd.Node {
	var params []*autograd.Node
	params = append(params, b.LN1.Parameters()...)
	params = append(params, b.Attn.Parameters()...)
	params = append(params, b.LN2.Parameters()...)
	params = append(params, b.FC.Parameters()...)
	params = append(params, b.Proj.Parameters()...)
	return params
}

func (b *Block) SetTraining(training bool) {
	b.LN1.SetTraining(training)
	b.Attn.SetTraining(training)
	b.LN2.SetTraining(training)
	b.FC.SetTraining(training)
	b.Proj.SetTraining(training)
	b.Drop.SetTraining(training)
}
