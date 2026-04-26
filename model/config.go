package model

// GPTConfig holds all hyperparameters for a GPT model.
type GPTConfig struct {
	VocabSize int
	BlockSize int // maximum sequence length

	NEmbd  int // embedding dimension
	NHead  int // number of attention heads (NEmbd must be divisible by NHead)
	NLayer int // number of transformer blocks

	EmbdDrop  float32 // dropout on token+position embeddings
	ResidDrop float32 // dropout on residual path
	AttnDrop  float32 // dropout on attention weights
}

// Preset configurations matching karpathy's model_type shortcuts.
var (
	GPTNano  = GPTConfig{NLayer: 3, NHead: 3, NEmbd: 48}
	GPTMicro = GPTConfig{NLayer: 4, NHead: 4, NEmbd: 128}
	GPTMini  = GPTConfig{NLayer: 6, NHead: 6, NEmbd: 192}
	GPT1     = GPTConfig{NLayer: 12, NHead: 12, NEmbd: 768}
)

// Preset returns a config preset by name (gpt-nano, gpt-micro, gpt-mini, gpt1).
func Preset(name string) GPTConfig {
	switch name {
	case "gpt-nano":
		return GPTNano
	case "gpt-micro":
		return GPTMicro
	case "gpt-mini":
		return GPTMini
	case "gpt1":
		return GPT1
	default:
		return GPTNano
	}
}
