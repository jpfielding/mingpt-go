package model

import (
	"encoding/gob"
	"fmt"
	"io"
	"os"
)

// checkpoint is the on-disk format: a GPTConfig plus flat parameter data.
// Parameter order is the fixed iteration order of (*GPT).Parameters().
type checkpoint struct {
	Version int
	Config  GPTConfig
	Params  []paramData
}

type paramData struct {
	Shape []int
	Data  []float32
}

const checkpointVersion = 1

// Save writes the model's config and all parameters to path in gob format.
func (g *GPT) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create checkpoint: %w", err)
	}
	defer f.Close()
	return g.SaveTo(f)
}

// SaveTo writes to an io.Writer (useful for tests that want a buffer).
func (g *GPT) SaveTo(w io.Writer) error {
	params := g.Parameters()
	ck := checkpoint{
		Version: checkpointVersion,
		Config:  g.cfg,
		Params:  make([]paramData, len(params)),
	}
	for i, p := range params {
		shape := make([]int, len(p.Value.Shape))
		copy(shape, p.Value.Shape)
		data := make([]float32, len(p.Value.Data))
		copy(data, p.Value.Data)
		ck.Params[i] = paramData{Shape: shape, Data: data}
	}
	if err := gob.NewEncoder(w).Encode(ck); err != nil {
		return fmt.Errorf("encode checkpoint: %w", err)
	}
	return nil
}

// LoadGPT reads a checkpoint and returns a new model. The config stored in
// the file is used to construct the model; pass nn.RNG as needed before
// calling so the parameter ordering is identical to when it was saved.
func LoadGPT(path string) (*GPT, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open checkpoint: %w", err)
	}
	defer f.Close()
	return LoadGPTFrom(f)
}

// LoadGPTFrom reads a checkpoint from r.
func LoadGPTFrom(r io.Reader) (*GPT, error) {
	var ck checkpoint
	if err := gob.NewDecoder(r).Decode(&ck); err != nil {
		return nil, fmt.Errorf("decode checkpoint: %w", err)
	}
	if ck.Version != checkpointVersion {
		return nil, fmt.Errorf("checkpoint version %d, this build expects %d", ck.Version, checkpointVersion)
	}

	g := NewGPT(ck.Config)
	params := g.Parameters()
	if len(params) != len(ck.Params) {
		return nil, fmt.Errorf("param count mismatch: checkpoint=%d model=%d", len(ck.Params), len(params))
	}
	for i, p := range params {
		pd := ck.Params[i]
		if len(pd.Data) != len(p.Value.Data) {
			return nil, fmt.Errorf("param %d size mismatch: checkpoint=%d model=%d", i, len(pd.Data), len(p.Value.Data))
		}
		if len(pd.Shape) != len(p.Value.Shape) {
			return nil, fmt.Errorf("param %d shape rank mismatch", i)
		}
		for d := range pd.Shape {
			if pd.Shape[d] != p.Value.Shape[d] {
				return nil, fmt.Errorf("param %d shape mismatch: checkpoint=%v model=%v", i, pd.Shape, p.Value.Shape)
			}
		}
		copy(p.Value.Data, pd.Data)
	}
	return g, nil
}
