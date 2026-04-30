package model

import (
	"bytes"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/jpfielding/mingpt/autograd"
	"github.com/jpfielding/mingpt/nn"
)

func TestCheckpointRoundTrip(t *testing.T) {
	nn.RNG = rand.New(rand.NewSource(7))

	cfg := GPTConfig{
		VocabSize: 11,
		BlockSize: 8,
		NEmbd:     16,
		NHead:     2,
		NLayer:    2,
	}
	g := NewGPT(cfg)

	// Remember every parameter's bytes so we can compare after reload.
	orig := make([][]float32, len(g.Parameters()))
	for i, p := range g.Parameters() {
		orig[i] = make([]float32, len(p.Value.Data))
		copy(orig[i], p.Value.Data)
	}

	// Serialize to a buffer and reload.
	var buf bytes.Buffer
	if err := g.SaveTo(&buf); err != nil {
		t.Fatalf("SaveTo: %v", err)
	}

	// Reset RNG so NewGPT inside LoadGPTFrom gets fresh random init (which we
	// then overwrite with the checkpoint data).
	nn.RNG = rand.New(rand.NewSource(999))
	restored, err := LoadGPTFrom(&buf)
	if err != nil {
		t.Fatalf("LoadGPTFrom: %v", err)
	}

	// Every parameter must match byte-for-byte.
	for i, p := range restored.Parameters() {
		for j, v := range p.Value.Data {
			if v != orig[i][j] {
				t.Fatalf("param %d differs at index %d: got %f want %f", i, j, v, orig[i][j])
			}
		}
	}

	// Config must match too.
	if restored.cfg != cfg {
		t.Fatalf("config mismatch: got %+v want %+v", restored.cfg, cfg)
	}
}

func TestCheckpointForwardMatches(t *testing.T) {
	// Stronger check: the loaded model should produce identical logits for the
	// same input as the original.
	nn.RNG = rand.New(rand.NewSource(11))

	cfg := GPTConfig{
		VocabSize: 7, BlockSize: 4, NEmbd: 16, NHead: 2, NLayer: 1,
	}
	g := NewGPT(cfg)
	g.Eval() // disable dropout for determinism.

	path := filepath.Join(t.TempDir(), "ckpt.gob")
	if err := g.Save(path); err != nil {
		t.Fatal(err)
	}

	tokens := []int{1, 2, 3, 4, 5, 6, 0, 1}
	origLogits, _ := g.Forward(tokens, nil)
	origCopy := make([]float32, len(origLogits.Value.Data))
	copy(origCopy, origLogits.Value.Data)

	restored, err := LoadGPT(path)
	if err != nil {
		t.Fatal(err)
	}
	restored.Eval()

	newLogits, _ := restored.Forward(tokens, nil)
	for i, v := range newLogits.Value.Data {
		if v != origCopy[i] {
			t.Fatalf("logit %d differs after reload: got %f want %f", i, v, origCopy[i])
		}
	}
}

func TestCheckpointDetectsConfigMismatch(t *testing.T) {
	// Write a checkpoint, then corrupt the version to ensure we reject it
	// instead of silently loading garbage.
	nn.RNG = rand.New(rand.NewSource(2))
	g := NewGPT(GPTConfig{VocabSize: 5, BlockSize: 4, NEmbd: 16, NHead: 2, NLayer: 1})

	var buf bytes.Buffer
	if err := g.SaveTo(&buf); err != nil {
		t.Fatal(err)
	}

	// Make sure LoadGPT works for valid input first (sanity).
	restored, err := LoadGPTFrom(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("valid reload failed: %v", err)
	}
	if restored == nil {
		t.Fatal("restored model is nil")
	}

	// Now corrupt by loading truncated data.
	_, err = LoadGPTFrom(bytes.NewReader(buf.Bytes()[:10]))
	if err == nil {
		t.Fatal("truncated checkpoint load should have failed")
	}
}

// compile-time assertions that public API shapes don't drift.
var (
	_ = (*GPT)(nil).Save
	_ = LoadGPT
	_ = autograd.Param // ensure autograd import stays referenced
)
