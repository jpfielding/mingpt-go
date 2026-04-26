package tokenizer

import (
	"math/rand"
	"testing"
)

func TestCharTokenizerRoundTrip(t *testing.T) {
	text := "hello, world! 123"
	tok := NewCharTokenizer(text)

	ids := tok.Encode(text)
	back := tok.Decode(ids)

	if back != text {
		t.Fatalf("round-trip mismatch:\n  in:  %q\n  out: %q", text, back)
	}
}

func TestCharTokenizerVocabDeterministic(t *testing.T) {
	// Running twice on the same text must produce the same mapping (chars are
	// sorted internally for determinism).
	text := "abracadabra"
	a := NewCharTokenizer(text)
	b := NewCharTokenizer(text)
	if a.VocabSize() != b.VocabSize() {
		t.Fatalf("vocab sizes differ: %d vs %d", a.VocabSize(), b.VocabSize())
	}
	for r, id := range a.Stoi {
		if b.Stoi[r] != id {
			t.Fatalf("char %q: a=%d b=%d", r, id, b.Stoi[r])
		}
	}
}

func TestCharDatasetGetBatchShape(t *testing.T) {
	text := "abcdefghijklmnopqrstuvwxyz"
	tok := NewCharTokenizer(text)
	ds := NewCharDataset(text, 4, tok)

	rng := rand.New(rand.NewSource(1))
	inputs, targets := ds.GetBatch(3, rng)
	if len(inputs) != 3*4 {
		t.Fatalf("inputs len %d, want 12", len(inputs))
	}
	if len(targets) != 3*4 {
		t.Fatalf("targets len %d, want 12", len(targets))
	}
	// targets[i] must be the token after inputs[i] within each row.
	for b := 0; b < 3; b++ {
		for i := 0; i < 3; i++ {
			// Decode input[b*T+i] and input[b*T+i+1] must match decode(target[b*T+i]).
			if targets[b*4+i] != 0 && inputs[b*4+i+1] != targets[b*4+i] {
				t.Fatalf("batch %d pos %d: input next %d != target %d",
					b, i, inputs[b*4+i+1], targets[b*4+i])
			}
		}
	}
}

func TestCharTokenizerUnknownIgnored(t *testing.T) {
	tok := NewCharTokenizer("abc")
	ids := tok.Encode("abcXYZ")
	// Unknown characters are silently dropped.
	if len(ids) != 3 {
		t.Fatalf("Encode dropped wrong count: got %d ids, want 3", len(ids))
	}
}
