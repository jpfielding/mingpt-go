package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFixtureFiles drops a vocab.json and merges.txt into dir and returns
// their paths.
func writeFixtureFiles(t *testing.T, dir string, vocab map[string]int, merges []string) (string, string) {
	t.Helper()

	vocabPath := filepath.Join(dir, "vocab.json")
	vb, err := json.Marshal(vocab)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(vocabPath, vb, 0644); err != nil {
		t.Fatal(err)
	}

	mergesPath := filepath.Join(dir, "merges.txt")
	contents := "#version: 0.2\n" + strings.Join(merges, "\n") + "\n"
	if err := os.WriteFile(mergesPath, []byte(contents), 0644); err != nil {
		t.Fatal(err)
	}

	return vocabPath, mergesPath
}

// buildCharVocab returns a vocab mapping each byte-encoded character in the
// given strings to a sequential ID. Useful when testing with zero BPE merges.
func buildCharVocab(texts ...string) map[string]int {
	tok := &BPETokenizer{} // need byteEncoder; fill it in
	be, _ := buildByteEncoder()
	tok.byteEncoder = be

	vocab := make(map[string]int)
	next := 0
	add := func(s string) {
		if _, ok := vocab[s]; !ok {
			vocab[s] = next
			next++
		}
	}
	for _, t := range texts {
		for _, b := range []byte(t) {
			add(string(be[b]))
		}
	}
	return vocab
}

// TestBPECharVocabRoundTrip verifies that with no merges and a vocab covering
// every byte-encoded character, Encode/Decode round-trips every input.
func TestBPECharVocabRoundTrip(t *testing.T) {
	texts := []string{
		"hello",
		"Hello, world!",
		"abc 123 xyz",
		"line1\nline2",
		"tabs\tand\tcontrols",
	}

	vocab := make(map[string]int)
	// Build a union vocab across all test strings.
	be, _ := buildByteEncoder()
	next := 0
	for _, text := range texts {
		for _, b := range []byte(text) {
			s := string(be[b])
			if _, ok := vocab[s]; !ok {
				vocab[s] = next
				next++
			}
		}
	}

	dir := t.TempDir()
	vocabPath, mergesPath := writeFixtureFiles(t, dir, vocab, nil)

	tok, err := NewBPETokenizer(vocabPath, mergesPath)
	if err != nil {
		t.Fatalf("NewBPETokenizer: %v", err)
	}
	if tok.VocabSize() != len(vocab) {
		t.Fatalf("VocabSize=%d, want %d", tok.VocabSize(), len(vocab))
	}

	for _, text := range texts {
		ids := tok.Encode(text)
		back := tok.Decode(ids)
		if back != text {
			t.Errorf("round-trip mismatch:\n  in:  %q\n  out: %q\n  ids: %v", text, back, ids)
		}
	}
}

// TestBPEMergesApplied checks that a merge rule collapses two tokens into one.
func TestBPEMergesApplied(t *testing.T) {
	// Base chars plus a merged "he" entry. Merge rule joins "h" + "e" → "he".
	vocab := map[string]int{
		"h": 0, "e": 1, "l": 2, "o": 3, "he": 4,
	}
	merges := []string{"h e"}

	dir := t.TempDir()
	vocabPath, mergesPath := writeFixtureFiles(t, dir, vocab, merges)

	tok, err := NewBPETokenizer(vocabPath, mergesPath)
	if err != nil {
		t.Fatal(err)
	}

	// "hello" pre-tokenises to one word ("hello"); byte-encoder is identity for
	// printable ASCII. BPE should merge "h" + "e" → "he" then stop (no rule for
	// "he l", "l l", or "l o"). Expected tokens: ["he", "l", "l", "o"].
	ids := tok.Encode("hello")
	want := []int{4, 2, 2, 3}
	if len(ids) != len(want) {
		t.Fatalf("ids %v, want %v", ids, want)
	}
	for i, id := range ids {
		if id != want[i] {
			t.Fatalf("ids[%d]=%d, want %d (full: %v)", i, id, want[i], ids)
		}
	}

	// Decode reverses it.
	back := tok.Decode(ids)
	if back != "hello" {
		t.Fatalf("Decode=%q, want %q", back, "hello")
	}
}

// TestBPEPreTokenize exercises the hand-rolled pre-tokenisation state machine.
func TestBPEPreTokenize(t *testing.T) {
	tok := &BPETokenizer{}
	cases := []struct {
		in   string
		want []string
	}{
		// Leading-space letter runs.
		{"hello world", []string{"hello", " world"}},
		// Contractions.
		{"don't", []string{"don", "'t"}},
		{"I'm here", []string{"I", "'m", " here"}},
		{"we're", []string{"we", "'re"}},
		{"they've", []string{"they", "'ve"}},
		{"I'll", []string{"I", "'ll"}},
		{"he'd", []string{"he", "'d"}},
		// Numbers.
		{"abc 123", []string{"abc", " 123"}},
		// Mixed punctuation stays together.
		{"!!!", []string{"!!!"}},
		// Multiple interior spaces before a letter stay grouped.
		// (GPT-2's ` ?\p{L}+` consumes at most one leading space.)
		{"a  b", []string{"a", "  ", "b"}},
	}
	for _, c := range cases {
		got := tok.preTokenize(c.in)
		if len(got) != len(c.want) {
			t.Errorf("preTokenize(%q) = %v, want %v", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("preTokenize(%q)[%d] = %q, want %q (full: %v)", c.in, i, got[i], c.want[i], got)
			}
		}
	}
}

// TestBPEByteEncoderBijective verifies the byte encoder is a bijection on all
// 256 bytes (critical for lossless encode/decode).
func TestBPEByteEncoderBijective(t *testing.T) {
	be, bd := buildByteEncoder()
	seen := make(map[rune]bool)
	for b := 0; b < 256; b++ {
		r := be[b]
		if seen[r] {
			t.Fatalf("byte encoder collision: byte %d maps to rune %q already seen", b, r)
		}
		seen[r] = true
		if bd[r] != byte(b) {
			t.Fatalf("byte decoder disagrees for byte %d: got %d", b, bd[r])
		}
	}
	if len(seen) != 256 {
		t.Fatalf("expected 256 unique runes, got %d", len(seen))
	}
}

// TestBPEDatasetShape smoke-tests that the dataset wrapper produces batches
// of the right length.
func TestBPEDatasetShape(t *testing.T) {
	vocab := buildCharVocab("abcdefghijklmnopqrstuvwxyz")
	dir := t.TempDir()
	vocabPath, mergesPath := writeFixtureFiles(t, dir, vocab, nil)

	tok, err := NewBPETokenizer(vocabPath, mergesPath)
	if err != nil {
		t.Fatal(err)
	}

	ds := NewBPEDataset("abcdefghijklmnop", 4, tok)
	if ds.Len() <= 0 {
		t.Fatalf("empty dataset: %d", ds.Len())
	}
}
