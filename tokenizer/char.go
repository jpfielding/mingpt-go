package tokenizer

import (
	"math/rand"
	"sort"
	"strings"
)

// CharTokenizer maps runes ↔ integer IDs.
type CharTokenizer struct {
	Stoi map[rune]int
	Itos []rune
}

// NewCharTokenizer builds a vocabulary from the unique characters in text.
// Characters are sorted for determinism.
func NewCharTokenizer(text string) *CharTokenizer {
	seen := make(map[rune]struct{})
	for _, r := range text {
		seen[r] = struct{}{}
	}
	chars := make([]rune, 0, len(seen))
	for r := range seen {
		chars = append(chars, r)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	stoi := make(map[rune]int, len(chars))
	for i, r := range chars {
		stoi[r] = i
	}
	return &CharTokenizer{Stoi: stoi, Itos: chars}
}

// VocabSize returns the number of unique tokens.
func (t *CharTokenizer) VocabSize() int { return len(t.Itos) }

// Encode converts text to integer IDs. Unknown characters map to 0.
func (t *CharTokenizer) Encode(s string) []int {
	ids := make([]int, 0, len(s))
	for _, r := range s {
		if id, ok := t.Stoi[r]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// Decode converts integer IDs back to a string.
func (t *CharTokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id >= 0 && id < len(t.Itos) {
			sb.WriteRune(t.Itos[id])
		}
	}
	return sb.String()
}

// CharDataset wraps an encoded text corpus for training.
type CharDataset struct {
	Data      []int
	BlockSize int
	Tokenizer *CharTokenizer
}

// NewCharDataset creates a dataset from text.
func NewCharDataset(text string, blockSize int, tok *CharTokenizer) *CharDataset {
	return &CharDataset{
		Data:      tok.Encode(text),
		BlockSize: blockSize,
		Tokenizer: tok,
	}
}

// Len returns the number of possible starting positions.
func (ds *CharDataset) Len() int { return len(ds.Data) - ds.BlockSize }

// GetBatch samples batchSize random windows from the corpus.
// inputs[i:i+T] → predict targets[i+1:i+T+1].
func (ds *CharDataset) GetBatch(batchSize int, rng *rand.Rand) (inputs, targets []int) {
	T := ds.BlockSize
	N := len(ds.Data) - T
	inputs = make([]int, batchSize*T)
	targets = make([]int, batchSize*T)
	for b := 0; b < batchSize; b++ {
		i := rng.Intn(N)
		copy(inputs[b*T:], ds.Data[i:i+T])
		copy(targets[b*T:], ds.Data[i+1:i+T+1])
	}
	return
}
