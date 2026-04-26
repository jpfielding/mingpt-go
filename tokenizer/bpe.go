package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"unicode"
)

// BPETokenizer is a GPT-2 compatible Byte Pair Encoding tokenizer.
// Load via NewBPETokenizer; data files can be downloaded from OpenAI's CDN.
type BPETokenizer struct {
	encoder     map[string]int
	decoder     map[int]string
	bpeRanks    map[[2]string]int
	byteEncoder [256]rune
	byteDecoder map[rune]byte
	cache       map[string]string
}

// NewBPETokenizer loads encoder and merge data from files.
// vocabPath:  path to vocab.json (encoder map)
// mergesPath: path to merges.txt (BPE merge rules)
func NewBPETokenizer(vocabPath, mergesPath string) (*BPETokenizer, error) {
	// Load encoder (vocab.json).
	vf, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("open vocab: %w", err)
	}
	defer vf.Close()
	encoder := make(map[string]int)
	if err := json.NewDecoder(vf).Decode(&encoder); err != nil {
		return nil, fmt.Errorf("decode vocab: %w", err)
	}

	decoder := make(map[int]string, len(encoder))
	for k, v := range encoder {
		decoder[v] = k
	}

	// Load merges.txt.
	mf, err := os.Open(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("open merges: %w", err)
	}
	defer mf.Close()

	bpeRanks := make(map[[2]string]int)
	scanner := bufio.NewScanner(mf)
	rank := 0
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		bpeRanks[[2]string{parts[0], parts[1]}] = rank
		rank++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read merges: %w", err)
	}

	be, bd := buildByteEncoder()

	return &BPETokenizer{
		encoder:     encoder,
		decoder:     decoder,
		bpeRanks:    bpeRanks,
		byteEncoder: be,
		byteDecoder: bd,
		cache:        make(map[string]string),
	}, nil
}

// VocabSize returns the number of tokens (50,257 for GPT-2).
func (t *BPETokenizer) VocabSize() int { return len(t.encoder) }

// Encode converts text to a sequence of token IDs.
func (t *BPETokenizer) Encode(text string) []int {
	var ids []int
	for _, word := range t.preTokenize(text) {
		// Convert word bytes to byte-encoder characters.
		var wordChars []string
		for _, b := range []byte(word) {
			wordChars = append(wordChars, string(t.byteEncoder[b]))
		}
		merged := t.bpe(wordChars)
		for _, tok := range strings.Split(merged, " ") {
			if id, ok := t.encoder[tok]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// Decode converts token IDs back to text.
func (t *BPETokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if s, ok := t.decoder[id]; ok {
			sb.WriteString(s)
		}
	}
	// Convert from byte-encoder characters back to bytes.
	encoded := sb.String()
	result := make([]byte, 0, len(encoded))
	for _, r := range encoded {
		if b, ok := t.byteDecoder[r]; ok {
			result = append(result, b)
		}
	}
	return string(result)
}

// bpe applies BPE merging to a list of characters (tokens).
// Returns a space-separated string of merged tokens.
func (t *BPETokenizer) bpe(word []string) string {
	key := strings.Join(word, " ")
	if cached, ok := t.cache[key]; ok {
		return cached
	}

	for len(word) > 1 {
		// Find the bigram with the lowest rank.
		bestRank := -1
		var bestPair [2]string
		found := false
		for i := 0; i < len(word)-1; i++ {
			pair := [2]string{word[i], word[i+1]}
			if rank, ok := t.bpeRanks[pair]; ok {
				if !found || rank < bestRank {
					bestRank = rank
					bestPair = pair
					found = true
				}
			}
		}
		if !found {
			break
		}

		// Merge all occurrences of bestPair.
		var merged []string
		i := 0
		for i < len(word) {
			if i < len(word)-1 && word[i] == bestPair[0] && word[i+1] == bestPair[1] {
				merged = append(merged, word[i]+word[i+1])
				i += 2
			} else {
				merged = append(merged, word[i])
				i++
			}
		}
		word = merged
	}

	result := strings.Join(word, " ")
	t.cache[key] = result
	return result
}

// preTokenize splits text using GPT-2's pre-tokenization rules.
// We implement this as a hand-rolled state machine to avoid regex \p{L} limitations.
func (t *BPETokenizer) preTokenize(text string) []string {
	var tokens []string
	runes := []rune(text)
	i := 0
	for i < len(runes) {
		r := runes[i]
		switch {
		// Contractions: 's, 't, 're, 've, 'm, 'll, 'd
		case r == '\'' && i+1 < len(runes):
			next := runes[i+1]
			switch next {
			case 's', 't', 'm':
				tokens = append(tokens, string(runes[i:i+2]))
				i += 2
			case 'r', 'v':
				if i+2 < len(runes) && runes[i+2] == 'e' {
					tokens = append(tokens, string(runes[i:i+3]))
					i += 3
				} else {
					tokens = append(tokens, string(r))
					i++
				}
			case 'l':
				if i+2 < len(runes) && runes[i+2] == 'l' {
					tokens = append(tokens, string(runes[i:i+3]))
					i += 3
				} else {
					tokens = append(tokens, string(r))
					i++
				}
			case 'd':
				tokens = append(tokens, string(runes[i:i+2]))
				i += 2
			default:
				tokens = append(tokens, string(r))
				i++
			}

		// Letter sequence (optionally preceded by a space).
		case unicode.IsLetter(r) || (r == ' ' && i+1 < len(runes) && unicode.IsLetter(runes[i+1])):
			j := i
			if runes[j] == ' ' {
				j++
			}
			for j < len(runes) && unicode.IsLetter(runes[j]) {
				j++
			}
			tokens = append(tokens, string(runes[i:j]))
			i = j

		// Number sequence (optionally preceded by a space).
		case unicode.IsNumber(r) || (r == ' ' && i+1 < len(runes) && unicode.IsNumber(runes[i+1])):
			j := i
			if runes[j] == ' ' {
				j++
			}
			for j < len(runes) && unicode.IsNumber(runes[j]) {
				j++
			}
			tokens = append(tokens, string(runes[i:j]))
			i = j

		// Whitespace run (not followed by non-whitespace — trailing).
		case r == ' ':
			// Consume space-only run.
			j := i
			for j < len(runes) && runes[j] == ' ' {
				j++
			}
			tokens = append(tokens, string(runes[i:j]))
			i = j

		// Other non-whitespace, non-letter, non-digit (punctuation etc).
		default:
			j := i
			if runes[j] == ' ' {
				j++
			}
			// Consume non-whitespace, non-letter, non-digit chars.
			start := j
			for j < len(runes) && !unicode.IsSpace(runes[j]) && !unicode.IsLetter(runes[j]) && !unicode.IsNumber(runes[j]) {
				j++
			}
			if j > start {
				tokens = append(tokens, string(runes[i:j]))
				i = j
			} else {
				tokens = append(tokens, string(r))
				i++
			}
		}
	}
	return tokens
}

// buildByteEncoder maps all 256 bytes to printable Unicode characters.
// Bytes that are already printable map to themselves;
// the others (control chars, etc.) map to code points starting at 256.
func buildByteEncoder() ([256]rune, map[rune]byte) {
	var be [256]rune
	bd := make(map[rune]byte, 256)

	// Printable ASCII ranges that map to themselves.
	printable := func(b byte) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}

	n := 256 // next code point for non-printable bytes
	for i := 0; i < 256; i++ {
		b := byte(i)
		if printable(b) {
			be[i] = rune(b)
		} else {
			be[i] = rune(n)
			n++
		}
	}
	// Build the reverse map.
	for i, r := range be {
		bd[r] = byte(i)
	}
	return be, bd
}

// BPEDataset wraps an encoded corpus, satisfying the trainer.Dataset interface.
type BPEDataset struct {
	Data      []int
	BlockSize int
	Tokenizer *BPETokenizer
}

// NewBPEDataset encodes text and wraps it as a dataset.
func NewBPEDataset(text string, blockSize int, tok *BPETokenizer) *BPEDataset {
	return &BPEDataset{
		Data:      tok.Encode(text),
		BlockSize: blockSize,
		Tokenizer: tok,
	}
}

func (ds *BPEDataset) Len() int { return len(ds.Data) - ds.BlockSize }

func (ds *BPEDataset) GetBatch(batchSize int, rng *rand.Rand) (inputs, targets []int) {
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
