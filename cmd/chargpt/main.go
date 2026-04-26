// chargpt trains a character-level GPT language model on a text file.
// Usage:
//
//	go run ./cmd/chargpt -input input.txt -model gpt-mini -max_iters 5000
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/fieldingj/mingpt/model"
	"github.com/fieldingj/mingpt/nn"
	"github.com/fieldingj/mingpt/tokenizer"
	"github.com/fieldingj/mingpt/trainer"
)

func main() {
	inputFile := flag.String("input", "input.txt", "training corpus (text file)")
	modelType := flag.String("model", "gpt-mini", "gpt-nano|gpt-micro|gpt-mini|gpt1")
	blockSize := flag.Int("block_size", 128, "context window length")
	maxIters := flag.Int("max_iters", 5000, "training iterations")
	batchSize := flag.Int("batch_size", 64, "batch size")
	lr := flag.Float64("lr", 5e-4, "learning rate")
	genEvery := flag.Int("gen_every", 500, "generate sample every N iters")
	genLen := flag.Int("gen_len", 500, "number of tokens to generate")
	genPrompt := flag.String("prompt", "O God, O God!", "generation prompt")
	seed := flag.Int64("seed", 3407, "random seed")
	flag.Parse()

	// Seed the shared RNG used for weight init and dropout.
	nn.RNG = rand.New(rand.NewSource(*seed))

	// Load corpus.
	raw, err := os.ReadFile(*inputFile)
	if err != nil {
		log.Fatalf("read input: %v", err)
	}
	text := string(raw)

	// Build tokenizer and dataset.
	tok := tokenizer.NewCharTokenizer(text)
	ds := tokenizer.NewCharDataset(text, *blockSize, tok)
	fmt.Printf("corpus: %d chars | vocab: %d | dataset: %d samples\n",
		len(raw), tok.VocabSize(), ds.Len())

	// Build model.
	cfg := model.Preset(*modelType)
	cfg.VocabSize = tok.VocabSize()
	cfg.BlockSize = *blockSize
	cfg.EmbdDrop = 0.1
	cfg.ResidDrop = 0.1
	cfg.AttnDrop = 0.1
	gpt := model.NewGPT(cfg)
	fmt.Printf("model: %s | params: %d\n", *modelType, gpt.NumParams())

	// Build trainer.
	trCfg := trainer.DefaultConfig()
	trCfg.MaxIters = *maxIters
	trCfg.BatchSize = *batchSize
	trCfg.LR = float32(*lr)
	trCfg.EvalEvery = *genEvery
	trCfg.Seed = *seed

	genRNG := rand.New(rand.NewSource(*seed + 1))
	tr := trainer.New(trCfg, gpt, ds)

	tr.AddCallback("on_eval", func(t *trainer.Trainer) {
		promptIDs := tok.Encode(*genPrompt)
		generated := gpt.Generate(promptIDs, *genLen, 1.0, 10, genRNG)
		fmt.Printf("\n--- sample (iter %d, loss %.4f) ---\n%s\n---\n\n",
			t.Iter, t.LastLoss, tok.Decode(generated))
	})

	// Train.
	tr.Run()

	fmt.Printf("\nfinal loss: %.4f\n", tr.LastLoss)
}
