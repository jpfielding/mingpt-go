package tensor

import "math/rand"

// testRng returns a deterministic RNG for tests in this package.
func testRng(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
