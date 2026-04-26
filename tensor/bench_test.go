package tensor

import (
	"math/rand"
	"testing"
)

func BenchmarkMatMul2D(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := RandNorm(rng, 1, 64, 128)
	w := RandNorm(rng, 1, 128, 256)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMul(a, w)
	}
}

func BenchmarkMatMul3D(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	// Batch 8, 64x128 @ 128x256 (mimics [B, T, C] @ [C, 4C]-like shape).
	a := RandNorm(rng, 1, 8, 64, 128)
	w := RandNorm(rng, 1, 8, 128, 256)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMul(a, w)
	}
}

func BenchmarkElemAdd(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	x := RandNorm(rng, 1, 1024, 1024)
	y := RandNorm(rng, 1, 1024, 1024)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ElemAdd(x, y)
	}
}
