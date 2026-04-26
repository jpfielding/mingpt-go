package tensor

import (
	"math"
	"testing"
)

func floatEq(a, b float32, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func TestElemAddBroadcast(t *testing.T) {
	// [1,2,3] + [3] → [1,2,3]
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float32{10, 20, 30}, 3)
	c := ElemAdd(a, b)
	want := []float32{11, 22, 33, 14, 25, 36}
	for i, v := range c.Data {
		if !floatEq(v, want[i], 1e-6) {
			t.Fatalf("ElemAdd[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestElemAddBroadcast3D(t *testing.T) {
	// [2,2,3] + [1,1,3]
	a := FromSlice([]float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	}, 2, 2, 3)
	b := FromSlice([]float32{100, 200, 300}, 1, 1, 3)
	c := ElemAdd(a, b)
	if !ShapeEqual(c.Shape, []int{2, 2, 3}) {
		t.Fatalf("shape = %v", c.Shape)
	}
	if c.Data[0] != 101 || c.Data[5] != 306 || c.Data[11] != 312 {
		t.Fatalf("broadcast add wrong: %v", c.Data)
	}
}

func TestMatMul2D(t *testing.T) {
	// [2,3] @ [3,2] = [2,2]
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float32{7, 8, 9, 10, 11, 12}, 3, 2)
	c := MatMul(a, b)
	// row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	// row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	want := []float32{58, 64, 139, 154}
	for i, v := range c.Data {
		if !floatEq(v, want[i], 1e-5) {
			t.Fatalf("MatMul[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestMatMulBatched(t *testing.T) {
	// [2,2,3] @ [2,3,1] = [2,2,1]
	a := FromSlice([]float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	}, 2, 2, 3)
	b := FromSlice([]float32{
		1, 1, 1,
		2, 2, 2,
	}, 2, 3, 1)
	c := MatMul(a, b)
	want := []float32{6, 15, 48, 66}
	for i, v := range c.Data {
		if !floatEq(v, want[i], 1e-5) {
			t.Fatalf("batched MatMul[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestSumAxis(t *testing.T) {
	// [2,3] sum axis=0 → [1,3]
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	s := SumAxis(a, 0)
	if !ShapeEqual(s.Shape, []int{1, 3}) {
		t.Fatalf("shape %v", s.Shape)
	}
	want := []float32{5, 7, 9}
	for i, v := range s.Data {
		if !floatEq(v, want[i], 1e-6) {
			t.Fatalf("SumAxis[%d] = %f, want %f", i, v, want[i])
		}
	}

	// sum axis=1 → [2,1]
	s2 := SumAxis(a, 1)
	if !ShapeEqual(s2.Shape, []int{2, 1}) {
		t.Fatalf("shape %v", s2.Shape)
	}
	if s2.Data[0] != 6 || s2.Data[1] != 15 {
		t.Fatalf("SumAxis=1: %v", s2.Data)
	}
}

func TestSumKeepBroadcastDims(t *testing.T) {
	// Simulate Add backward for a+b where a=[1,3], b=[2,3].
	dZ := FromSlice([]float32{1, 1, 1, 1, 1, 1}, 2, 3)
	dA := SumKeepBroadcastDims(dZ, []int{1, 3})
	if !ShapeEqual(dA.Shape, []int{1, 3}) {
		t.Fatalf("shape %v", dA.Shape)
	}
	want := []float32{2, 2, 2}
	for i, v := range dA.Data {
		if !floatEq(v, want[i], 1e-6) {
			t.Fatalf("dA[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestTransposeRoundTrip(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	aT := a.Transpose(0, 1)
	if !ShapeEqual(aT.Shape, []int{3, 2}) {
		t.Fatalf("shape %v", aT.Shape)
	}
	back := aT.Transpose(0, 1)
	for i, v := range back.Data {
		if v != a.Data[i] {
			t.Fatalf("round-trip[%d]: %v vs %v", i, v, a.Data[i])
		}
	}
}

func TestRandNormStats(t *testing.T) {
	rng := testRng(42)
	x := RandNorm(rng, 1.0, 10000)
	var sum, sq float64
	for _, v := range x.Data {
		sum += float64(v)
		sq += float64(v) * float64(v)
	}
	mean := sum / float64(len(x.Data))
	variance := sq/float64(len(x.Data)) - mean*mean
	if math.Abs(mean) > 0.05 {
		t.Fatalf("mean %.4f too far from 0", mean)
	}
	if math.Abs(variance-1.0) > 0.1 {
		t.Fatalf("variance %.4f too far from 1", variance)
	}
}
