package tensor

import (
	"fmt"
	"math"
	"math/rand"
)

// Tensor is a dense float32 array with an associated shape (row-major).
type Tensor struct {
	Shape []int
	Data  []float32
}

// Numel returns the total number of elements.
func (t *Tensor) Numel() int {
	n := 1
	for _, s := range t.Shape {
		n *= s
	}
	return n
}

// Stride returns the stride for dimension d (product of trailing dims).
func (t *Tensor) Stride(d int) int {
	s := 1
	for i := d + 1; i < len(t.Shape); i++ {
		s *= t.Shape[i]
	}
	return s
}

// NDim returns the number of dimensions.
func (t *Tensor) NDim() int { return len(t.Shape) }

// Clone returns a deep copy.
func (t *Tensor) Clone() *Tensor {
	d := make([]float32, len(t.Data))
	copy(d, t.Data)
	s := make([]int, len(t.Shape))
	copy(s, t.Shape)
	return &Tensor{Shape: s, Data: d}
}

// cloneShape returns a copy of the shape slice.
func cloneShape(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
}

// Scalar returns a 0-d tensor (shape=[]) wrapping a single value.
func Scalar(v float32) *Tensor {
	return &Tensor{Shape: []int{1}, Data: []float32{v}}
}

// Zeros returns a zero tensor of the given shape.
func Zeros(shape ...int) *Tensor {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return &Tensor{Shape: cloneShape(shape), Data: make([]float32, n)}
}

// Ones returns an all-ones tensor.
func Ones(shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = 1
	}
	return t
}

// Full returns a tensor filled with value v.
func Full(v float32, shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = v
	}
	return t
}

// FromSlice wraps an existing []float32 with a shape (no copy).
func FromSlice(data []float32, shape ...int) *Tensor {
	return &Tensor{Shape: cloneShape(shape), Data: data}
}

// RandNorm returns a tensor with values drawn from N(0, std) using Box-Muller.
func RandNorm(rng *rand.Rand, std float32, shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := 0; i < len(t.Data); i += 2 {
		u1 := rng.Float64()
		u2 := rng.Float64()
		// Avoid log(0)
		for u1 == 0 {
			u1 = rng.Float64()
		}
		z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		z1 := math.Sqrt(-2*math.Log(u1)) * math.Sin(2*math.Pi*u2)
		t.Data[i] = float32(z0) * std
		if i+1 < len(t.Data) {
			t.Data[i+1] = float32(z1) * std
		}
	}
	return t
}

// Reshape returns a new Tensor sharing the same Data slice with a new shape.
// Panics if Numel doesn't match.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	n := 1
	for _, s := range shape {
		n *= s
	}
	if n != len(t.Data) {
		panic(fmt.Sprintf("reshape: cannot reshape tensor of size %d into shape %v", len(t.Data), shape))
	}
	return &Tensor{Shape: cloneShape(shape), Data: t.Data}
}

// Transpose swaps two dimensions, materialising the result (copy).
func (t *Tensor) Transpose(dim0, dim1 int) *Tensor {
	ndim := len(t.Shape)
	if dim0 < 0 {
		dim0 += ndim
	}
	if dim1 < 0 {
		dim1 += ndim
	}
	if dim0 == dim1 {
		return t.Clone()
	}

	newShape := cloneShape(t.Shape)
	newShape[dim0], newShape[dim1] = newShape[dim1], newShape[dim0]

	out := Zeros(newShape...)

	// Build strides for the original tensor.
	srcStrides := make([]int, ndim)
	srcStrides[ndim-1] = 1
	for i := ndim - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * t.Shape[i+1]
	}

	// Build strides for the output tensor.
	dstStrides := make([]int, ndim)
	dstStrides[ndim-1] = 1
	for i := ndim - 2; i >= 0; i-- {
		dstStrides[i] = dstStrides[i+1] * out.Shape[i+1]
	}

	// Iterate over all multi-indices.
	idx := make([]int, ndim)
	total := t.Numel()
	for flat := 0; flat < total; flat++ {
		// Compute multi-index from flat source index.
		rem := flat
		for d := 0; d < ndim; d++ {
			idx[d] = rem / srcStrides[d]
			rem %= srcStrides[d]
		}

		// Compute destination multi-index (swap dim0 and dim1).
		dstFlat := 0
		for d := 0; d < ndim; d++ {
			dstD := d
			if d == dim0 {
				dstD = dim1
			} else if d == dim1 {
				dstD = dim0
			}
			dstFlat += idx[d] * dstStrides[dstD]
		}

		out.Data[dstFlat] = t.Data[flat]
	}
	return out
}

// String returns a human-readable summary of the tensor.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor%v", t.Shape)
}
