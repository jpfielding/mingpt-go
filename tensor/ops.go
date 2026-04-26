package tensor

import (
	"fmt"
	"math"
)

// ---- Element-wise operations ----

// ElemAdd adds two tensors element-wise. Supports broadcasting along leading dims
// (e.g. [1,T,C] + [B,T,C] → [B,T,C]).
func ElemAdd(a, b *Tensor) *Tensor {
	out := broadcastBinaryOp(a, b, func(x, y float32) float32 { return x + y })
	return out
}

// ElemSub subtracts b from a element-wise with broadcasting.
func ElemSub(a, b *Tensor) *Tensor {
	return broadcastBinaryOp(a, b, func(x, y float32) float32 { return x - y })
}

// ElemMul multiplies two tensors element-wise with broadcasting.
func ElemMul(a, b *Tensor) *Tensor {
	return broadcastBinaryOp(a, b, func(x, y float32) float32 { return x * y })
}

// ElemDiv divides a by b element-wise with broadcasting.
func ElemDiv(a, b *Tensor) *Tensor {
	return broadcastBinaryOp(a, b, func(x, y float32) float32 { return x / y })
}

// Scale multiplies every element by a scalar.
func Scale(t *Tensor, s float32) *Tensor {
	out := Zeros(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = v * s
	}
	return out
}

// AddInPlace adds b into a in place (a += b, both must have the same shape).
func AddInPlace(a, b *Tensor) {
	if len(a.Data) != len(b.Data) {
		panic(fmt.Sprintf("AddInPlace: shape mismatch %v vs %v", a.Shape, b.Shape))
	}
	for i := range a.Data {
		a.Data[i] += b.Data[i]
	}
}

// broadcastBinaryOp applies op(a, b) with NumPy-style broadcasting over leading dims.
// Supports the case where a or b has shape [1, ...] along some leading dimension.
func broadcastBinaryOp(a, b *Tensor, op func(float32, float32) float32) *Tensor {
	outShape := broadcastShape(a.Shape, b.Shape)
	out := Zeros(outShape...)

	total := out.Numel()
	aStrides := BroadcastStrides(a.Shape, outShape)
	bStrides := BroadcastStrides(b.Shape, outShape)

	outStrides := make([]int, len(outShape))
	outStrides[len(outShape)-1] = 1
	for i := len(outShape) - 2; i >= 0; i-- {
		outStrides[i] = outStrides[i+1] * outShape[i+1]
	}

	idx := make([]int, len(outShape))
	for flat := 0; flat < total; flat++ {
		// Decompose flat index into multi-index.
		rem := flat
		for d := 0; d < len(outShape); d++ {
			idx[d] = rem / outStrides[d]
			rem %= outStrides[d]
		}
		aFlat := 0
		bFlat := 0
		for d := 0; d < len(outShape); d++ {
			aFlat += idx[d] * aStrides[d]
			bFlat += idx[d] * bStrides[d]
		}
		out.Data[flat] = op(a.Data[aFlat], b.Data[bFlat])
	}
	return out
}

// broadcastShape returns the output shape for two tensors, numpy-style.
func broadcastShape(a, b []int) []int {
	ndim := len(a)
	if len(b) > ndim {
		ndim = len(b)
	}
	out := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		ai := 1
		if i >= ndim-len(a) {
			ai = a[i-(ndim-len(a))]
		}
		bi := 1
		if i >= ndim-len(b) {
			bi = b[i-(ndim-len(b))]
		}
		if ai == bi {
			out[i] = ai
		} else if ai == 1 {
			out[i] = bi
		} else if bi == 1 {
			out[i] = ai
		} else {
			panic(fmt.Sprintf("broadcastShape: incompatible shapes %v and %v", a, b))
		}
	}
	return out
}

// BroadcastStrides returns effective strides for a tensor of shape src broadcast to dstShape.
// Broadcast dims get stride 0.
func BroadcastStrides(src, dst []int) []int {
	ndim := len(dst)
	strides := make([]int, ndim)

	// Compute natural strides for src.
	srcNatural := make([]int, len(src))
	srcNatural[len(src)-1] = 1
	for i := len(src) - 2; i >= 0; i-- {
		srcNatural[i] = srcNatural[i+1] * src[i+1]
	}

	offset := ndim - len(src)
	for d := 0; d < ndim; d++ {
		srcD := d - offset
		if srcD < 0 {
			strides[d] = 0
		} else if src[srcD] == 1 && dst[d] > 1 {
			strides[d] = 0
		} else {
			strides[d] = srcNatural[srcD]
		}
	}
	return strides
}

// ---- Reductions ----

// Sum sums all elements.
func Sum(t *Tensor) float32 {
	var s float64
	for _, v := range t.Data {
		s += float64(v)
	}
	return float32(s)
}

// SumAxis reduces along axis, keeping the dimension (keepdim=true).
// Returns a tensor with shape[axis]=1.
func SumAxis(t *Tensor, axis int) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	outShape := cloneShape(t.Shape)
	outShape[axis] = 1
	out := Zeros(outShape...)

	// Strides for iterating the input.
	stride := t.Stride(axis)
	axisSize := t.Shape[axis]

	total := t.Numel() / axisSize
	for i := 0; i < total; i++ {
		// Compute the base flat index in 'out' corresponding to 'i'.
		// We treat the input as if axis is collapsed to 1.
		outer := i / stride
		inner := i % stride
		outIdx := outer*stride + inner

		var acc float64
		for j := 0; j < axisSize; j++ {
			acc += float64(t.Data[outer*axisSize*stride+j*stride+inner])
		}
		out.Data[outIdx] = float32(acc)
	}
	return out
}

// MeanAxis reduces along axis, keeping the dimension.
func MeanAxis(t *Tensor, axis int) *Tensor {
	s := SumAxis(t, axis)
	n := float32(t.Shape[axis])
	return Scale(s, 1/n)
}

// MaxAxis takes the max along axis, keeping the dimension.
func MaxAxis(t *Tensor, axis int) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	outShape := cloneShape(t.Shape)
	outShape[axis] = 1
	out := Full(float32(math.Inf(-1)), outShape...)

	stride := t.Stride(axis)
	axisSize := t.Shape[axis]

	total := t.Numel() / axisSize
	for i := 0; i < total; i++ {
		outer := i / stride
		inner := i % stride
		outIdx := outer*stride + inner

		for j := 0; j < axisSize; j++ {
			v := t.Data[outer*axisSize*stride+j*stride+inner]
			if v > out.Data[outIdx] {
				out.Data[outIdx] = v
			}
		}
	}
	return out
}

// SumKeepBroadcastDims reduces dZ to targetShape by summing broadcast dimensions.
// Used in Add backward to map the output gradient back to an operand's shape.
func SumKeepBroadcastDims(dZ *Tensor, targetShape []int) *Tensor {
	result := dZ

	// Step 1: Sum and squeeze extra leading dimensions.
	for len(result.Shape) > len(targetShape) {
		result = SumAxis(result, 0)
		// Squeeze: remove the leading dim (now size 1).
		result = result.Reshape(result.Shape[1:]...)
	}

	// Step 2: Sum over dimensions where targetShape[d]==1 but result.Shape[d]>1.
	for d := 0; d < len(targetShape); d++ {
		if targetShape[d] == 1 && result.Shape[d] > 1 {
			result = SumAxis(result, d)
		}
	}

	return result.Reshape(targetShape...)
}

// ---- Matrix multiply ----

// MatMul performs batched matrix multiplication.
// Supports:
//   - 2D x 2D: [M,K] x [K,N] → [M,N]
//   - 3D x 3D: [B,M,K] x [B,K,N] → [B,M,N]
//   - 4D x 4D: [B,H,M,K] x [B,H,K,N] → [B,H,M,N]
//   - 3D x 2D: [B,M,K] x [K,N] → [B,M,N]  (weight shared across batch)
func MatMul(a, b *Tensor) *Tensor {
	ndimA := len(a.Shape)
	ndimB := len(b.Shape)

	switch {
	case ndimA == 2 && ndimB == 2:
		return matMul2D(a, b)
	case ndimA == 3 && ndimB == 2:
		return matMul3D2D(a, b)
	case ndimA == 3 && ndimB == 3:
		return matMulBatched(a, b, 3)
	case ndimA == 4 && ndimB == 4:
		return matMulBatched(a, b, 4)
	default:
		panic(fmt.Sprintf("MatMul: unsupported shapes %v x %v", a.Shape, b.Shape))
	}
}

func matMul2D(a, b *Tensor) *Tensor {
	M, K := a.Shape[0], a.Shape[1]
	K2, N := b.Shape[0], b.Shape[1]
	if K != K2 {
		panic(fmt.Sprintf("matMul2D: inner dims mismatch %d vs %d", K, K2))
	}
	out := Zeros(M, N)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			av := a.Data[m*K+k]
			for n := 0; n < N; n++ {
				out.Data[m*N+n] += av * b.Data[k*N+n]
			}
		}
	}
	return out
}

// matMul3D2D: [B,M,K] x [K,N] → [B,M,N]
func matMul3D2D(a, b *Tensor) *Tensor {
	B, M, K := a.Shape[0], a.Shape[1], a.Shape[2]
	K2, N := b.Shape[0], b.Shape[1]
	if K != K2 {
		panic(fmt.Sprintf("matMul3D2D: inner dims mismatch %d vs %d", K, K2))
	}
	out := Zeros(B, M, N)
	for batch := 0; batch < B; batch++ {
		for m := 0; m < M; m++ {
			for k := 0; k < K; k++ {
				av := a.Data[batch*M*K+m*K+k]
				for n := 0; n < N; n++ {
					out.Data[batch*M*N+m*N+n] += av * b.Data[k*N+n]
				}
			}
		}
	}
	return out
}

// matMulBatched handles 3D and 4D batched matmul where the leading dimensions
// are batch dimensions.
func matMulBatched(a, b *Tensor, ndim int) *Tensor {
	// The last two dims are the matrix dims; the rest are batch dims.
	batchDims := ndim - 2
	M := a.Shape[batchDims]
	K := a.Shape[batchDims+1]
	K2 := b.Shape[batchDims]
	N := b.Shape[batchDims+1]
	if K != K2 {
		panic(fmt.Sprintf("matMulBatched: inner dims mismatch %d vs %d", K, K2))
	}

	// Compute number of batch elements.
	nBatch := 1
	outShape := make([]int, ndim)
	for d := 0; d < batchDims; d++ {
		if a.Shape[d] != b.Shape[d] {
			panic(fmt.Sprintf("matMulBatched: batch dim %d mismatch %d vs %d", d, a.Shape[d], b.Shape[d]))
		}
		outShape[d] = a.Shape[d]
		nBatch *= a.Shape[d]
	}
	outShape[batchDims] = M
	outShape[batchDims+1] = N

	out := Zeros(outShape...)
	for batch := 0; batch < nBatch; batch++ {
		aOff := batch * M * K
		bOff := batch * K * N
		oOff := batch * M * N
		for m := 0; m < M; m++ {
			for k := 0; k < K; k++ {
				av := a.Data[aOff+m*K+k]
				for n := 0; n < N; n++ {
					out.Data[oOff+m*N+n] += av * b.Data[bOff+k*N+n]
				}
			}
		}
	}
	return out
}

// ---- Utility ----

// Exp returns element-wise exp(x).
func Exp(t *Tensor) *Tensor {
	out := Zeros(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = float32(math.Exp(float64(v)))
	}
	return out
}

// Log returns element-wise log(x).
func Log(t *Tensor) *Tensor {
	out := Zeros(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = float32(math.Log(float64(v)))
	}
	return out
}

// Sqrt returns element-wise sqrt(x).
func Sqrt(t *Tensor) *Tensor {
	out := Zeros(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = float32(math.Sqrt(float64(v)))
	}
	return out
}

// Pow2 returns element-wise x^2.
func Pow2(t *Tensor) *Tensor {
	out := Zeros(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = v * v
	}
	return out
}

// Tanh returns element-wise tanh(x).
func Tanh(t *Tensor) *Tensor {
	out := Zeros(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = float32(math.Tanh(float64(v)))
	}
	return out
}

// Neg returns element-wise -x.
func Neg(t *Tensor) *Tensor {
	return Scale(t, -1)
}

// ShapeEqual checks if two shapes are identical.
func ShapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Concat concatenates tensors along axis 0 (only 1D and 2D for simplicity).
func Concat0(tensors ...*Tensor) *Tensor {
	if len(tensors) == 0 {
		panic("Concat0: no tensors")
	}
	totalRows := 0
	cols := tensors[0].Shape[len(tensors[0].Shape)-1]
	for _, t := range tensors {
		totalRows += t.Shape[0]
	}
	out := Zeros(totalRows, cols)
	offset := 0
	for _, t := range tensors {
		copy(out.Data[offset:], t.Data)
		offset += len(t.Data)
	}
	return out
}

// SliceAxis0 returns a view of rows [start:end] along axis 0.
// The result shares the underlying slice.
func SliceAxis0(t *Tensor, start, end int) *Tensor {
	stride := t.Stride(0)
	newShape := cloneShape(t.Shape)
	newShape[0] = end - start
	return &Tensor{
		Shape: newShape,
		Data:  t.Data[start*stride : end*stride],
	}
}
