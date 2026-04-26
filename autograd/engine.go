package autograd

import "github.com/fieldingj/mingpt/tensor"

// Node is a value in the computation graph.
// It carries the forward result (Value) and accumulates gradients (Grad).
type Node struct {
	Value    *tensor.Tensor
	Grad     *tensor.Tensor // accumulated gradient; nil until backward runs
	op       string
	inputs   []*Node
	backward func()
}

// Param creates a leaf parameter node. The backward function is nil —
// its gradient is accumulated by parent nodes via accumulateGrad.
func Param(t *tensor.Tensor) *Node {
	return &Node{Value: t, op: "param"}
}

// Const creates a constant node. No gradient flows back.
func Const(t *tensor.Tensor) *Node {
	return &Node{Value: t, op: "const"}
}

// ZeroGrad zeroes the gradient in-place (or allocates a zero tensor if nil).
func (n *Node) ZeroGrad() {
	if n.Grad == nil {
		n.Grad = tensor.Zeros(n.Value.Shape...)
	} else {
		for i := range n.Grad.Data {
			n.Grad.Data[i] = 0
		}
	}
}

// NewNode creates a node without a backward function.
// Call SetBackwardFn after creation to set up the closure (which may capture n itself).
func NewNode(value *tensor.Tensor, op string, inputs []*Node, _ func()) *Node {
	return &Node{Value: value, op: op, inputs: inputs}
}

// SetBackwardFn sets the backward function. Call after NewNode when the closure
// needs to capture the returned node.
func (n *Node) SetBackwardFn(fn func()) {
	n.backward = fn
}

// AccumulateGrad is the exported version of accumulateGrad for external packages.
func AccumulateGrad(n *Node, delta *tensor.Tensor) {
	accumulateGrad(n, delta)
}

// accumulateGrad adds delta into n.Grad, allocating zeros if needed.
func accumulateGrad(n *Node, delta *tensor.Tensor) {
	if n.op == "const" {
		return
	}
	if n.Grad == nil {
		n.Grad = tensor.Zeros(n.Value.Shape...)
	}
	tensor.AddInPlace(n.Grad, delta)
}

// Backward runs the reverse-mode pass from node loss (assumed scalar).
func Backward(loss *Node) {
	// Topological sort.
	order := topoSort(loss)

	// Seed gradient at the loss.
	loss.Grad = tensor.Ones(loss.Value.Shape...)

	// Walk in reverse order, calling each backward closure.
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		if n.backward != nil && n.Grad != nil {
			n.backward()
		}
	}
}

// topoSort returns nodes in topological order (leaves first, loss last).
func topoSort(root *Node) []*Node {
	var order []*Node
	visited := make(map[*Node]bool)

	var visit func(*Node)
	visit = func(n *Node) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, inp := range n.inputs {
			visit(inp)
		}
		order = append(order, n)
	}
	visit(root)
	return order
}
