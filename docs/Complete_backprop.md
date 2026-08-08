# Neural Network From Scratch

# Complete Mathematical Derivation

## Architecture

$$
784
\rightarrow
128
\rightarrow
64
\rightarrow
10
$$

where

- Input Features = 784
- Hidden Layer 1 = 128 neurons
- Hidden Layer 2 = 64 neurons
- Output Layer = 10 neurons
- Number of training examples = $n$

---

# Mathematical Notation

## Input Matrix

The input matrix is

$$
X=A^{(0)}
$$

Shape

$$
X=A^{(0)}
\in
\mathbb{R}^{784\times n}
$$

where

- 784 = Number of input features
- $n$ = Number of training examples

---

## Weight Matrix

For layer $l$

$$
W^{(l)}
$$

Shape

### Layer 1

$$
W^{(1)}
\in
\mathbb{R}^{128\times784}
$$

### Layer 2

$$
W^{(2)}
\in
\mathbb{R}^{64\times128}
$$

### Layer 3

$$
W^{(3)}
\in
\mathbb{R}^{10\times64}
$$

---

## Bias Vector

Bias vector

$$
b^{(l)}
$$

Shapes

### Layer 1

$$
b^{(1)}
\in
\mathbb{R}^{128\times1}
$$

### Layer 2

$$
b^{(2)}
\in
\mathbb{R}^{64\times1}
$$

### Layer 3

$$
b^{(3)}
\in
\mathbb{R}^{10\times1}
$$

---

## Linear Transformation

The linear transformation before activation is

$$
Z^{(l)}
=
W^{(l)}
A^{(l-1)}
+
b^{(l)}
$$

Shapes

### Layer 1

$$
Z^{(1)}
\in
\mathbb{R}^{128\times n}
$$

### Layer 2

$$
Z^{(2)}
\in
\mathbb{R}^{64\times n}
$$

### Layer 3

$$
Z^{(3)}
\in
\mathbb{R}^{10\times n}
$$

---

## Activation

Activation output

$$
A^{(l)}
=
g
\left(
Z^{(l)}
\right)
$$

Shapes

### Layer 1

$$
A^{(1)}
\in
\mathbb{R}^{128\times n}
$$

### Layer 2

$$
A^{(2)}
\in
\mathbb{R}^{64\times n}
$$

### Layer 3

$$
A^{(3)}
\in
\mathbb{R}^{10\times n}
$$

---

## Ground Truth Labels

Ground truth labels

$$
Y
\in
\mathbb{R}^{10\times n}
$$

---

# Cost Function

The cost is

$$
C
=
\mathcal{L}
\left(
A^{(3)},
Y
\right)
$$

where

$$
C
\in
\mathbb{R}
$$

The cost is a scalar.

---

# Gradient Notation

Gradient of cost with respect to any variable

$$
\frac{\partial C}{\partial X}
$$

Examples

$$
\frac{\partial C}{\partial W^{(1)}}
$$

$$
\frac{\partial C}{\partial W^{(2)}}
$$

$$
\frac{\partial C}{\partial W^{(3)}}
$$

$$
\frac{\partial C}{\partial b^{(1)}}
$$

$$
\frac{\partial C}{\partial b^{(2)}}
$$

$$
\frac{\partial C}{\partial b^{(3)}}
$$

---

# Matrix Multiplication

Matrix multiplication is denoted by

$$
AB
$$

or

$$
A\cdot B
$$

Example

$$
(128\times784)
(784\times n)
=
(128\times n)
$$

---

# Hadamard Product

Element-wise multiplication is denoted by

$$
A
\odot
B
$$

Both matrices must have identical shapes.

Example

$$
(64\times n)
\odot
(64\times n)
=
(64\times n)
$$

---

# Dot Product

For vectors

$$
a\cdot b
=
\sum_i a_ib_i
$$

---

# Outer Product

The outer product is

$$
ab^T
$$

If

$$
a
\in
\mathbb{R}^{m\times1}
$$

and

$$
b
\in
\mathbb{R}^{n\times1}
$$

then

$$
ab^T
\in
\mathbb{R}^{m\times n}
$$

---

# Transpose

Transpose is written as

$$
A^T
$$

Example

$$
W^{(3)}
\in
\mathbb{R}^{10\times64}
$$

then

$$
(W^{(3)})^T
\in
\mathbb{R}^{64\times10}
$$

---

# Chain Rule

For composite functions

$$
\frac{\partial C}{\partial x}
=
\frac{\partial C}{\partial y}
\frac{\partial y}{\partial z}
\frac{\partial z}{\partial x}
$$

Backpropagation is simply repeated application of the chain rule through every layer.

---

# Forward Propagation

The forward propagation computes the prediction of the neural network by propagating the input through each layer.

The network architecture is

$$
784
\rightarrow
128
\rightarrow
64
\rightarrow
10
$$

---

# Overall Forward Pass

```text
                    W¹ (128 × 784)
               ─────────────────────►
                    b¹ (128 × 1)

X = A⁰ (784 × n)
        │
        ▼
Z¹ = W¹A⁰ + b¹
        │
        ▼
A¹ = g(Z¹)

                    W² (64 × 128)
               ───────────────────►
                    b² (64 × 1)

        │
        ▼
Z² = W²A¹ + b²
        │
        ▼
A² = g(Z²)

                    W³ (10 × 64)
               ─────────────────►
                    b³ (10 × 1)

        │
        ▼
Z³ = W³A² + b³
        │
        ▼
A³ = Softmax(Z³)
        │
        ▼
Cost = C(A³,Y)
```

---

# Layer 1

Input

$$
A^{(0)}=X
$$

Shape

$$
A^{(0)}
\in
\mathbb{R}^{784\times n}
$$

Weight Matrix

$$
W^{(1)}
\in
\mathbb{R}^{128\times784}
$$

Bias

$$
b^{(1)}
\in
\mathbb{R}^{128\times1}
$$

Linear Transformation

$$
Z^{(1)}
=
W^{(1)}
A^{(0)}
+
b^{(1)}
$$

Shape Verification

$$
(128\times784)
(784\times n)
+
(128\times1)
=
(128\times n)
$$

Therefore

$$
Z^{(1)}
\in
\mathbb{R}^{128\times n}
$$

Activation

$$
A^{(1)}
=
g
\left(
Z^{(1)}
\right)
$$

Shape

$$
A^{(1)}
\in
\mathbb{R}^{128\times n}
$$

---

# Layer 2

Input

$$
A^{(1)}
\in
\mathbb{R}^{128\times n}
$$

Weight Matrix

$$
W^{(2)}
\in
\mathbb{R}^{64\times128}
$$

Bias

$$
b^{(2)}
\in
\mathbb{R}^{64\times1}
$$

Linear Transformation

$$
Z^{(2)}
=
W^{(2)}
A^{(1)}
+
b^{(2)}
$$

Shape Verification

$$
(64\times128)
(128\times n)
+
(64\times1)
=
(64\times n)
$$

Therefore

$$
Z^{(2)}
\in
\mathbb{R}^{64\times n}
$$

Activation

$$
A^{(2)}
=
g
\left(
Z^{(2)}
\right)
$$

Shape

$$
A^{(2)}
\in
\mathbb{R}^{64\times n}
$$

---

# Layer 3 (Output Layer)

Input

$$
A^{(2)}
\in
\mathbb{R}^{64\times n}
$$

Weight Matrix

$$
W^{(3)}
\in
\mathbb{R}^{10\times64}
$$

Bias

$$
b^{(3)}
\in
\mathbb{R}^{10\times1}
$$

Linear Transformation

$$
Z^{(3)}
=
W^{(3)}
A^{(2)}
+
b^{(3)}
$$

Shape Verification

$$
(10\times64)
(64\times n)
+
(10\times1)
=
(10\times n)
$$

Therefore

$$
Z^{(3)}
\in
\mathbb{R}^{10\times n}
$$

Output Activation

$$
A^{(3)}
=
\operatorname{Softmax}
\left(
Z^{(3)}
\right)
$$

Shape

$$
A^{(3)}
\in
\mathbb{R}^{10\times n}
$$

---

# Network Prediction

The prediction of the neural network is

$$
\hat{Y}
=
A^{(3)}
$$

where

$$
\hat{Y}
\in
\mathbb{R}^{10\times n}
$$

---

# Cost Function

The prediction is compared with the true labels

$$
Y
\in
\mathbb{R}^{10\times n}
$$

The cost is

$$
C
=
\mathcal{L}
\left(
A^{(3)},
Y
\right)
$$

where

$$
C
\in
\mathbb{R}
$$

The forward propagation ends after computing the cost. Backpropagation starts from this scalar cost and propagates gradients backward through the network.

# Backpropagation (Delta Notation)

The forward propagation computes activations, while backpropagation computes the gradients of the cost function with respect to every trainable parameter.

The central quantity in backpropagation is the **error vector** (delta).

Define

$$
\boxed{
\delta^{(l)}
=
\frac{\partial C}
{\partial Z^{(l)}}
}
$$

where

$$
\delta^{(l)}
\in
\mathbb{R}^{n_l\times n}
$$

---

# Overall Backward Pass

```text
                     Cost
                      │
                      ▼
                    δ³
             ┌────────┴────────┐
             ▼                 ▼
        dW³ = δ³(A²)ᵀ      db³ = Σδ³
             │
             ▼
      dA² = (W³)ᵀδ³
             │
             ▼
            δ²
      = dA² ⊙ g'(Z²)
             │
      ┌──────┴──────┐
      ▼             ▼
 dW² = δ²(A¹)ᵀ   db² = Σδ²
      │
      ▼
dA¹ = (W²)ᵀδ²
      │
      ▼
     δ¹
= dA¹ ⊙ g'(Z¹)
      │
 ┌────┴─────┐
 ▼          ▼
dW¹       db¹
```

---

# Output Layer

## Step 1

The output error is

$$
\boxed{
\delta^{(3)}
=
\frac{\partial C}
{\partial A^{(3)}}
\odot
g'
\left(
Z^{(3)}
\right)
}
$$

Shape

$$
(10\times n)
\odot
(10\times n)
=
(10\times n)
$$

Therefore

$$
\delta^{(3)}
\in
\mathbb{R}^{10\times n}
$$

---

## Step 2

Weight gradient

$$
\boxed{
\frac{\partial C}
{\partial W^{(3)}}
=
\delta^{(3)}
(A^{(2)})^T
}
$$

Shape

$$
(10\times n)
(n\times64)
=
(10\times64)
$$

---

## Step 3

Bias gradient

$$
\boxed{
\frac{\partial C}
{\partial b^{(3)}}
=
\sum_{i=1}^{n}
\delta_i^{(3)}
}
$$

Shape

$$
(10\times1)
$$

---

## Step 4

Previous activation gradient

$$
\boxed{
\frac{\partial C}
{\partial A^{(2)}}
=
(W^{(3)})^T
\delta^{(3)}
}
$$

Shape

$$
(64\times10)
(10\times n)
=
(64\times n)
$$

---

# Hidden Layer 2

## Step 1

Hidden layer error

$$
\boxed{
\delta^{(2)}
=
\frac{\partial C}
{\partial A^{(2)}}
\odot
g'
\left(
Z^{(2)}
\right)
}
$$

Shape

$$
(64\times n)
\odot
(64\times n)
=
(64\times n)
$$

---

## Step 2

Weight gradient

$$
\boxed{
\frac{\partial C}
{\partial W^{(2)}}
=
\delta^{(2)}
(A^{(1)})^T
}
$$

Shape

$$
(64\times n)
(n\times128)
=
(64\times128)
$$

---

## Step 3

Bias gradient

$$
\boxed{
\frac{\partial C}
{\partial b^{(2)}}
=
\sum_{i=1}^{n}
\delta_i^{(2)}
}
$$

Shape

$$
(64\times1)
$$

---

## Step 4

Previous activation gradient

$$
\boxed{
\frac{\partial C}
{\partial A^{(1)}}
=
(W^{(2)})^T
\delta^{(2)}
}
$$

Shape

$$
(128\times64)
(64\times n)
=
(128\times n)
$$

---

# Hidden Layer 1

## Step 1

Hidden layer error

$$
\boxed{
\delta^{(1)}
=
\frac{\partial C}
{\partial A^{(1)}}
\odot
g'
\left(
Z^{(1)}
\right)
}
$$

Shape

$$
(128\times n)
\odot
(128\times n)
=
(128\times n)
$$

---

## Step 2

Weight gradient

$$
\boxed{
\frac{\partial C}
{\partial W^{(1)}}
=
\delta^{(1)}
(A^{(0)})^T
}
$$

Shape

$$
(128\times n)
(n\times784)
=
(128\times784)
$$

---

## Step 3

Bias gradient

$$
\boxed{
\frac{\partial C}
{\partial b^{(1)}}
=
\sum_{i=1}^{n}
\delta_i^{(1)}
}
$$

Shape

$$
(128\times1)
$$

---

# Final Gradient Set

After backpropagation, the complete gradient set is

$$
\boxed{
\left\{
\frac{\partial C}{\partial W^{(1)}},
\frac{\partial C}{\partial b^{(1)}},
\frac{\partial C}{\partial W^{(2)}},
\frac{\partial C}{\partial b^{(2)}},
\frac{\partial C}{\partial W^{(3)}},
\frac{\partial C}{\partial b^{(3)}}
\right\}
}
$$

These gradients are then passed to the optimization algorithm (e.g., Gradient Descent, Momentum, RMSProp, or Adam) to update the network parameters.