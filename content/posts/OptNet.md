---
date: '2024-06-28T19:41:37-04:00'
draft: false
params:
  math: true
title: 'A Note on OptNet'
---

Not long ago, I came across [this interesting paper](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf), which explored the possibility of training a neural network
that embeds a constrained quadratic optimization problem as one of its layers. I was immediately hooked on the idea
and began exploring it in the paper. However, I wouldn't describe the experience of reading that paper as
enjoyable, mainly because the authors assumed readers were already familiar with some theorems from real analysis, and
many derivation details for the results presented in the paper were missing. I did a lot of research and homework to 
equip myself with the necessary knowledge to understand that paper. The whole process took several weeks, and it is this 
lengthy process that drives me to write this blog, which provides detailed explanations of how the results were derived, 
for people who also have interest in this paper.

Before you start, I highly recommend reading [this blog](https://implicit-layers-tutorial.org/) first (chapter 1 should
be enough). It provides a clear explanation of the Implicit Function Theorem, a mathematical tool that is essential for 
grasping the idea behind the paper. In addition, you might want to do some review on constrained optimization theory.

## 1. Implicit function

We started by noticing the problem
\[
\begin{aligned}
& \underset{z}{\text{minimize}}
& & \frac{1}{2}z^TQz + q^T z \\
& \text{subject to}
& & Az = b, \\ 
& & & Gz \leq h, \\
\end{aligned}
\]
is a convex optimization problem and the KKT conditions are sufficient for global
optimality. Thus, at optimality, we have
\[
\begin{bmatrix}
Qz^* + q + A^T v^* + G^T \lambda^* \\
D(\lambda^*)(Gz^* - h) \\
Az^* - b
\end{bmatrix} = \mathbf{0},
\]
where $D(\cdot)$ creates a diagonal matrix from a vector and $z^*$, $\lambda^*$, $v^*$, are the optimal primal and dual
variables. Observe that $\{Q, q, A, b, G, h\}$ are parameters of the above vector-valued function, and for each
different set of parameters, there exists a unique set of optimal primal and dual variables. By leveraging the implicit
function theorem, an implicit function $F(x, \theta^*(x))$ can be defined as
\[
\label{eq:implicit function}
F(x, \theta^*(x)) =
\begin{bmatrix}
Qz^* + q + A^T v^* + G^T \lambda^* \\
D(\lambda^*)(Gz^* - h) \\
Az^* - b
\end{bmatrix}
\in \mathbb{R}^{n+m+p},
\]
with $x$ and $\theta^*(x)$ defined as
\[
\begin{aligned}
&x = \left[ \text{vec}(Q)^T, q^T, \text{vec}(A)^T, b^T, \text{vec}(G)^T, h^T \right]^T \in
\mathbb{R}^{n^2+n+mn+m+pn+p} \\
&\theta^*(x) = \left[ z^*(x), \lambda^*(x), v^*(x) \right]^T \in \mathbb{R}^{n+m+p}.
\end{aligned}
\]
To align with the function definition in the implicit function theorem and to avoid dealing with tensors in matrix
differentiation, we have vectorized the matrices to vectors in a column-major order. For example, matrix $Q$ is
expressed as a column vector $\text{vec}(Q) =
\left[ Q_{11}, \ldots Q_{n1}, Q_{12}, \ldots Q_{n2} \ldots Q_{1n} \ldots Q_{nn}\right]^T \in \mathbb{R}^{n^2}$.
Given that $F(x, \theta^*(x)) = \mathbf{0}$ at optimality, by differentiating both sides with respect to $x$ we obtain
\[
\frac{\partial F(x, \theta^*(x))}{\partial x} = \frac{\partial F(x, \theta^*)}{\partial x} + \frac{\partial F(x,
\theta^*)}{\partial \theta^*} \frac{\partial \theta^*(x)}{\partial x} = \mathbf{0}.
\]
The essential components $\frac{\partial \theta^*(x)}{\partial x}$ required for gradient computation can therefore be
obtained by
\[
\label{eq:implicit gradient}
\frac{\partial \theta^*(x)}{\partial x} = - \left( \frac{\partial F(x, \theta^*)}{\partial \theta^*} \right)^{-1}
\frac{\partial F(x, \theta^*)}{\partial x}.
\]

## 2. Component Derivation

In this section, we provide detailed derivation for each component of the last equation above. By applying
basic matrix calculus results, we immediately obtain
\[
\frac{\partial F(x, \theta^*)}{\partial \theta^*} =
\begin{bmatrix}
Q & G^T & A^T \\
D(\lambda^*)G & D(Gz^*-h) & 0 \\
A & 0 & 0
\end{bmatrix}.
\]
The exact form of $\frac{\partial \theta^*(x)}{\partial x}$ cannot be directly derived due to its implicit nature, and
the derivation of $\frac{\partial \theta^*(x)}{\partial x}$ requires a component-wise treatment. We find it beneficial
to expand them in vector and matrix formats to aid readers' understanding of subsequent content. Therefore, we express
$\frac{\partial \theta^*(x)}{\partial x}$ as

\[
\frac{\partial \theta^*(x)}{\partial x} = 
\begin{bmatrix}
\frac{\partial z^*(x)}{\partial x} \\
\frac{\partial \lambda^*(x)}{\partial x} \\
\frac{\partial v^*(x)}{\partial x}
\end{bmatrix}=
\begin{bmatrix}
\frac{\partial z^*(x)}{\partial \text{vec}(Q)} & \cdots & \frac{\partial z^*(x)}{\partial h} \\
\frac{\partial \lambda^*(x)}{\partial \text{vec}(Q)} & \cdots & \frac{\partial \lambda^*(x)}{\partial h} \\
\frac{\partial v^*(x)}{\partial \text{vec}(Q)} & \cdots & \frac{\partial v^*(x)}{\partial h}
\end{bmatrix}.
\]
Similarly, the vector and matrix expansion for $\frac{\partial F(x, \theta^*)}{\partial x}$ is expressed as
\[
\label{eq:parameter matrix}
\frac{\partial F(x, \theta^*)}{\partial x} = 
\begin{bmatrix}
\frac{\partial F_1(x, \theta^*)}{\partial x} \\
\frac{\partial F_2(x, \theta^*)}{\partial x} \\
\frac{\partial F_3(x, \theta^*)}{\partial x}
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(Q)} & \cdots & \frac{\partial F_1(x, \theta^*)}{\partial h} \\
\frac{\partial F_2(x, \theta^*)}{\partial \text{vec}(Q)} & \cdots & \frac{\partial F_2(x, \theta^*)}{\partial h} \\
\frac{\partial F_3(x, \theta^*)}{\partial \text{vec}(Q)} & \cdots & \frac{\partial F_3(x, \theta^*)}{\partial h}
\end{bmatrix},
\]
where $F_1(x, \theta^*)$, $F_2(x, \theta^*)$, and $F_3(x, \theta^*)$ correspond to the $1_{st}$, $2_{nd}$, and $3_{rd}$
rows of function $F(x, \theta^*(x))$. Now, we perform derivation for each component of the last matrix above.

### $F_1(x, \theta^*)$
We immediately notice that in the $1_{st}$ row of matrix $\frac{\partial F(x, \theta^*)}{\partial x}$
\[
\begin{aligned}
\frac{\partial F_1(x, \theta^*)}{\partial q} &= \frac{\partial q}{\partial q} = I \\
\frac{\partial F_1(x, \theta^*)}{\partial b} &= \frac{\partial F_1(x, \theta^*)}{\partial h} = 0.
\end{aligned}
\]
$\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(Q)}$ can be expressed as
\[
\frac{\partial Qz^*}{\partial \text{vec}(Q)} =
\begin{bmatrix}
\frac{\partial (Qz^*)_1}{\partial \text{vec}(Q)} \\
\vdots \\
\frac{\partial (Qz^*)_n}{\partial \text{vec}(Q)}
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial (Qz^*)_1}{\partial Q_{11}} & \cdots & \frac{\partial (Qz^*)_1}{\partial Q_{nn}} \\
\vdots & \ddots & \vdots \\
\frac{\partial (Qz^*)_n}{\partial Q_{11}} & \cdots & \frac{\partial (Qz^*)_n}{\partial Q_{nn}}
\end{bmatrix}.
\]
The partial derivative of each component of this matrix can be expressed compactly as
\[
\frac{\partial (Qz^*)_i}{\partial Q_{jk}} = \frac{\partial \sum_{\ell=1}^{n} Q_{i\ell} z^*_\ell}{\partial Q_{jk}} = 
\begin{cases} 
z^*_k & \text{if } i = j \\
0 & \text{otherwise}
\end{cases},
\]
resulting in a matrix of form
\[
\begin{bmatrix}
z^*_1  & 0      & \cdots & 0      & \cdots & z^*_n  & 0      & \cdots & 0       \\
0      & z^*_1  & \cdots & 0      & \cdots & 0      & z^*_n  & \cdots & 0       \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots  \\
0      & 0      & \cdots & z^*_1  & \cdots & 0      & 0      & \cdots & z^*_n   \\
\end{bmatrix}.
\]
By using Kronecker product, we can express this matrix as
\[
\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(Q)} = (z^*)^T \otimes I_n \in \mathbb{R}^{n \times n^2}.
\]
$\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(A)}$ can be expressed as
\[
\frac{\partial A^T v^*}{\partial \text{vec}(A)} =
\begin{bmatrix}
\frac{\partial (A^T v^*)_1}{\partial \text{vec}(A)} \\
\vdots \\
\frac{\partial (A^T v^*)_n}{\partial \text{vec}(A)}
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial (A^T v^*)_1}{\partial A_{11}} & \cdots & \frac{\partial (A^T v^*)_1}{\partial A_{mn}} \\
\vdots & \ddots & \vdots \\
\frac{\partial (A^T v^*)_n}{\partial A_{11}} & \cdots & \frac{\partial (A^T v^*)_n}{\partial A_{mn}}
\end{bmatrix}.
\]
The partial derivative of each component of this matrix can be expressed compactly as
\[
\frac{\partial (A^T v^*)_i}{\partial A_{jk}} = \frac{\partial \sum_{\ell=1}^{m} A^T_{i\ell} v^*_\ell}{\partial A_{jk}} =
\frac{\partial \sum_{\ell=1}^{m} A_{\ell i} v^*_\ell}{\partial A_{jk}} =
\begin{cases} 
v^*_j & \text{if } i=k \\
0 & \text{otherwise}
\end{cases},
\]
resulting in a matrix of form
\[
\begin{bmatrix}
v^*_1  & \cdots & v^*_m  & 0      & \cdots & 0      & \cdots & 0      & \cdots & 0     \\
0      & \cdots & 0      & v^*_1  & \cdots & v^*_m  & \cdots & 0      & \cdots & 0     \\
\vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \ddots & \vdots\\
0      & \cdots & 0      & 0      & \cdots & 0      & \cdots & v^*_1  & \cdots & v^*_m \\
\end{bmatrix}.
\]
By using Kronecker product, we can express this matrix as
\[
\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(A)} = I_n \otimes (v^*)^T \in \mathbb{R}^{n \times mn}.
\]
Using the same derivation procedure of $\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(A)}$, we can show that
\[
\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(G)} = I_n \otimes (\lambda^*)^T \in \mathbb{R}^{n \times pn}.
\]

### $F_2(x, \theta^*)$

It can be readily seen that
\[
\begin{aligned}
\frac{\partial F_2(x, \theta^*)}{\partial \text{vec}(Q)} &= \frac{\partial F_2(x, \theta^*)}{\partial q} = \frac{\partial F_2(x, \theta^*)}{\partial \text{vec}(A)} = \frac{\partial F_2(x, \theta^*)}{\partial b} = 0 \\
\frac{\partial F_2(x, \theta^*)}{\partial h} &= \frac{\partial D(\lambda^*)h}{\partial h} = -D(\lambda^*).
\end{aligned}
\]
$\frac{\partial F_2(x, \theta^*)}{\partial \text{vec}(G)}$ can be expressed as
\[
\frac{\partial D(\lambda^*)Gz^*}{\partial \text{vec}(G)} =
\begin{bmatrix}
\frac{\partial \lambda^*_1(Gz^*)_1}{\partial \text{vec}(G)} \\
\vdots \\
\frac{\partial \lambda^*_p(Gz^*)_p}{\partial \text{vec}(G)}
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial \lambda^*_1(Gz^*)_1}{\partial G_{11}} & \cdots & \frac{\partial \lambda^*_1(Gz^*)_1}{\partial G_{pn}} \\
\vdots & \ddots & \vdots \\
\frac{\partial \lambda^*_p(Gz^*)_p}{\partial G_{11}} & \cdots & \frac{\partial \lambda^*_p(Gz^*)_p}{\partial G_{pn}}
\end{bmatrix}.
\]
The partial derivative of each component of this matrix can be expressed compactly as
\[
\frac{\partial \lambda^*_i(Gz^*)_i}{\partial G_{jk}} = \frac{\partial \lambda^*_i\sum_{\ell=1}^{n} G_{i\ell} z^*_\ell}{\partial G_{jk}} = 
\begin{cases} 
\lambda^*_j z^*_k & \text{if } i = j \\
0 & \text{otherwise}
\end{cases},
\]
resulting in a matrix of form
\[
\begin{bmatrix}
\lambda^*_1 z^*_1  & 0                  & \cdots & 0                  & \cdots & \lambda^*_1 z^*_n  & 0                  & \cdots & 0       \\
0                  & \lambda^*_2 z^*_1  & \cdots & 0                  & \cdots & 0                  & \lambda^*_2 z^*_n  & \cdots & 0       \\
\vdots             & \vdots             & \ddots & \vdots             & \ddots & \vdots             & \vdots             & \ddots & \vdots  \\
0                  & 0                  & \cdots & \lambda^*_p z^*_1  & \cdots & 0                  & 0                  & \cdots & \lambda^*_p z^*_n   \\
\end{bmatrix}.
\]
By using Kronecker product, we can express this matrix as
\[
\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(G)} = (z^*)^T \otimes D(\lambda^*) \in \mathbb{R}^{p \times pn}.
\]

### $F_3(x, \theta^*)$

It can be readily seen that
\[
\begin{aligned}
\frac{\partial F_3(x, \theta^*)}{\partial \text{vec}(Q)} &= \frac{\partial F_3(x, \theta^*)}{\partial q} = \frac{\partial F_3(x, \theta^*)}{\partial \text{vec}(G)} = \frac{\partial F_3(x, \theta^*)}{\partial h} = 0 \\
\frac{\partial F_3(x, \theta^*)}{\partial b} &= -\frac{\partial b}{\partial b} = -I.
\end{aligned}
\]
By employing the same derivation procedure of $\frac{\partial F_1(x, \theta^*)}{\partial \text{vec}(Q)}$, we can also show that
\[
\frac{\partial F_3(x, \theta^*)}{\partial \text{vec}(A)} = (z^*)^T \otimes I_m \in \mathbb{R}^{m \times mn}.
\]

### Summary

By incorporating everything derived before, we obtain following $(n+p+m) \times (n^2+n+mn+m+pn+p)$ matrix:
\[
\frac{\partial F(x, \theta^*)}{\partial x} = 
\begin{bmatrix}
(z^*)^T \otimes I_n & I_n & I_n \otimes (v^*)^T & 0 & I_n \otimes (\lambda^*)^T & 0 \\
0 & 0 & 0 & 0 & (z^*)^T \otimes D(\lambda^*) & -D(\lambda^*) \\
0 & 0 & (z^*)^T \otimes I_m & -I_m & 0 & 0
\end{bmatrix},
\]
where, from left to right, each column corresponds to $\frac{\partial F(x, \theta^*)}{\partial \text{vec}(Q)}$,
$\frac{\partial F(x, \theta^*)}{\partial q}$, $\frac{\partial F(x, \theta^*)}{\partial \text{vec}(A)}$, $\frac{\partial
F(x, \theta^*)}{\partial b}$, $\frac{\partial F(x, \theta^*)}{\partial \text{vec}(G)}$, and $\frac{\partial F(x,
\theta^*)}{\partial h}$, respectively.

## 3. Gradient Computation

The gradients of loss function $\ell$ with respect to optimization problem parameters $x$ are computed by employing
chain rule. Assuming all vectors are column vectors, the gradient can be expressed as
\[
\label{eq:gradient1}
\begin{aligned}
\nabla_x \ell &= \left( \frac{\partial \ell}{\partial \theta^*} \right)^T \frac{\partial \theta^*(x)}{\partial x} \quad (\text{row vector}) \\
&= \left( \frac{\partial \theta^*(x)}{\partial x} \right)^T \frac{\partial \ell}{\partial \theta^*} \quad (\text{column vector}) \\
&= -\left( \frac{\partial F(x, \theta^*)}{\partial x} \right)^T \left( \frac{\partial F(x, \theta^*)}{\partial \theta^*} \right)^{-T} \frac{\partial \ell}{\partial \theta^*},
\end{aligned}
\]
where the last equality is obtained by employing result obtained in Section 1 and
$\frac{\partial \ell}{\partial \theta^*} = \left[ \frac{\partial \ell}{\partial z^*}, \frac{\partial \ell}{\partial
\lambda^*}, \frac{\partial \ell}{\partial v^*} \right]^T \in \mathbb{R}^{n+p+m}$.

During forward pass, only the output of optimization problem layer, i.e., the optimal primal variables $z^*$, is fed to
next layer, therefore there are only gradients with respect to $z^*$ and $\frac{\partial \ell}{\partial \theta^*} =
\left[ \frac{\partial \ell}{\partial z^*}, 0, 0\right]$. We define an intermediate vector
$\left[ d_z, d_{\lambda}, d_v\right]^T$ to represent the product of $\left( \frac{\partial F(x, \theta^*)}{\partial \theta^*} \right)^{-T} \frac{\partial \ell}{\partial \theta^*}$.
Using the results obtained in Section 2, the intermediate vector can be computed as
\[
\label{eq:7}
\begin{bmatrix}
d_z \\
d_{\lambda} \\
d_v
\end{bmatrix}
= -
\begin{bmatrix}
Q & G^T & D(\lambda^*) \\
G & D(Gz^*-h) & 0 \\
A & 0 & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
\frac{\partial \ell}{\partial z^*} \\
0 \\
0 \\
\end{bmatrix},
\]
which corresponds to equation (7) of the paper. In addition, the transpose of $\frac{\partial F(x, \theta^*)}{\partial
x}$ can be expressed as
\[
(\frac{\partial F(x, \theta^*)}{\partial x})^T = 
\begin{bmatrix}
z^* \otimes I_n & 0 & 0 \\
I_n & 0 & 0 \\
I_n \otimes v^* & 0 & z^* \otimes I_m \\
0 & 0 & -I_m \\
I_n \otimes \lambda^* & z^* \otimes D(\lambda^*) & 0 \\
0 & -D(\lambda^*) & 0
\end{bmatrix},
\]
where we have used Kronecker product properties to perform transpose. You can visit [here](https://en.wikipedia.org/wiki/Kronecker_product) 
for details about properties of Kronecker product. Finally, the gradients of loss function $\ell$ with respect to 
$\{\text{vec}(Q), q, \text{vec}(A), b, \text{vec}(G), h\}$ are computed by
\[
\nabla_x \ell = 
\begin{bmatrix}
\frac{\partial \ell}{\partial \text{vec}(Q)} \\
\vdots \\
\frac{\partial \ell}{\partial h}
\end{bmatrix}
= (\frac{\partial F(x, \theta^*)}{\partial x})^T
\begin{bmatrix}
d_z \\
d_{\lambda} \\
d_v
\end{bmatrix},
\]
which can also be expressed compactly as
\[
\label{eq:gradient2}
\begin{aligned}
\frac{\partial \ell}{\partial \text{vec}(Q)} &= (z^* \otimes I_n) d_z & \quad \frac{\partial \ell}{\partial q} &= I_n d_z = d_z \\
\frac{\partial \ell}{\partial \text{vec}(A)} &= (I_n \otimes v^*) d_z + (z^* \otimes I_m) d_v & \quad \frac{\partial \ell}{\partial b} &= -I_m d_v = -d_v \\
\frac{\partial \ell}{\partial \text{vec}(G)} &= (I_n \otimes \lambda^*) d_z + (z^* \otimes D(\lambda^*)) d_{\lambda} & \quad \frac{\partial \ell}{\partial h} &= -D(\lambda^*) d_{\lambda}.
\end{aligned}
\]
The second column already aligns with the second column of equation (8) in the paper. We now demonstrate that the first 
column can be further simplified to yield identical expressions too.

For $\frac{\partial \ell}{\partial \text{vec}(Q)}$, we expand $(z^* \otimes I_n) d_z$ as
\[
\begin{bmatrix}
z^*_1  & 0      & \cdots & 0 \\
0      & z^*_1  & \cdots & 0 \\
\vdots & \vdots & \ddots & 0 \\
0      & 0      & \cdots & z^*_1 \\
\vdots & \vdots & \vdots & \vdots \\
z^*_n  & 0      & \cdots & 0 \\
0      & z^*_n  & \cdots & 0 \\
\vdots & \vdots & \ddots & 0 \\
0      & 0      & \cdots & z^*_n
\end{bmatrix}
\begin{bmatrix}
(d_z)_1 \\
\vdots \\
(d_z)_n
\end{bmatrix} = 
\begin{bmatrix}
z^*_1(d_z)_1 \\
\vdots \\
z^*_1(d_z)_n \\
\vdots \\
z^*_n(d_z)_1 \\
\vdots \\
z^*_n(d_z)_n
\end{bmatrix}.
\]
Notice that this vector can be obtained by vectorizing a matrix of form
\[
\begin{bmatrix}
(d_z)_1 z^*_1 & \cdots & (d_z)_1 z^*_n \\
\vdots       & \ddots & \vdots \\
(d_z)_n z^*_1 & \cdots & (d_z)_n z^*_n
\end{bmatrix} = d_z(z^*)^T.
\] 
Therefore, $\frac{\partial \ell}{\partial \text{vec}(Q)} = \text{vec}(d_z(z^*)^T)$, and $\frac{\partial
\ell}{\partial Q} = d_z(z^*)^T$ after reshaping. Moreover, the positive semidefiniteness of matrix $Q$ implies symmetry,
which further implies $\frac{\partial \ell}{\partial \text{vec}(Q)} = \frac{\partial \ell}{\partial \text{vec}(Q^T)}$.
Applying almost the same derivation procedure as for $\frac{\partial \ell}{\partial \text{vec}(Q)}$, we obtain
$\frac{\partial \ell}{\partial \text{vec}(Q^T)} = (I_n \otimes z^*) d_z$, which can be expanded as
\[
\begin{bmatrix}
z^*_1  & \cdots & 0 \\
z^*_2  & \cdots & 0 \\
\vdots & \ddots & \vdots\\
z^*_n  & \cdots & 0 \\
\vdots & \ddots & \vdots\\
0      & \cdots & z^*_1 \\
0      & \cdots & z^*_2 \\
\vdots & \ddots & \vdots\\
0      & \cdots & z^*_n
\end{bmatrix}
\begin{bmatrix}
(d_z)_1 \\
\vdots \\
(d_z)_n
\end{bmatrix} = 
\begin{bmatrix}
z^*_1(d_z)_1 \\
\vdots \\
z^*_n(d_z)_1 \\
\vdots \\
z^*_1(d_z)_n \\
\vdots \\
z^*_n(d_z)_n
\end{bmatrix},
\]
Notice that this vector can be obtained by vectorizing a matrix of form
\[
\begin{bmatrix}
z^*_1 (d_z)_1 & \cdots & z^*_1 (d_z)_n \\
\vdots       & \ddots & \vdots \\
z^*_n (d_z)_1 & \cdots & z^*_n (d_z)_n
\end{bmatrix} = z^* d_z^T.
\]
Therefore, $\frac{\partial \ell}{\partial \text{vec}(Q^T)} = \text{vec}(z^* d_z^T)$, and $\frac{\partial \ell}{\partial
Q^T} = z^* d_z^T$ after reshaping. The symmetry suggests that $\text{vec}(d_z(z^*)^T) = \text{vec}(z^* d_z^T)$ and thus
$d_z(z^*)^T = z^* d_z^T$. By decomposing square matrix $d_z(z^*)^T$ into its symmetric and skew-symmetric part, we
obtain
\[
\begin{aligned}
d_z(z^*)^T &= \frac{1}{2}(d_z(z^*)^T + (d_z(z^*)^T)^T) + \frac{1}{2}(d_z(z^*)^T - (d_z(z^*)^T)^T) \\
&= \frac{1}{2}(d_z(z^*)^T + z^* d_z^T) + \frac{1}{2}(d_z(z^*)^T - z^* d_z^T) \\
&= \frac{1}{2}(d_z(z^*)^T + z^* d_z^T),
\end{aligned}
\]
from which we can conclude that $\frac{\partial \ell}{\partial Q} = \frac{1}{2}(d_z(z^*)^T + z^* d_z^T)$.

For $\frac{\partial \ell}{\partial \text{vec}(A)}$, by applying the same expansion and vectorization procedure as for
$\frac{\partial \ell}{\partial \text{vec}(Q)}$, we obtain
\[
\begin{aligned}
\frac{\partial \ell}{\partial \text{vec}(A)} &= (I_n \otimes v^*) d_z + (z^* \otimes I_m) d_v \\
&= vec(v^* d_z^T) + vec(d_v (z^*)^T).
\end{aligned}
\]
Therefore, $\frac{\partial \ell}{\partial A} = v^* d_z^T + d_v (z^*)^T$ after reshaping.

For $\frac{\partial \ell}{\partial \text{vec}(G)}$, the derivation of the first terms follows the same procedure as
before, while the expansion of the second term requires a slight modification. Expanding second term $(z^* \otimes D(
\lambda^*)) d_{\lambda}$ yields
\[
\begin{bmatrix}
z^*_1 \lambda^*_1  & 0      & \cdots & 0 \\
0      & z^*_1 \lambda^*_2  & \cdots & 0 \\
\vdots & \vdots & \ddots & 0 \\
0      & 0      & \cdots & z^*_1 \lambda^*_p \\
\vdots & \vdots & \vdots & \vdots \\
z^*_n \lambda^*_1  & 0      & \cdots & 0 \\
0      & z^*_n \lambda^*_2  & \cdots & 0 \\
\vdots & \vdots & \ddots & 0 \\
0      & 0      & \cdots & z^*_n \lambda^*_p
\end{bmatrix}
\begin{bmatrix}
(d_{\lambda})_1 \\
\vdots \\
(d_{\lambda})_p
\end{bmatrix} = 
\begin{bmatrix}
z^*_1 \lambda^*_1 (d_{\lambda})_1 \\
\vdots \\
z^*_1 \lambda^*_p (d_{\lambda})_p \\
\vdots \\
z^*_n \lambda^*_1 (d_{\lambda})_1\\
\vdots \\
z^*_n \lambda^*_p (d_{\lambda})_p
\end{bmatrix},
\]
which is a vector that is obtained by vectorizing a matrix of form
\[
\begin{aligned}
\begin{bmatrix}
z^*_1 \lambda^*_1 (d_{\lambda})_1 & \cdots & z^*_n \lambda^*_1 (d_{\lambda})_1 \\
\vdots       & \ddots & \vdots \\
z^*_1 \lambda^*_p (d_{\lambda})_p & \cdots & z^*_n \lambda^*_p (d_{\lambda})_p
\end{bmatrix} &= 
\begin{bmatrix}
\lambda^*_1  & 0      & \cdots & 0 \\
0      & \lambda^*_2  & \cdots & 0 \\
\vdots & \vdots & \ddots & 0 \\
0      & 0      & \cdots & \lambda^*_p \\
\end{bmatrix}
\begin{bmatrix}
z^*_1 (d_{\lambda})_1 & \cdots & z^*_n (d_{\lambda})_1 \\
\vdots       & \ddots & \vdots \\
z^*_1 (d_{\lambda})_p & \cdots & z^*_n (d_{\lambda})_p
\end{bmatrix} \\ 
&= D(\lambda^*) d_{\lambda} (z^*)^T.
\end{aligned}
\]
Incorporating the first term, we obtain
\[
\begin{aligned}
\frac{\partial \ell}{\partial \text{vec}(G)} &= (I_n \otimes \lambda^*) d_z + (z^* \otimes D(\lambda^*)) d_{\lambda} \\
&= vec(\lambda^* d_z^T) + vec(D(\lambda^*) d_{\lambda} (z^*)^T).
\end{aligned}
\]
Therefore, $\frac{\partial \ell}{\partial G} = \lambda^* d_z^T + D(\lambda^*) d_{\lambda} (z^*)^T$ after reshaping.

## 4. Gradient in Primal-Dual Interior Point Method

We first obtain the symmetrized version of matrix $K$ at optimality by scaling the $2_{nd}$ row by $D(1/s^*)$, which yields
\[
K_{sym} = 
\begin{bmatrix}
Q & 0 & G^T & A^T \\
0 & D(\lambda^*/s^*) & I & 0 \\
G & I & 0 & 0 \\
A & 0 & 0 & 0 \\
\end{bmatrix}.
\]
Here, we have used the fact that $D(\lambda^*)D(1/s^*) = D(\lambda^*/s^*)$ and $D(s^*)D(1/s^*) = I$. The introduction of
extra slack variable $s$ does not prevent us from using equation (8) of the paper to compute backpropagated gradients.
This can be seen by noticing
\[
K_{sym} =
\begin{bmatrix}
d_z \\
d_s \\
\tilde{d}_{\lambda} \\
d_v
\end{bmatrix} = 
\begin{bmatrix}
Q d_z + G^T \tilde{d}_{\lambda} + A^T d_v \\
D(\lambda^*/s^*) d_s + \tilde{d}_{\lambda} \\ 
G d_z + d_s \\
A d_z
\end{bmatrix}.
\]
Substituting $\tilde{d}_{\lambda}$ with $D(\lambda^*) d_{\lambda}$, we obtain
\[
\begin{bmatrix}
Q d_z + G^T D(\lambda^*) d_{\lambda} + A^T d_v \\
d_s + \left(D(\lambda^*/s^*)\right)^{-1} D(\lambda^*) d_{\lambda} \\
G d_z + d_s \\
A d_z
\end{bmatrix} = 
\begin{bmatrix}
-\frac{\partial \ell}{\partial z_{i+1}} \\
0 \\
0 \\
0
\end{bmatrix},
\]
from which we can see that the $2_{nd}$ and $3_{rd}$ rows together yields
\[
\begin{aligned}
G d_z + d_s &= G d_z - \left(D(\lambda^*/s^*)\right)^{-1} D(\lambda^*) d_{\lambda} \\
&= G d_z - D(s^*) d_{\lambda} \\
&= G d_z + D(-s^*) d_{\lambda} \\
&= G d_z + D(G z^* - h) d_{\lambda},
\end{aligned}
\]
and $z^* = z_{i+1}$. This means that solving equation (11) of the paper is equivalent of solving
\[
\begin{bmatrix}
Q d_z + G^T D(\lambda^*) d_{\lambda} + A^T d_v \\
G d_z + D(G z^* - h) d_{\lambda} \\
A d_z
\end{bmatrix} = 
\begin{bmatrix}
-\frac{\partial \ell}{\partial z^*} \\
0 \\
0
\end{bmatrix},
\]
which is basically the equation (7) of the paper with terms rearranged.

The use of primal-dual interior point method and symmetrized version of matrix $K$ allows a batch of quadratic
optimization problems with same form of the problem the paper consider to be solved in parallel. The gradients
computed during this optimization process can be reused to compute backward gradients using equation (11) in the paper.
These features make OptNet training faster and more resource-efficient.

The remaining part of Section 3 of the paper contains relatively more details than other parts. The three theorems are
proved with great details in the Supplementary Material section, we therefore refer readers to the paper for more
details.