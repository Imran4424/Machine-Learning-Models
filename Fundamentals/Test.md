Here’s a careful walk-through of Question 4.

---

## 1. What is Question 4 asking?

The question says (paraphrased):

> Show that ridge regression admits the dual representation  
> $$
> \hat y_\lambda 
> = X (X^\top X + \lambda I_d)^{-1} X^\top y
> = K (K + \lambda I_n)^{-1} y,\quad K := XX^\top,
> $$
> where $I_d$ and $I_n$ are identity matrices of size $d \times d$ and $n \times n$.

Key objects:

- $X \in \mathbb{R}^{n \times d}$: rows are feature vectors $x_i^\top$.
- $y \in \mathbb{R}^n$: labels.
- Ridge regression solution:
  $$
  \hat\theta_\lambda = (X^\top X + \lambda I_d)^{-1} X^\top y.
  $$
- Prediction on the training points:
  $$
  \hat y_\lambda = X \hat\theta_\lambda.
  $$

The question wants you to show:

1. You can write $\hat y_\lambda$ purely in terms of $K = XX^\top \in \mathbb{R}^{n \times n}$ and $y$.
2. From that, conclude that predictions only depend on **inner products** $x_i^\top x_j$, not on the coordinates of $x_i$ themselves. This is the key idea behind **kernel methods**.

---

## 2. Step 1 – “Primal” form of ridge predictions

Ridge regression solution (you already saw earlier):

$$
\hat\theta_\lambda = (X^\top X + \lambda I_d)^{-1} X^\top y.
$$

Prediction on all training points at once:

$$
\hat y_\lambda = X \hat\theta_\lambda
= X (X^\top X + \lambda I_d)^{-1} X^\top y.
$$

This is the first expression in the question:

$$
\hat y_\lambda = X (X^\top X + \lambda I_d)^{-1} X^\top y.
$$

Here:

- $X$ is $n \times d$,
- $(X^\top X + \lambda I_d)^{-1}$ is $d \times d$,
- $X^\top$ is $d \times n$,
- so $X (X^\top X + \lambda I_d)^{-1} X^\top$ is $n \times n$; multiplying by $y \in \mathbb{R}^n$ gives an $n$-vector of predictions.

So far, everything is in terms of $X$ and the $d$-dimensional parameter vector.

---

## 3. Step 2 – Show the “dual” form with $K = XX^\top$

We want to show:

$$
X (X^\top X + \lambda I_d)^{-1} X^\top
\;=\;
K (K + \lambda I_n)^{-1},
\quad
K = XX^\top.
$$

If we prove this matrix identity, then multiplying both sides by $y$ will give

$$
\hat y_\lambda = K (K + \lambda I_n)^{-1} y,
$$

which is the desired dual representation.

### 3.1 Define $Z$

Let

$$
Z := X (X^\top X + \lambda I_d)^{-1} X^\top.
$$

We want to show $Z = K (K + \lambda I_n)^{-1}$.

A neat way: show that $Z (K + \lambda I_n) = K$. Then multiply both sides on the right by $(K + \lambda I_n)^{-1}$ to solve for $Z$.

So let’s compute:

$$
Z (K + \lambda I_n)
= X (X^\top X + \lambda I_d)^{-1} X^\top (XX^\top + \lambda I_n).
$$

Use $K = XX^\top$:

$$Z (K + \lambda I_n) = X (X^\top X + \lambda I_d)^{-1} X^\top K + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.$$

Now, $X^\top K = X^\top (XX^\top) = X^\top X X^\top$. So

$$Z (K + \lambda I_n) = X (X^\top X + \lambda I_d)^{-1} (X^\top X X^\top) + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.$$

Factor out $X$ and $X^\top$:

$$Z (K + \lambda I_n) = X \big[(X^\top X + \lambda I_d)^{-1} X^\top X\big] X^\top + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.$$


### 3.2 Use a key matrix identity

Let $A = X^\top X$. Then we have the identity

$$
(A + \lambda I)^{-1} A = I - \lambda (A + \lambda I)^{-1}.
$$

You can see this by starting from $(A + \lambda I)^{-1}(A + \lambda I) = I$ and expanding:

$$
(A + \lambda I)^{-1} A + \lambda (A + \lambda I)^{-1} = I
\quad\Rightarrow\quad
(A + \lambda I)^{-1} A = I - \lambda (A + \lambda I)^{-1}.
$$

Apply this with $A = X^\top X$:

$$
(X^\top X + \lambda I_d)^{-1} X^\top X
= I_d - \lambda (X^\top X + \lambda I_d)^{-1}.
$$

Now plug this into our expression:

$$Z (K + \lambda I_n) = X \big( I_d - \lambda (X^\top X + \lambda I_d)^{-1} \big) X^\top + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.$$

$$Z (K + \lambda I_n) = X X^\top - \lambda X (X^\top X + \lambda I_d)^{-1} X^\top + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.$$

$$Z (K + \lambda I_n) = X X^\top.$$

$$Z (K + \lambda I_n) = K.$$

The two terms with $\lambda$ cancel exactly.

So we have:

$$
Z (K + \lambda I_n) = K.
$$

Right-multiply both sides by $(K + \lambda I_n)^{-1}$:

$$
Z = K (K + \lambda I_n)^{-1}.
$$

Recall $Z = X (X^\top X + \lambda I_d)^{-1} X^\top$, so:

$$
X (X^\top X + \lambda I_d)^{-1} X^\top
= K (K + \lambda I_n)^{-1}.
$$

Multiplying both sides by $y$ gives:

$$
\hat y_\lambda
= X (X^\top X + \lambda I_d)^{-1} X^\top y
= K (K + \lambda I_n)^{-1} y.
$$

That is exactly the dual representation the question asked you to show.

---

## 4. Step 3 – Why does this mean predictions depend only on inner products?

So far, for the **training points**, we have:

$$
\hat y_\lambda = K (K + \lambda I_n)^{-1} y.
$$

Define

$$
\alpha := (K + \lambda I_n)^{-1} y \in \mathbb{R}^n.
$$

Then

$$
\hat y_\lambda = K \alpha.
$$

The $i$-th entry of $\hat y_\lambda$ is

$$
\hat y_{\lambda,i}
= (K \alpha)_i
= \sum_{j=1}^n K_{ij} \alpha_j
= \sum_{j=1}^n (x_i^\top x_j)\, \alpha_j.
$$

So the prediction for training point $x_i$ is a linear combination of **inner products** $x_i^\top x_j$ with other training points $x_j$.

For a **new test point** $x \in \mathbb{R}^d$:

- “Primal” formula:
  $$
  \hat y_\lambda(x)
  = x^\top \hat\theta_\lambda
  = x^\top (X^\top X + \lambda I_d)^{-1} X^\top y.
  $$
- Using the dual representation and $\alpha$, we can show
  $$
  \hat y_\lambda(x)
  = k(x)^\top \alpha,
  $$
  where $k(x) \in \mathbb{R}^n$ has entries
  $$
  k(x)_i = x_i^\top x.
  $$

So **everything** can be written in terms of:

- $K_{ij} = x_i^\top x_j$ (inner products between training points),
- $k(x)_i = x_i^\top x$ (inner products between training point $x_i$ and test point $x$).

No step requires the raw coordinates of $x_i$; only their dot products appear.

That’s exactly what the question wants you to “deduce”: ridge predictions depend only on **inner products of features**.

---

## 5. Why is this important for kernels?

Once your algorithm only needs inner products $x_i^\top x_j$, you can **replace** that inner product with a *kernel* function $k(x_i, x_j)$ that behaves like an inner product in some (possibly high-dimensional) feature space, without ever computing that feature map explicitly.

- Replace $K_{ij} = x_i^\top x_j$ with $K_{ij} = k(x_i, x_j)$.
- Replace $k(x)_i = x_i^\top x$ with $k(x)_i = k(x_i, x)$.

This gives **kernel ridge regression**, a non-linear version of ridge regression.

That’s why showing the dual form and the “inner products only” dependency is a key step toward understanding kernel methods.
