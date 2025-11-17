# Question 1

## What the question is asking

You fit an Ordinary Least Squares (OLS) regression:

$$
\boldsymbol{\hat{y} = Hy}
$$

where,

$$
\boldsymbol{H = X (X^T X)^{-1} X^T}
$$

# Question 2

## What the question wants

You have a linear model with noisy data. It asks you to:

1) Write the **ridge** solution for the parameters.  
2) For a **new input** $x$, find:
   - the **average** prediction (its mean),
   - how much that prediction **varies** if you re-sample noisy data (its variance),
   - the **expected squared error** of predicting the true $y$ at $x$ (split into bias, variance, and irreducible noise).

---

## The answer explained plainly

### 1) What ridge is (and the formula you use)

- Ordinary least squares can overfit when features are collinear or noisy.
- **Ridge** puts a small penalty on big coefficients to stabilize the fit.

**You minimize**

$$
\|y - X\theta\|^2 + \lambda\|\theta\|^2
$$

and the **closed-form solution** is

$$
\hat{\theta}_\lambda = (X^\top X + \lambda I)^{-1} X^\top y
$$

- $\lambda \ge 0$ is the knob:  
  - $\lambda = 0$ → ordinary least squares.  
  - Larger $\lambda$ → more shrinkage (smaller coefficients).

---

### 2) Predictions for a new point $x$

#### 2a) Mean (average) of the prediction

Your predictor is $\hat y(x)=x^\top\hat\theta_\lambda$. Because $\hat\theta_\lambda$ is a **linear** function of $y$, its average over many noisy datasets is easy to compute:

$$
\mathbb{E}[\hat{y}(x)] = x^\top (X^\top X + \lambda I)^{-1} X^\top X\, \theta^\star
$$


Interpretation:

- Define $B_\lambda = (X^\top X + \lambda I)^{-1} X^\top X$.  
- Then $\mathbb{E}[\hat{y}(x)] = x^\top B_\lambda \theta^\star$.  
- When $\lambda = 0$, $B_\lambda = I$ (no shrinkage, unbiased).  
- As $\lambda$ grows, $B_\lambda$ pulls the mean prediction **toward 0** (more bias).


#### 2b) Variance (how wiggly the prediction is)
How much would $\hat y(x)$ change if you re-collected the same $X$ with new noise in $y$? 

That’s the variance:

$$
{Var}(\hat{y}(x)) = \sigma^2\, x^\top (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1} x
$$


Interpretation:

- As $\lambda$ increases, the middle matrix is **squeezed**, so the variance **drops**.  
- This is the payoff of ridge: less wiggly predictions (lower variance).

#### 2c) Expected squared error at $x$

Now include the **label noise** of the new point itself, say $y = x^T \theta^\star$ + $\tilde{\varepsilon}$ with $\tilde{\varepsilon}$ $\sim \mathcal{N}(0,\zeta^2)$ (this is noise you can’t remove).


The classic decomposition is:

$$
E\big[(y-\hat y(x))^2\big] = \text{Bias}^2 + \text{Variance} + \zeta^2
$$

$\text{Bias}^2$ = systematic offset  
Variance = fit wobble  
$\zeta^2$ = unavoidable


Compute each piece:

- **Bias at $x$** = $x^T\theta^\star - E[\hat y(x)]$  

$$
x^T\theta^\star - x^T B_\lambda \theta^\star
= x^T(I - B_\lambda)\theta^\star
= \lambda\, x^T (X^T X + \lambda I)^{-1}\theta^\star.
$$


So, $\text{Bias}^2$:

$$
\big(\lambda\, x^T (X^T X + \lambda I)^{-1}\theta^\star\big)^2
$$

- **Variance at $x$** is exactly the variance from 2b.

- **Noise at $x$** is $\zeta^2$.

Putting them together:

$$
E\left[(y-\hat y(x))^2\right]=(\lambda,x^T(X^T X+\lambda I)^{-1}\theta^\star)^2+\sigma^2,x^T(X^T X+\lambda I)^{-1}X^T X(X^T X+\lambda I)^{-1}x+\zeta^2
$$

# Question 3

## 1. What is Question 3 really asking?

The question states:

> Let  
> $\hat\theta_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$.  
> Show that $\|\hat\theta_\lambda\|_2$ is non-increasing in $\lambda$.

Translated:

- $\hat\theta_\lambda$ is the **ridge regression** solution with regularization strength $\lambda \ge 0$.
- $\|\cdot\|_2$ is the usual Euclidean norm.
- “Non-increasing in $\lambda$” means: if $\lambda_1 < \lambda_2$, then 

$$
\|\hat\theta_{\lambda_2}\|_2 \le \|\hat\theta_{\lambda_1}\|_2  
$$

So the question is:

> As we increase the ridge penalty $\lambda$, can the size (length) of the parameter vector $\hat\theta_\lambda$ ever go up?

The expected answer: **No**. Ridge regression penalizes large coefficients, and mathematically you can show that as you increase $\lambda$, the norm of the solution vector cannot increase; it either stays the same or decreases.

A very simple 1D intuition:

Suppose $X$ is just one column (scalar feature) with value vector $x$, 
so, $X^\top X = \|x\|^2$ is a scalar.

Then

$$
\hat\theta_\lambda = \frac{x^\top y}{\|x\|^2 + \lambda}.
$$

As $\lambda$ increases, the denominator $\|x\|^2+\lambda$ grows, so the magnitude $|\hat\theta_\lambda|$ decreases.

In higher-dimensions, the same kind of effect happens in a more complicated matrix way.

The assignment wants you to show this **formally** for the general case.

---

## 2. Strategy of the official solution

The solution does this:

1. Look at the squared norm  
   $f(\lambda) = \|\hat\theta_\lambda\|_2^2$.
2. Express $f(\lambda)$ in a form where $\lambda$ appears in a simple scalar way.
3. Differentiate $f(\lambda)$ with respect to $\lambda$.
4. Show the derivative is always $\le 0$, hence $f(\lambda)$ is non-increasing, hence $\|\hat\theta_\lambda\|_2$ is non-increasing.

Why squared norm? Because $\|\hat\theta_\lambda\|_2^2$ is easier to differentiate and monotonicity is preserved:
- If $f(\lambda)$ is non-increasing and nonnegative, then $\sqrt{f(\lambda)}$ is also non-increasing.

---

## 3. Step-by-step through the solution

### 3.1 Define the function $f(\lambda)$

They define:

$$
f(\lambda) = \|\hat\theta_\lambda\|_2^2 = \hat\theta_\lambda^\top \hat\theta_\lambda.
$$

We know:

$$
\hat\theta_\lambda = (X^\top X + \lambda I)^{-1} X^\top y.
$$

Substitute that in:

$$
f(\lambda)
= \left[(X^\top X + \lambda I)^{-1} X^\top y\right]^\top
  \left[(X^\top X + \lambda I)^{-1} X^\top y\right].
$$

Now simplify this expression.

---

### 3.2 Use symmetry of $(X^\top X + \lambda I)$

The matrix $X^\top X$ is symmetric, and so is $X^\top X + \lambda I$. The inverse of a symmetric matrix is also symmetric.

So

$$
\left( (X^\top X + \lambda I)^{-1} \right)^\top
= (X^\top X + \lambda I)^{-1}.
$$

Then:

$$
\begin{aligned}
f(\lambda)
&= y^\top X (X^\top X + \lambda I)^{-1}
   (X^\top X + \lambda I)^{-1} X^\top y \\
&= y^\top X (X^\top X + \lambda I)^{-2} X^\top y.
\end{aligned}
$$

So we have:

$$
f(\lambda) = y^\top X (X^\top X + \lambda I)^{-2} X^\top y.
$$

This is already a nice expression, but still matrix-heavy. Next step: diagonalize $X^\top X$.

---

### 3.3 Spectral decomposition (eigendecomposition) of $X^\top X$

Because $X^\top X$ is symmetric and positive semidefinite, we can write:

$$
X^\top X = V \Lambda V^\top,
$$

where:

- $V$ is an orthogonal matrix ($V^\top V = VV^\top = I$).
- $\Lambda = \mathrm{diag}(\mu_1, \dots, \mu_d)$ is a diagonal matrix of eigenvalues $\mu_i \ge 0$.

Now:

$$
\begin{aligned}
X^\top X + \lambda I 
&= V \Lambda V^\top + \lambda I \\
&= V \Lambda V^\top + \lambda V V^\top \\
&= V (\Lambda + \lambda I) V^\top.
\end{aligned}
$$

So,

$$
(X^\top X + \lambda I)^{-2}
= \big[ V(\Lambda + \lambda I)V^\top \big]^{-2}
= V (\Lambda + \lambda I)^{-2} V^\top.
$$

Why? Because:

- $(VAV^\top)^{-1} = V A^{-1} V^\top$ for orthogonal $V$,
- and squaring the inverse simply squares the diagonal entries.

Now plug this back into $f(\lambda)$:

$$
f(\lambda)
= y^\top X \, V (\Lambda + \lambda I)^{-2} V^\top X^\top y.
$$

---

### 3.4 Rewrite using a new vector $z$

Define:

$$
z = V^\top X^\top y.
$$

Since $V$ is orthogonal, $z$ is just a rotated version of $X^\top y$. Then:

$$
f(\lambda)
= z^\top (\Lambda + \lambda I)^{-2} z.
$$

Because $(\Lambda + \lambda I)^{-2}$ is diagonal with entries $\frac{1}{(\mu_i + \lambda)^2}$, we can write:

$$
f(\lambda)
= \sum_{i=1}^d \frac{z_i^2}{(\mu_i + \lambda)^2}.
$$

This is now a **scalar sum over coordinates**, which is perfect for differentiation.

---

### 3.5 Differentiate $f(\lambda)$ with respect to $\lambda$

We have:

$$
f(\lambda) = \sum_{i=1}^d z_i^2 (\mu_i + \lambda)^{-2}.
$$

Differentiate term-by-term:

$$
\frac{d}{d\lambda} f(\lambda)
= \sum_{i=1}^d z_i^2 \cdot \frac{d}{d\lambda} (\mu_i + \lambda)^{-2}.
$$

But:

$$
\frac{d}{d\lambda} (\mu_i + \lambda)^{-2}
= -2 (\mu_i + \lambda)^{-3}.
$$

So:

$$
\begin{aligned}
f'(\lambda)
&= \sum_{i=1}^d z_i^2 \cdot \big(-2 (\mu_i + \lambda)^{-3}\big) \\
&= -2 \sum_{i=1}^d \frac{z_i^2}{(\mu_i + \lambda)^3}.
\end{aligned}
$$

Now look at the sign:

- $z_i^2 \ge 0$ for all $i$.
- $\mu_i \ge 0$ as eigenvalues of $X^\top X$.
- $\lambda > 0$ by ridge definition.
- So $(\mu_i + \lambda)^3 > 0$ for all $i$.

Therefore each term $\frac{z_i^2}{(\mu_i + \lambda)^3} \ge 0$, and hence:

$$
f'(\lambda) = -2 \sum_{i=1}^d \frac{z_i^2}{(\mu_i + \lambda)^3} \le 0.
$$

So the derivative is **non-positive for all $\lambda>0$**.

This implies:

- $f(\lambda) = \|\hat\theta_\lambda\|_2^2$ is a **non-increasing** function of $\lambda$.
- Therefore the norm $\|\hat\theta_\lambda\|_2 = \sqrt{f(\lambda)}$ is also non-increasing in $\lambda$.

That’s exactly what the question asked you to show.

---

## 4. Intuition recap

Conceptually, ridge regression solves:

$$
\hat\theta_\lambda = \arg\min_\theta \left( \|y - X\theta\|_2^2 + \lambda \|\theta\|_2^2 \right).
$$

- Increasing $\lambda$ means you **care more about keeping $\theta$ small** (stronger penalty).
- The linear algebra proof above formalizes that as $\lambda$ grows, the parameter vector cannot get larger in length; it either stays the same or shrinks.


# Question 4

## 1. What is Question 4 asking?

The question says (paraphrased):

> Show that ridge regression admits the dual representation
>
> $$
> \hat y_\lambda 
> = X (X^\top X + \lambda I_d)^{-1} X^\top y
> = K (K + \lambda I_n)^{-1} y,\quad K := XX^\top,
> $$
> 
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

$$
Z (K + \lambda I_n) = X (X^\top X + \lambda I_d)^{-1} X^\top K + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.
$$

Now, $X^\top K = X^\top (XX^\top) = X^\top X X^\top$. So

$$
Z (K + \lambda I_n) = X (X^\top X + \lambda I_d)^{-1} (X^\top X X^\top) + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.
$$

Factor out $X$ and $X^\top$:

$$
Z (K + \lambda I_n) = X \big[(X^\top X + \lambda I_d)^{-1} X^\top X\big] X^\top + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.
$$


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

$$
Z (K + \lambda I_n) = X \big( I_d - \lambda (X^\top X + \lambda I_d)^{-1} \big) X^\top + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.
$$

$$
Z (K + \lambda I_n) = X X^\top - \lambda X (X^\top X + \lambda I_d)^{-1} X^\top + \lambda X (X^\top X + \lambda I_d)^{-1} X^\top.
$$

$$
Z (K + \lambda I_n) = X X^\top.
$$

$$
Z (K + \lambda I_n) = K.
$$

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
