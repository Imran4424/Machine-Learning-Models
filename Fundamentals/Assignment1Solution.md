# Question 2

# What the question wants

You have a linear model with noisy data. It asks you to:

1) Write the **ridge** solution for the parameters.  
2) For a **new input** $x$, find:
   - the **average** prediction (its mean),
   - how much that prediction **varies** if you re-sample noisy data (its variance),
   - the **expected squared error** of predicting the true $y$ at $x$ (split into bias, variance, and irreducible noise).

---

# The answer explained plainly

## 1) What ridge is (and the formula you use)

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

## 2) Predictions for a new point $x$

### 2a) Mean (average) of the prediction

Your predictor is $\hat y(x)=x^\top\hat\theta_\lambda$. Because $\hat\theta_\lambda$ is a **linear** function of $y$, its average over many noisy datasets is easy to compute:

$$
\mathbb{E}[\hat{y}(x)] = x^\top (X^\top X + \lambda I)^{-1} X^\top X\, \theta^\star
$$


Interpretation:

- Define $B_\lambda = (X^\top X + \lambda I)^{-1} X^\top X$.  
- Then $\mathbb{E}[\hat{y}(x)] = x^\top B_\lambda \theta^\star$.  
- When $\lambda = 0$, $B_\lambda = I$ (no shrinkage, unbiased).  
- As $\lambda$ grows, $B_\lambda$ pulls the mean prediction **toward 0** (more bias).


### 2b) Variance (how wiggly the prediction is)
How much would $\hat y(x)$ change if you re-collected the same $X$ with new noise in $y$? 

That’s the variance:

$$
\operatorname{Var}(\hat{y}(x)) = \sigma^2\, x^\top (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1} x
$$


Interpretation:

- As $\lambda$ increases, the middle matrix is **squeezed**, so the variance **drops**.  
- This is the payoff of ridge: less wiggly predictions (lower variance).

### 2c) Expected squared error at $x$

Now include the **label noise** of the new point itself, say $y = x^T \theta^\star + \tilde{\varepsilon}$ with $\tilde{\varepsilon} \sim \mathcal{N}(0,\zeta^2)$ (this is noise you can’t remove).

The classic decomposition is:

$$
E\big[(y-\hat y(x))^2\big] = \text{Bias}^2 + \text{Variance} + \zeta^2
$$

Bias$^2$ = systematic offset  
Variance = fit wobble  
$\zeta^2$ = unavoidable


Compute each piece:

- **Bias at $x$** = $x^T\theta^\star - E[\hat y(x)]$  

  $$
  x^T\theta^\star - x^T B_\lambda \theta^\star
  = x^T(I - B_\lambda)\theta^\star
  = \lambda\, x^T (X^T X + \lambda I)^{-1}\theta^\star.
  $$

  So, Bias$^2$:
  $$
  \big(\lambda\, x^T (X^T X + \lambda I)^{-1}\theta^\star\big)^2
  $$

- **Variance at $x$** is exactly the variance from 2b.

- **Noise at $x$** is $\zeta^2$.

Putting them together:

$$
E\left[(y-\hat y(x))^2\right]=(\lambda,x^T(X^T X+\lambda I)^{-1}\theta^\star)^2+\sigma^2,x^T(X^T X+\lambda I)^{-1}X^T X(X^T X+\lambda I)^{-1}x+\zeta^2
$$

