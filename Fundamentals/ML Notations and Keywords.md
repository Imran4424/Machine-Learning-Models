# Hat Matrix

The hat matrix is a projection matrix used in linear regression that transforms a vector of observed responses into a vector of predicted responses.

The hat matrix defined as below
$$
\boldsymbol{H = X (X^T X)^{-1} X^T}
$$

where X is the design matrix

### Design Matrix - X

The design matrix or model matrix, X is a matrix of values of the independent (explanatory or predictor) variables for all the observations in the dataset.

#### Structure of the Design Matrix (\(\mathbf{X}\))

##### Rows

Each row in the matrix corresponds to a single observation or data point (e.g., a single person in a study). The number of rows, denoted by \(n\), is the total number of observations.

##### Columns

Each column corresponds to a specific variable or predictor in the model. The number of columns, denoted by \(p\), is the total number of parameters (including the intercept).

For a typical multiple linear regression model that includes an intercept term, the design matrix is structured as follows: 

$$
\mathbf{X} = \left(
\begin{matrix}
1 & x_{11} & x_{12} & \dots & x_{1,p-1} \\ 
1 & x_{21} & x_{22} & \dots & x_{2,p-1} \\ 
\vdots & \vdots & \vdots & \ddots & \vdots \\ 
1 & x_{n1} & x_{n2} & \dots & x_{n,p-1}
\end{matrix}
\right)
$$

##### First Column (Intercept)

The first column is typically a column of ones. This allows the model to estimate a y-intercept ($\beta_0$)

##### Subsequent Columns

The remaining columns contain the actual observed values for each independent variable (e.g., \(x_{1},x_{2},\dots ,x_{p-1}\))

### Role in the Hat Matrix

The hat matrix uses the design matrix to project the vector of observed dependent variable values ($\mathbf{y}$) onto the column space of $\mathbf{X}$ to obtain the predicted values ($\^{\mathbf{y}}$). It encapsulates all the information about the design of the experiment and the relationships between the predictor variables.

### Function and properties

##### Projection

It projects the observed response vector ($y$) onto the column space of the design matrix ($X$), producing the vector of fitted values,

$$
\boldsymbol{\^y = Hy}
$$

where,
$\boldsymbol{\^y}$ = Predicted values
$\boldsymbol{y}$ = Observed values
$\boldsymbol{H}$ = Hat matrix

##### Transforms data

It turns observed values into predicted values through the **linear least squares method**.

##### Symmetric and idempotent

The hat matrix is symmetric and idempotent, meaning,

$$
\boldsymbol{H ^ T = H} \\
\boldsymbol{H ^ 2 = H}
$$

##### Leverage values

The diagonal elements of the hat matrix ($h_{ii}$) are called leverages. They indicate the influence of the $i$-th observation on its own predicted value.

##### High leverage

Data points with high leverage can disproportionately influence the model's fit and are important to investigate for potential outliers.

### How  it is used

##### Model diagnostics

The hat matrix is a key tool for diagnosing the quality and stability of a regression model.

##### Outlier detection

By analyzing leverage values, analysts can identify individual data points that have a strong influence on the model's results.

##### Simplifies calculations

 It provides a way to calculate predicted values directly from the observed data, without needing to re-compute the regression coefficients each time.