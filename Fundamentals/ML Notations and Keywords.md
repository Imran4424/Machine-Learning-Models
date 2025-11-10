# Hat Matrix

The hat matrix is a projection matrix used in linear regression that transforms a vector of observed responses into a vector of predicted responses.

The hat matrix defined as below

$$
\boldsymbol{H = X (X^T X)^{-1} X^T}
$$

where X is the design matrix

### Design Matrix - X

The design matrix or model matrix, X is a matrix of values of the independent (explanatory or predictor) variables for all the observations in the dataset.

#### Structure of the Design Matrix ($\boldsymbol{X}$)

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

The remaining columns contain the actual observed values for each independent variable (e.g., $x_{1},x_{2},\dots ,x_{p-1}$)

### Role in the Hat Matrix

The hat matrix uses the design matrix to project the vector of observed dependent variable values ($\mathbf{y}$) onto the column space of $\mathbf{X}$ to obtain the predicted values ($\boldsymbol{\hat{y}}$). It encapsulates all the information about the design of the experiment and the relationships between the predictor variables.

### Function and properties

##### Projection

It projects the observed response vector ($y$) onto the column space of the design matrix ($X$), producing the vector of fitted values,

$$
\boldsymbol{\hat{y} = Hy}
$$

where,
- $\boldsymbol{\hat{y}}$ = Predicted values
- $\boldsymbol{y}$ = Observed values
- $\boldsymbol{H}$ = Hat matrix

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


# Ordinary Least of Squares (OLS)

Ordinary Least of Squares (OLS) is a statistical method which is a type of linear least squares method for choosing the unknown parameters in a linear regression model.

Ordinary Least Squares (OLS) is used in linear regression to find the best-fitting straight line through a set of data points by minimizing the sum of the squared vertical distances between the data points and the line. It is a fundamental technique for estimating the parameters of a linear regression model, allowing for predictions of a dependent variable based on one or more independent variables.

Ordinary Least of Squares (OLS) starts with a linear model, which can represented by the following equation for a single independent variable,

$$
\boldsymbol{y = \beta_0 + \beta_1 x}
$$

Where,
- y = Dependent variable (Target variable which we are trying to predict)
- x = Independent variable (features / input variable)
- $\beta_0$ = intercept of the line (which adds additional degree of freedom)
- $\beta_1$ = Linear Regression coefficient

### Calculate residuals

The core principle of Ordinary Least of Squares (OLS) is to minimize the residual sum of squares (RSS), also known as the sum of squared errors (SSE). The residual is the difference between the actual observed value and the value predicted by the regression line. Squaring these differences prevents them from canceling out and gives more weight to larger errors

To calculate the Residual Sum of Squares (RSS), also known as the Sum of Squared Errors (SSE)

$$
\boldsymbol{ RSS = SSE = \sum_{i = 1}^{n} (y_i - \hat{y_i})^2}
$$

Where,
- $n$ is the number of observations in the dataset.
- $y_i$ is the actual (observed) value of the dependent variable for the $i$-th observation.
- $\hat{y_i}$ is the predicted value of the dependent variable for the $i$-th observation from the regression model.

# Rank of a matrix
# Rank of a Matrix

The rank of a matrix is the maximum number of linearly independent rows or columns in the matrix.

It is also equal to the number of non-zero rows in the matrix after it has been reduced to echelon form.

### Row Echelon form of Matrix

Row Echelon form is a way of transforming a matrix into a simpler, staircase-like structure through a series of elementary row operations, a process called Gaussian elimination.


##### Properties of a matrix in row echelon form

- Zero rows at the bottom: Any row that consists entirely of zeros is located at the bottom of the matrix, below all rows with non-zero entries.
- If a row does not consist entirely of zeros, then the first nonzero number in the row is a 1. In other deifinitions the first non-zero being 1 is not must.
- Staircase pattern: In any two successive rows that do not consist entirely of zeros, the leading 1 in the lower row occurs further to the right than the leading 1 in the higher row.
- Row echelon form is not unique
- Reduced Row echelon form is unique


#### Reduced Row Echelon form of Matrix

Reduced row echelon form (RREF) is a specific form of a matrix where each leading entry is a \(1\), and all other entries in a pivot column (the column with a leading \(1\)) are zeros. 

To achieve RREF, a matrix must first be in row echelon form, and then additional steps are taken to make pivots into \(1\)s and clear the entries above them, in addition to the zeros below them. 

This form is a unique, simplified representation of a matrix, often used to solve systems of linear equations.

- Zero rows at the bottom: Any row that consists entirely of zeros is located at the bottom of the matrix, below all rows with non-zero entries.
- If a row does not consist entirely of zeros, then the first nonzero number in the row must be 1.
- Staircase pattern: In any two successive rows that do not consist entirely of zeros, the leading 1 in the lower row occurs further to the right than the leading 1 in the higher row.
- Each column containing a leading 1 has zeros in all its other entries.
- Reduced Row echelon form is unique.

### Key Concepts

- Linear Independence: The rank represents the number of vectors (rows or columns) that are not scalar multiples of each other. If the rank is \(r\), there are \(r\) linearly independent rows and \(r\) linearly independent columns.
- Dimension: The rank is the dimension of the vector space spanned by the rows or the columns of the matrix.
- Minor: The rank is the order of the largest square sub-matrix (minor) that has a non-zero determinant. 

## How to find the rank

### Step 1: Reduce to Echelon Form (Row Echelon Form)

Use elementary row operations to transform the matrix into its echelon form. This involves using row replacement to create zeros below the "leading" non-zero entry (called a pivot) in each row.

### Step 2: Count Non-Zero Rows

Once the matrix is in echelon form, count the number of rows that contain at least one non-zero element. This number is the rank of the matrix.
