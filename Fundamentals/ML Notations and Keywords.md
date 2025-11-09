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
