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

#### Rows
Each row in the matrix corresponds to a single observation or data point (e.g., a single person in a study). The number of rows, denoted by \(n\), is the total number of observations.