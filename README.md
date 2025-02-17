# Machine-Learning-Models

#### According to wikipedia

**Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.**

Machine Learning is a subset of artificial intelligence in which computers learn from data and improve their performance on a task without being explicitly programmed.

Data is really important for machine learning because machine learning relies on data to train models and learn patterns.

#### Formal Definition

Arthur Samuels Describes **Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed.**

#### Machine Learning Examples

- Classifying emails as spam or not spam
- Google Search Engine
- Facebook photo recognization
- Self Driving Cars

#### Types of Machine Learning

There are basically three main types of Machine Learning

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

Moreover, there is a more specific category called **semi-supervised** learning, which combines elements of both **supervised** and **unsupervised** learning.

## Artificial Intelligence vs Machine Learning

Artificial Intelligence and Machine Learning are technical buzzwords now.

But, people are often confused about two. They think they are all the same but they are not. There are some differences between artificial intelligence and machine learning.

#### Artificial Intelligence

Artificial Intelligence is the theory and development of computer systems that are able to perform tasks that normally require human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages.

#### Machine Learning

Machine Learning is a subset of artificial intelligence.

As Arthur Samuels Describes **Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed.**

As I said, machine learning is a subset of artificial intelligence. So let's say it old school way, Every machine learning project or theories are artificial intelligence but every artificial intelligence theory or project is not machine learning.

Then let's see now, what's not machine learning to better understand the difference.

- Machine learning is not based on logic or rule it is based on statistics
- Machine learning has not understanding of the world
- Machine learning is not study of neuro science

## Essential mathematics for ML

The important mathematics for machine learning. This list is sorted by the importance rate. The first item on the list is the most important

- Linear Algebra
- Multivariative Calculus
- Probability Theory
- Discrete Mathematics
- Statistics

#### Linear Algebra

Free Resoures where we can learn

- [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Gilbert Strang Lectures](https://www.youtube.com/playlist?list=PL49CF3715CB9EF31D)
- [Khan Academy Videos](https://www.youtube.com/playlist?list=PLFD0EB975BA0CC1E0)
- [Khan Academy Website](https://www.khanacademy.org/math/linear-algebra)
- [Coding the Matrix by Philip Klein](https://cs.brown.edu/video/channels/coding-matrix-fall-2014/?page=2)
- [TrevTutor](https://www.youtube.com/playlist?list=PLDDGPdw7e6AjJacaEe9awozSaOou-NIx_)

Paid Linear Algebra resources

- [Become a Linear Algebra Master](https://www.udemy.com/course/linear-algebra-course/?ranMID=39197&ranEAID=JVFxdTr9V80&ranSiteID=JVFxdTr9V80-ifogBxRxVqsHhdZOvMPLIQ&LSNPUBID=JVFxdTr9V80&utm_source=aff-campaign&utm_medium=udemyads&couponCode=LETSLEARNNOW)
- [Complete linear algebra: theory and implementation in code](https://www.udemy.com/course/linear-algebra-theory-and-implementation/?ranMID=39197&ranEAID=JVFxdTr9V80&ranSiteID=JVFxdTr9V80-SZqFw5YMCKDMjE0N_HP7VQ&LSNPUBID=JVFxdTr9V80&utm_source=aff-campaign&utm_medium=udemyads&couponCode=LETSLEARNNOW)

#### Multivariative Calculus

Free Resources where we can learn

- [Khan Academy Videos](https://www.youtube.com/playlist?list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7)
- [Khan Academy Website](https://www.khanacademy.org/math/multivariable-calculus)
- [MIT OpenCourseWare](https://www.youtube.com/playlist?list=PL4C4C8A7D06566F38)
- [The Bright Side of Mathematics](https://www.youtube.com/playlist?list=PLBh2i93oe2qv4G2AyarkbR3OKBml0hXEg)

#### Probability Theory

Free Resources where we can learn

- [The Bright Side of Mathematics](https://www.youtube.com/playlist?list=PLBh2i93oe2qswFOC98oSFc37-0f4S3D4z)
- [Khan Academy Videos](https://www.youtube.com/playlist?list=PLC58778F28211FA19)
- [MIT OpenCourseWare](https://www.youtube.com/playlist?list=PLUl4u3cNGP60hI9ATjSFgLZpbNJ7myAg6)

#### Discrete Mathematics

Free Resources where we can learn

- [Kimberly Brehm](https://www.youtube.com/playlist?list=PLl-gb0E4MII28GykmtuBXNUNoej-vY5Rz)
- [TrevTutor](https://www.youtube.com/playlist?list=PLDDGPdw7e6Ag1EIznZ-m-qXu4XX3A0cIz)
- [CodeAcademy](https://www.codecademy.com/learn/discrete-math)

#### Statistics

Free Resources where we can learn

- [Khan Academy Videos](https://www.youtube.com/playlist?list=PL1328115D3D8A2566)
- [Khan Academy](https://www.khanacademy.org/math/statistics-probability)
- [The Organic Chemistry Tutor](https://www.youtube.com/playlist?list=PL0o_zxa4K1BVsziIRdfv4Hl4UIqDZhXWV)
- [CrashCourse](https://www.youtube.com/playlist?list=PL8dPuuaLjXtNM_Y-bUAhblSAdWRnmBUcr)
- [Kimberly Brehm](https://www.youtube.com/playlist?list=PLl-gb0E4MII1dkfGxmdt8YA0Dgabdvdmq)

# Supervised Learning

In supervised learning, when we have a Dataset,

- There will be any number of independent variables which are also called input variables or features
- A dependent variable which is also target or output variable

In supervised learning datasets can be divided into two main parts

1. Training Dataset
2. Testing Dataset

#### Training Dataset

A training dataset is also called a labelled dataset which is used to train a model. During training, the model learns the features extracted from the training dataset. From the label, the model will learn about the expected prediction based on the given features of the training dataset.

#### Testing Dataset

A Testing dataset is also known as unlabelled data which contains all the features of the training dataset but not the expected prediction label. Testing datasets are used to determine the performance accuracy of a trained model.

Usually, we need to divide the whole dataset as a Training and testing dataset. The divination ratios have three common choices

- 80% (Training), 20% (Testing) - Most popular choice (As far as I know)
- 90% (Training), 10% (Testing) - When the dataset size is small
- 70% (Training), 30% (Testing) - When the dataset size is huge

Supervised Machine Learning Models are broadly categorized into two subcategories.

- Regresstion
- Classification

## Regresstion

Regresstion is a supervised machine learning technique where the goal is to predict a continuous numerical value based on one or more independent features. In order to make predictions, it determines the relationships between variables. In regression, there are two different kinds of variables:

- Independent variables: The features of the subject acting as input variables that influence the prediction
- Dependent Variable: Output variable of the subject which we are trying to predict based on the features of the subject

There are various types of regression model available based on the number of predictor variables and the nature of the relationship between variables:

- Linear Regresstion
- Polynomial Regresstion
- Support Vector Regression
- Ridge and Lasso Regression
- Decision Tree Regression
- Random Forest Regression
- Neural Network Regression

### Linear Regression

A Linear regression is a linear appoximation of a casual relationship between two or more variables.

Linear regression sometimes called as mother of all machine learning models. Linear regression model try to determine the linear relationship between the independent and dependent variables. This indicates that, the change in the independent variables will be reflected in the dependent variable proportionally.

**Mathematical representation of Linear Regression Model**

$$
\boldsymbol{y = \beta_0 + \beta_1 x + \epsilon}
$$

Where,
y = Dependent variable (Target variable which we are trying to predict)
x = Independent variable (features / input variable)
$\beta_0$ = intercept of the line (which adds additional degree of freedoom)
$\beta_1$ = Linear Regresstion coefficient
$\epsilon$ = Error of estimation (difference between observed income and the income regression predicted)

There are two types of linear regression model based on the number of input variables

- Simple Linear Regression
- Multiple Linear Regression

#### Simple Linear Regression

In simple linear regression model, one independent variable is used as an input for predicting the output, the dependent variable of the model

The above shown of linear regression is actually for simple linear regression

$$
\boldsymbol{y = \beta_0 + \beta_1 x + \epsilon}
$$

#### Multiple Linear Regression

In multiple linear regression model, more than one independent variable is used as input for predicting the output, the dependent variable of the model

**Mathematical representation of Multiple Linear Regression Model**

$$
\boldsymbol{y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon}
$$

We can also simplified the equation as following

$$
\boldsymbol{y = \beta_0 + \displaystyle\sum_{i=1}^{n} \beta_i x_i + \epsilon}
$$

Where,
y = Dependent variable (Target variable which we are trying to predict)
$x_i$ = Independent variables (features / input variable)
$\beta_0$ = intercept of the line (which adds additional degree of freedoom)
$\beta_i$ = Linear Regresstion coefficients of their associated independent variables
$\epsilon$ = Error of estimation (difference between observed income and the income regression predicted)

#### Linear Regression Examples

- Predicting house price or rent based on the house size, house age, distance from main road, where it is located
- Predicting salary based on the education level, job position, years of experience and so on

### Polynomial Regression

The Polynomial Regression Model is a special case of multiple linear regression in which the coefficients are all linear, but the independent variable x is an nth-degree polynomial.

In a Polynomial Regression Model, the relationship between the independent(input) variable and dependent(output) variable is not linear. Simple and Multiple Linear Regression models are best suited for straight lines. But when the relationships between independent and dependent variables are better represented by a curve, the Polynomial Regression Model is a better choice than previously discussed linear regression models.

Now, a question can arise: Can you call a Polynomial Regression Model a linear regression model since the input variables are n-degree polynomials?
Yes, We can call the Polynomial Regression Model a Linear Regression Model since the coefficients are all linear.

**Mathematical representation of Polynomial Regression Model**

$$
\boldsymbol{y = \beta_0 + \beta_1 x^1 + \beta_2 x^2 + \dots + \beta_n x^n + \epsilon}
$$

We can also simplified the equation as following

$$
\boldsymbol{y = \beta_0 + \displaystyle\sum_{i=1}^{n} \beta_i x^i + \epsilon}
$$

Where,
y = Dependent variable (Target variable which we are trying to predict)
$x$ = Independent variables (features / input variable)
$n$ = The degree of the polynomial
$\beta_0$ = intercept of the line (which adds additional degree of freedoom)
$\beta_i$ = Coefficients of their polynomial terms
$\epsilon$ = Error of estimation (difference between observed income and the income regression predicted)

**When to use Polynomial Regression Model?**
When the relationship between the dependent and independent variables is non-linear, we can not use simple and multiple linear regression to fit the data well. Then, we can use the Polynomial regression model. The best way to find out is by plotting x vs y. If the plotted graph shows parabolic, cubic, or other curved patterns, then the Polynomial Regression Model is a good choice.

**Among n-degree order which order should we use for our Polynomial Regression Model?**
We can use Bayes Information Criteria for figuring out the appropiate regression model order.

$$
\boldsymbol{BIC_k = n log(SS_\epsilon) + k log(n)}
$$

where,
BIC = Bayes Information Criteria
k = number of parameter or chosen order the model to test
n = number of data points
$SS_\epsilon$ = Error Sum of Squares

The formula of $SS_\epsilon$ is:

$$
\boldsymbol{SS_\epsilon = \displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2}
$$

where,
$x_i$ = individual data points in the dataset.
$\bar{x}$ = mean(average) of all data points

$\bar{x}$ is calculated as following:

$$
\boldsymbol{\bar{x} = \frac{1}{n} \displaystyle\sum_{i=1}^{n} x_i}
$$

We need to run through $BIC_k$ equation for some orders and create a plot **polynomial model order vs BIC**. Then find out the minimum plot from the plot.

#### Polynomial Regression Examples

- Stock market trend prediction
- Inflation and GDP growth prediction
- Energy consumtion prediction compared to outdoor environment

### Decision Tree

Decision Tree Regression is a non-linear regression model that can handle complex datasets with complicated patterns. It makes predictions using a tree-like model, making it flexible and easy to interpret.

Decision Tree Regression at the core is a binary tree which predicts continuous values. This model creates a root node on entire dataset then splits the data into smaller subsets based on decision rules derived from the input features. Each split is made to minimize the error in predicting the target variable.

Where the split gonna happen is determined by the Information Gain. To calculate Information Gain first we need to calculate Information Entropy. For the first split, we look at the whole dataset for Information Entropy and Information Gain. For the next split, we consider the relevant split portion we are working on: local Information entropy and Information gain for the further split (more like a recursive split operation).

While building the Decision Tree, the below points are to be considered

- Features to choose
- Conditions for splitting (Information Entropy and Information gain)
- To know where to stop
- Pruning

#### Information Entropy

The provided dataset's disorder or impurity is measured by entropy.

The values of the feature vectors linked to each data point are used for splitting the messy data in the decision tree. With each split, the data becomes more homogenous which will decrease the entropy. But in average case, some data in some nodes will not be homogenous and that will result into higher entropy values. With higher entropy it is really hard to predict any appropiate values. When the tree eventually reaches the terminal or leaf node, the highest level of purity is applied.

The entropy of a dataset calculated by Shannon’s entropy formula

$$
\boldsymbol{E(S) = -\displaystyle\sum_{i=1}^{c}P_i log_2 P_i}
$$

where,
E(S) is the entropy of the dataset.
c is the number of unique classes in the dataset
$P_i$ is the proportion of instances in class $i$

Here,

$$
\boldsymbol{P_i = \frac{\text{instances in class i}}{\text{total instances}}}
$$

Between two choice in a decision tree if all the choices belong to one particular choice then the entropy will be 0.

Between two choice in a decision tree if the choices are distributed among two choices equally then the entropy will be 1.

#### Information Gain

Information gain decides which feature to split at each step in building the tree. Creating sub-nodes increases homogeneity, decreasing the entropy of these nodes. The more homogeneous the child node, the more the variance will decrease after each split. Thus, Information Gain is the variance reduction and can be calculated by determining how much the variance decreases after each split.

Information gain of a parent node can be calculated as the entropy of the parent node subtracted from the entropy of the weighted average of the child node.

For a dataset with many features, each feature's information gain is calculated. The feature with maximum information gain will be the most important feature, the root node for the decision tree.

#### Gini Index

The Gini index can also be used for feature selection. The tree chooses the feature that minimizes the Gini impurity index. The higher value of the Gini Index indicates the impurity is higher. Both the Gini Index and Gini Impurity are used interchangeably. The Gini Index or Gini Impurity favors large partitions and is simple to implement. It performs only binary split. For categorical variables, it gives the results of “success” or “failure”.

The Gini Index can be calculated by following formula

$$
\boldsymbol{Gini = 1 -\displaystyle\sum_{i=1}^{c}(P_i)^2}
$$

The Gini Index or Gini Impurity favors large partitions and is simple to implement. It performs only binary split. For categorical variables, the results are given in terms of “success” or “failure.”

### Random Forest

Random Forest is nothing but collection of mutiple decision trees.

Before talking more about Random Forest. Let's talk about Ensemble Learning

#### Ensemble Learning

Ensemble Learning is a fantastic machine learning technique that combines multiple machine learning models to improve predictions.

Multiple machine-learning model combinations can happen in the following ways.

- Ensemble learning combining multiple models (individual different models)
- Ensemble learning by combining the same model multiple times

In random forest, we will use the later version of Ensemble Learning, which combines multiple decision tree models to make predictions, improving accuracy and decreasing overfitting compared to a single decision tree.

We can build Random Forest regression model using the following steps

1. **Bootstrap Sampling:** Create multiple subsets of training data by randomly selecting samples with replacement.
2. **Random Forest Construction:** Use the subsets of training data created in Step 1 to create decision trees. Each subset is used to train an independent Decision Tree, which uses a random subset of features to split nodes.
3. **Prediction Aggregation:** The Random Forest Regression model takes the average of all individual tree predictions (In terms of Ensemble Learning, this technique is called bagging).

Mathematical formula for the Random Forest Model,

$$
\boldsymbol{\hat{y} = \frac{1}{T}\displaystyle\sum_{i=1}^{T}y_i}
$$

where,
T is the total number of decision trees in the forest
$y_i$ is the prediction from the i-th Decision Tree Model
$\hat{y}$ is the final prediction of the random forest

### Neural Network

## Classification

Classification is a supervised machine learning technique where the goal is to indentify (predict) the category of new observations based on training data.

Similar to regression, Classification has two different kinds of variables:

- Independent variables: The features of the subject acting as input variables that influence the prediction
- Dependent Variable: Output variable of the subject which we are trying to predict based on the features of the subject

Spam detection in email services is the best real-world example of classification techniques.

When email services only detect spam and not spam classification, it is called the binary classification technique.

In the current world, we primarily use Gmail email services and Gmail provides multi-class classification classification:

- Primary: main email box
- Promotions: product-related promotions email box
- Social: Social Network related email box
- Updates: Updates from different online services
- Spam: The emails which are classified as spam

There are various types of classification models available based on the number of predictor variables (input features) and the nature of the relationship between variables:

- Logistic Regresstion
- K-Nearest Neighbors (K-NN)
- Support Vector Machine
- Kernel Support Vector Machine
- Naive Bayes
- Decision Tree Classification
- Random Forest Classification
- Neural Network

### Logistic Regresstion

fasdf
dasd

### Support Vector Machine

### Naive Bayes

# Unsupervised Learning

Unsupervised Machine Learning Models

#### Clustering

- K-Means
- Hierarchical
- Mean Shift
- Density-based

#### Dimensionality Reduction

- Feature Elimination
- Feature Extraction
- Principal Component Analysis (PCA)
