# Machine-Learning-Models

#### According to wikipedia

**Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms
that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.**

Machine Learning is a subset of artificial intelligence in which computers learn from data and improve their performance on a task without
being explicitly programmed.

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

Artificial Intelligence is the theory and development of computer systems that are able to perform tasks that normally require human intelligence,
such as visual perception, speech recognition, decision-making, and translation between languages.

#### Machine Learning

Machine Learning is a subset of artificial intelligence.

As Arthur Samuels Describes **Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed.**

As I said, machine learning is a subset of artificial intelligence. So let's say it old school way, Every machine learning project or theories are artificial intelligence
but every artificial intelligence theory or project is not machine learning.

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

A training dataset is also called a labelled dataset which is used to train a model. During training, the model learns the features
extracted from the training dataset. From the label, the model will learn about the expected prediction based on the given features
of the training dataset.

#### Testing Dataset

A Testing dataset is also known as unlabelled data which contains all the features of the training dataset but not the expected
prediction label. Testing datasets are used to determine the performance accuracy of a trained model.

Usually, we need to divide the whole dataset as a Training and testing dataset. The divination ratios have three common choices

- 80% (Training), 20% (Testing) - Most popular choice (As far as I know)
- 90% (Training), 10% (Testing) - When the dataset size is small
- 70% (Training), 30% (Testing) - When the dataset size is huge

Supervised Machine Learning Models are broadly categorized into two subcategories.

- Regresstion
- Classification

#### Regresstion

Regresstion is a supervised machine learning technique where the goal is to predict a continuous numerical value based on one or
more independent features. In order to make predictions, it determines the relationships between variables. In regression, there
are two different kinds of variables:

- Independent variables: The features of the subject acting as input variables that influence the prediction
- Dependent Variable: Output variable of the subject which we are trying to predict based on the features of the subject

- Linear Regresstion
- Mutiple Linear Regresstion
- Polynomial Regresstion
- Decision Tree
- Random Forest
- Neural Network

#### Classification

- Logistic Regresstion
- Support Vector Machine
- Naive Bayes
- Decision Tree, Random Forest, Neural Network

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
