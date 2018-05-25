# Machine Learning Foundations
Generally in machine learning we have a problem to solve and this problem usually relates to evaluating some data and making predictions. In order to solve it, we use tools such as algorithms that include linear/ logistic regression, decision trees, neural networks, etc. We then use some measurement tools to evaluate these algorithms and their parameters to decide which ones are best for our problem.

> 1. Train a bunch of models using training data
> 2. Use the Cross Validation data to pick the best of these models
> 3. Test it with the testing data to make sure model is good

## Algorithms
- Decision Trees
- Naive Bayes 
- Gradient Descent
- Linear Regression
- Logistic Regression
- Support Vector Machines
- Neural Networks
- Kernel Method
- K-means Clustering
- Hiearchical Clustering
- etc

## Training Models
- How well is my model doing?
- How do we improve it based on certain metrics?

Define the classifier, then fit the classifier to the data (usually `X`, `y`).
```
classifier.fit(X, y)
```

### Examples
#### Logistic Regression 
```
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
```

#### Neural Networks
```
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
```

#### Decision Trees
```
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
```

#### Support Vector Machines
```
from sklearn.svm import SVC
classifier = SVC()
```

### Training a Model
```
import pandas as pd
import numpy as np

# Read the data
data = pd.read_csv('data.csv')

# Split the data into X and y
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# import statements for the classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# TODO: Pick an algorithm from the list:
# - Logistic Regression
# - Decision Trees
# - Support Vector Machines
# Define a classifier (bonus: Specify some parameters!)
# and use it to fit the data
# Click on `Test Run` to see how your algorithm fit the data!
logr_classifier = LogisticRegression()
gb_classifier = GradientBoostingClassifier()
svm_classifier = SVC()

<classifiername>_classifier.fit(X,y)
```

## Testing Models
> Thou shalt never use your testing data for training

### Types of Errors
#### Underfitting
- Oversimplification of problem where model is unable to explain the data
- Characterized by poor training and test scores (high training and testing errors)
- High bias error (low variance error)

#### Overfitting
- Overcomplication of problem where model "memorizes" training data
- Characterized by great training score but poor test score (low training error, high testing error)
- High variance error (low bias error)

### Cross Validation
- Training set used for training the parameters
- Cross Validation set used for making decisions about the model
- Testing set used for final testing of the model

#### K-Fold Cross Validation
- Separate data into `k` buckets
- Train model `k` times, each time using a different bucket as testing set and the remaining points as training set
- Average results to get final model

```
# Parameters are the size of the data and the size of the testing set
# Randomize data using the 'shuffle' parameter

from sklearn.model_selection import KFold
kf = KFold(12, 3, shuffle=True)

for train_indices, test_indices in kf:
  print (train_indices, test_indices)
```

#### Learning Curves
```
train_sizes, train_scores, test_scores = learning_curve(
  estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))
```

- `estimator`, is the actual classifier we're using for the data, e.g., `LogisticRegression()` or `GradientBoostingClassifier()`.
- `X` and `y` is our data, split into features and labels.
- `train_sizes` are the sizes of the chunks of data used to draw each point in the curve.
- `train_scores` are the training scores for the algorithm trained on each chunk of data.
- `test_scores` are the testing scores for the algorithm trained on each chunk of data.

#### Important Observations
- The training and testing scores come in as a list of 3 values, and this is because the function uses 3-Fold Cross-Validation.
- **Very important:** As you can see, we defined our curves with Training and Testing Error, and this function defines them with Training and Testing Score. These are opposite, so the higher the error, the lower the score. Thus, when you see the curve, you need to flip it upside down in your mind, in order to compare it with the curves above.

### Grid Search
#### Parameters vs Hyper-Parameters
- Parameters of a algorithm may be the coefficients of the polynomial in a Logistic Regression model or the threshold of the leaves and the nodes in a Decision Tree
- Hyper-Parameters may be the degree of the polynomial or the maximum depth of the tree 

What happens if you need to tune multiple Hyper-Parameters? We can use `Grid Search` 