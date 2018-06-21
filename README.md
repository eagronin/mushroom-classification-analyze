# Analysis

This section describes an analysis of classifying mushrooms into edible and not edible by appllying several classifiers to a feature space reduced to two principal components. The classifiers used in the analysis include Logistic Regression, KNN, Decision Tree, Random Forest, SVC, Naive Bayes and Neural Network.

Data cleaning and processing is described in the [previous section](https://eagronin.github.io/mushroom-classification-prepare/).

Accuracy of the classifiers and visualizations of their respective decision boundaries and probabilities are reported in the [next section](https://eagronin.github.io/mushroom-classification-report/).

The following code plots the target against the first two principal componets of the feature space:

```python
# Plot the target as a function of the two principal components
plt.cla()
plt.clf()

plt.figure(dpi=120)
plt.scatter(X_train[y_train.values==0, 0], X_train[y_train.values==0, 1], label = 'Edible', alpha = 0.5, s = 2)
plt.scatter(X_train[y_train.values==1, 0], X_train[y_train.values==1, 1], label = 'Poisonous', alpha = 0.5, s = 2)
plt.title('Mushroom Data Set\nFirst Two Principal Components')
plt.legend(frameon=True)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.gca().set_aspect('equal')

pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/pca.png')
```

The plot of the target against principal componets is presented in the [next section](https://eagronin.github.io/mushroom-classification-report/).

Next, after each classifier is fitted to the training data, the function below is called to visualize the decision boundary and decision probabilities for that classifier:

```python
def decision_boundary(X, y, fitted_model):
    
    fig = plt.figure(figsize = (10,5), dpi=100)
    
    for i, plot_type in enumerate(['Decision Boundary', 'Decision Probabilities']):
        
        plt.subplot(1,2,i+1)
        step = 0.01
        x_max = X[:,0].max() + .1
        x_min = X[:,0].min() - .1
        y_max = X[:,1].max() + .1
        y_min = X[:,1].min() - .1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

        if i==0:
            # Flatten the meshgrid into one column to perform prediction
            Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            try:
                # Flatten the meshgrid into one column to perform prediction
                Z = fitted_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
            except:
                plt.text(0.4, 0.5, 'Probabilities Unavailable', 
                         horizontalalignment='center', verticalalignment='center', 
                         transform = plt.gca().transAxes, fontsize=12)
                plt.axis('off')
                break        
        
        # Reshape the column with predicted values into a two-dimensional grid
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary and probabilities
        plt.scatter(X[y.values==0,0], X[y.values==0,1], label = 'Edible', alpha = 0.4, s = 5)
        plt.scatter(X[y.values==1,0], X[y.values==1,1], label = 'Poisonous', alpha = 0.4, s = 5)
        plt.imshow(Z, interpolation = 'nearest', extent = (x_min, x_max, y_min, y_max), alpha = .15, origin = 'lower')
        plt.title(plot_type + '\n' + str(fitted_model).split('(')[0] +
                                         'Test Accuracy: ' + str(round(fitted_model.score(X, y),5)))
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    return fig
```

Each classifier is fitted to the training data as shown in the code below.  The performance of each classifier is then
evaluated using the test data.  The plots of decision boundary and decision probabilities for each classifier along 
with the accuracy score are presented in the [next section](https://eagronin.github.io/mushroom-classification-report/).

```python
# Fit logistic regression and plot decision boundary and probabilities
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)
fig = decision_boundary(X_test, y_test, model)
pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/logit.png')

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 20).fit(X_train, y_train)
fig = decision_boundary(X_test, y_test, model)
pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/knn.png')

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier().fit(X_train, y_train)
fig = decision_boundary(X_test, y_test, model)
pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/tree.png')

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X_train, y_train)
fig = decision_boundary(X_test, y_test, model)
pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/forest.png')

from sklearn.svm import SVC
model = SVC(kernel = 'rbf', C = 10).fit(X_train, y_train)
fig = decision_boundary(X_test, y_test, model)
pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/svc.png')

from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X_train, y_train)
fig = decision_boundary(X_test, y_test, model)
pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/naive.png')

from sklearn.neural_network import MLPClassifier
model = MLPClassifier().fit(X_train, y_train)
fig = decision_boundary(X_test, y_test, model)
pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/mlp.png')
```

Next step: [Results](https://eagronin.github.io/mushroom-classification-report/)
