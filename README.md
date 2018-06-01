# Analysis

This section describes the classification analysis to evaluate performance of several classifiers, 
including Logistic Regression, KNN, Decision Tree, Random Forest, SVC, Naive Bayes and Neural Network, 
in predicting whether a mushroom with particular attributes is edible or not.

```
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

The plot of the target against the principal componets is presented in the [next section](link to analysis section).

Next, the function below visualizes the decision boundary and decision probabilities for a classifier,
and is called after each classifier was fitted to the training data:

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

Finally, the following code fits each classifier to the training data.  The performance of each classifier is then
evaluated using the test data.  The plots of decision boundary and decision probabilities for each classifier along 
with the accuracy scores are presented in the [next section](link).

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

Next step: [Reporting](link)
