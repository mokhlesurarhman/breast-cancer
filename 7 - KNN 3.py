# Avoiding warning
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


cancer = load_breast_cancer();
#print(cancer.DESCR)

print('Features Name-->');
print(cancer.feature_names)

print('-------------------------------------------');
print('Target Name-->');
print(cancer.target_names)

print('-------------------------------------------');
print('Cancer Data');
print(cancer.data)

print('-------------------------------------------');
print('Dataset Type:');
print(type(cancer.data));

print('-------------------------------------------');
print('Dataset Shape:');
print(cancer.data.shape);

print('-------------------------------------------');
import pandas as pd
raw_data = pd.read_csv('breast-cancer-wisconsin-data.csv', delimiter=',');
print('Print last 10 row:');
print(raw_data.tail(10));

#KNN Classifier Overview
#pip install mglearn
    # import mglearn
    # mglearn.plots.plot_knn_classification(n_neighbors=3)
#-------Not sure----#

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
#-----------------stratify-----------------
# This stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the
# proportion of values provided to parameter stratify.
# For example, if variable y is a binary categorical variable with values 0 and 1 and
# there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.
#-----------------random_state-----------------
#This is to check and validate the data when running the code multiple times.
# Setting random_state a fixed value will guarantee that same sequence of random numbers are generated each time you run the code.
# And unless there is some other randomness present in the process,
# the results produced will be same as always. This helps in verifying the output.


knn = KNeighborsClassifier();
knn.fit(X_train, y_train);

# How KNeighborsClassifier() works and their parameter
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')

print('Accuracy of KNN n-5, on the training set: {:.3f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNN n-5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = [];
test_accuracy = [];

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors);
    clf.fit(X_train, y_train);
    training_accuracy.append(clf.score(X_train, y_train));
    test_accuracy.append(clf.score(X_test, y_test));

plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_settings, test_accuracy, label='accuracy of the test state');
plt.legend();
plt.ylabel('Accuracy');
plt.xlabel('Number of neighbors');

fig1 = plt.gcf(); #get current figure
plt.show();
plt.draw();
fig1.savefig('7 - KNN 3.png', dpi=100);




