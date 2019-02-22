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
