import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Importing dataset using pandas dataframe
dataset = pd.read_csv('dataset/car.csv', names=['buying', 'maint','doors','persons', 'lug_boot', 'safety', 'class'])

print(dataset.head())

print(dataset.info())

dataset['class'],class_names = pd.factorize(dataset['class'])
print(class_names)
print(dataset['class'].unique())

# Converting all categorical variables to numeric
dataset['buying'],_ = pd.factorize(dataset['buying'])
dataset['maint'],_ = pd.factorize(dataset['maint'])
dataset['doors'],_ = pd.factorize(dataset['doors'])
dataset['persons'],_ = pd.factorize(dataset['persons'])
dataset['lug_boot'],_ = pd.factorize(dataset['lug_boot'])
dataset['safety'],_ = pd.factorize(dataset['safety'])


print(dataset.info())


X=dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

# Split dataset in training and test datasets
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.4, random_state=2)

# Instantiate the classifier
clf = GaussianNB()

# Train classifier
clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)

mislabeled_points = (Y_test != prediction).sum()

print('Number of mislabeled points out of a total points: {}'.format(mislabeled_points))

performance  = accuracy_score(Y_test,prediction)


print("Classifier performance: {:.2f} %".format(performance))


