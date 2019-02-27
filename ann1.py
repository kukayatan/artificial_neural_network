# the goal of this code is to build up a ANN model that would be able to 
# predict the with high accuracy if the customers will or will not leave the bank.
# thee dataset being used in this model consists of a few fundamental customer
# information like age, account balance, origine, gender and so on..

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

# first, load dataset and split it into two variables (features vs predicted value)
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# since the file contains some categorical varaibles (origin and gender)
# dummy varieable techniques are being applied
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# the last step of the preprocessing is to rescale the features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# the ANN is being built .. 11 features + 1 dependent variable, for activations Rectified linear unit 
# function is being used, for output layer, the logistic function is being used
# the ANN is built with two hidden layers..
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 5, epochs = 70)


# the calculated model is being used to predict data on X_test
# the treshold is being set up
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# the confusion metrix and accuracy of the results are calculated and printed..
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print('\n')
print('The prediction accuracy of the ANN model is ' + str(((1545 + 147)/2000)*100) + ' %')
