#****Hand digit Recognition using SVM(Support Vector Machine)****#

#Importing basic libraries
import numpy as np
from sklearn.datasets import load_digits

#load datasets
dataset= load_digits()

#summarize dataset
print(dataset.data)
print(dataset.target)

print(dataset.data.shape)
print(dataset.images.shape)

dataimageLength= len(dataset.images)
print(dataimageLength)


#visualize the dataset

n=9 #no of sample out of samples total 1797

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.images[n])
plt.show()

dataset.images[n]


#seggregate the data X(dependent value) and Y(independent value)

X= dataset.images.reshape((dataimageLength,-1))
X

Y= dataset.target
Y

#Splitting Dataset into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size= 0.25, random_state=0)
print(X_train.shape)
print(X_test.shape)

#training 

from sklearn import svm
model= svm.SVC(kernel='linear')
model.fit(X_train,Y_train)

#predicting, what the digit is from Test data

n=1795
result=model.predict(dataset.images[n].reshape((1,-1)))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r,interpolation='nearest')
print(result)
print("\n")
plt.axis('off')
plt.title('%i' %result)
plt.show()

#Prediction for Test Data
Y_pred = model.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

#Evaluate model- Accuracy Score

from sklearn.metrics import accuracy_score
print("Accuracy of the model: {0}%".format(accuracy_score(Y_test,Y_pred)*100))


#Play with the Different Method

from sklearn import svm
model1 = svm.SVC(kernel='linear')
model2 = svm.SVC(kernel='rbf')
model3 = svm.SVC(gamma=0.001)
model4 = svm.SVC(gamma=0.001,C=0.1)

model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)
model3.fit(X_train,Y_train)
model4.fit(X_train,Y_train)

y_predModel1 = model1.predict(X_test)
y_predModel2 = model2.predict(X_test)
y_predModel3 = model3.predict(X_test)
y_predModel4 = model4.predict(X_test)

print("Accuracy of the Model 1: {0}%".format(accuracy_score(Y_test, y_predModel1)*100))
print("Accuracy of the Model 2: {0}%".format(accuracy_score(Y_test, y_predModel2)*100))
print("Accuracy of the Model 3: {0}%".format(accuracy_score(Y_test, y_predModel3)*100))
print("Accuracy of the Model 4: {0}%".format(accuracy_score(Y_test, y_predModel4)*100))