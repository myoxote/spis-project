import numpy as np
import keras.datasets as keras_data

# load the dataset from keras.dataset and directly split the tuples into seperated variables

(fashion_train_data,fashion_train_labels),(fashion_test_data,fashion_test_labels)=keras_data.fashion_mnist.load_data()
print(fashion_train_data.shape)
import matplotlib.pyplot as plt

plt.imshow(fashion_train_data[6],cmap=plt.cm.binary)
plt.show()

x_train=fashion_train_data.reshape((60000,28*28))
x_test=fashion_test_data.reshape((10000,28*28))

y_train=fashion_train_labels.copy()
y_train[y_train==0]=0
y_train[y_train==1]=0
y_train[y_train==2]=0
y_train[y_train==3]=0
y_train[y_train==4]=0
y_train[y_train==5]=1
y_train[y_train==6]=0
y_train[y_train==7]=1
y_train[y_train==8]=0
y_train[y_train==9]=1

y_test=fashion_test_labels.copy()
y_test[y_test==0]=0
y_test[y_test==1]=0
y_test[y_test==2]=0
y_test[y_test==3]=0
y_test[y_test==4]=0
y_test[y_test==5]=1
y_test[y_test==6]=0
y_test[y_test==7]=1
y_test[y_test==8]=0
y_test[y_test==9]=1


print(x_train.shape)
print(x_test.shape)

%%time

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB().fit(x_train, y_train)

print("train shape: " + str(x_train.shape))
print("score on test: " + str(mnb.score(x_test, y_test)))
print("score on train: "+ str(mnb.score(x_train, y_train)))
