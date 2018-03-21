
# coding: utf-8

# In[2]:

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
import os
import cv2
import numpy as np
from keras.applications import vgg16


# In[9]:

shape=(50,67)
X=[]
for f in os.listdir('im_data2'):
    if('.jpg' in f):
        im=cv2.imread('im_data2/'+f,0)
        im=cv2.resize(im,shape)
        im=np.reshape(im,(shape[0],shape[1],1))
        im=im/255.
        X.append(im)
X=np.array(X)
print(X.shape)


# In[10]:

Y=[]
labels=pd.read_csv('target.csv')
Y=labels[['goal_home','goal_away']]


# In[11]:

batch_size = 128
num_classes = 3
epochs = 50
img_rows, img_cols = shape
input_shape = (img_rows, img_cols,1)


# In[17]:

def find_target(home,away):
    if(home>away):
        return 0
    if(home<away):
        return 1
    return 2

Y['target']=Y.apply(lambda x: find_target(x['goal_home'],x['goal_away']), axis=1)
Yclf=keras.utils.to_categorical(Y[['target']], num_classes)
p = np.random.permutation(X.shape[0])
X=X[p]
Yclf=Yclf[p]
Xtrain,Xtest=X[:2300],X[2300:]
# Ytrain,Ytest=Y[['target']][:2300].as_matrix(),Y[['target']][2300:].as_matrix()
Ytrain,Ytest=Yclf[:2300],Yclf[2300:]


# In[18]:

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(Xtrain, Ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(Xtest, Ytest))


# In[19]:

model=vgg16.VGG16(include_top=True,weights=None,input_shape=input_shape,pooling='max',classes=3)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(Xtrain, Ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(Xtest, Ytest))


# In[ ]:



