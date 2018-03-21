
# coding: utf-8

# In[1]:

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
import numpy as np


# In[23]:

data=pd.read_csv('Final_data.csv')
data.columns.values
data
def find_results(h,a):
    if(h>a):
        return 0
    if(h<a):
        return 1
    return 2
#data['result']=data.apply(lambda row: find_results(row['home_team_goal'],row['away_team_goal']),axis=1)

X=data[data.columns.difference(['winner'])]
Y=data[['winner']]
X.columns.values
X


# In[24]:

X=X.fillna(X.mean())
X=X.as_matrix()
X = (X - X.mean(axis=0)) / X.std(axis=0)
X


# In[25]:

from keras.utils import to_categorical
Y=to_categorical(Y)

p = np.random.permutation(X.shape[0])

X=X[p]
Y=Y[p]

Xtrain,Xtest=X[:7000],X[7000:]
Ytrain,Ytest=Y[:7000],Y[7000:]
Ytrain.shape


# In[26]:

def FNN():
    model = Sequential()
    model.add(Dense(200, input_dim=23))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(100))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
    return model


# In[27]:

Xtrain


# In[46]:

model=FNN()
history=model.fit(x=Xtrain, 
          y=Ytrain, 
          batch_size=128, 
          epochs=30,
          verbose=1,
          validation_data=[Xtest,Ytest], 
         )


# In[42]:


ypred=np.argmax(model.predict(np.array(Xtest)),axis=1)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(np.argmax(Ytest,axis=1),ypred)
ypred.shape
df=pd.DataFrame(columns=['Predicted 0','Predicted 1','Predicted 2'], index=['Actual0','Actual1','Actual2'])
for i in range(3):
    row='Actual'+str(i)
    for j in range (3): 
        col='Predicted '+str(j)
        df.loc[row,col]=confusion[i][j]
print(df)
df.to_csv("Confusion.csv")


# In[52]:

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[51]:

from matplotlib import pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16, 6)

