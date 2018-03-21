
# coding: utf-8

# In[58]:

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# In[74]:

def plot_graph(no_of_trees, error_on_train,error_on_test):
    fig,ax=plt.subplots()
    ax.plot(no_of_trees,error_on_train,label="Plot of train data")
    ax.plot(no_of_trees,error_on_test,label="Plot of test data")
    ax.set_ylabel('Error')
    ax.set_title('Error vs number of weak classifiers')
    ax.set_xlabel('Number of weak classifiers')
    legend = ax.legend(loc='upper right', shadow=True)
    ind=[5,10,15,20,25,30,35,40]
    ax.set_xticks(ind)
    plt.axis([1,42,0,1])
    plt.ylabel('Error')
    fig.savefig("error_vs_number of weak classifiers.png")
    plt.show()


# In[2]:

def build_the_data(data):
    out=data[['winner']]
    Y=out.as_matrix()
    del data['winner']
    data=data.fillna(data.mean())
    X=data.as_matrix()
    arr=np.random.permutation(8528)
    X=np.take(X,arr,axis=0)
    Y=np.take(Y,arr,axis=0)
    points=7000
    train_X=X[0:points,:]
    train_Y=Y[0:points,:]
    test_X=X[points:,:]
    test_Y=Y[points:,:]
    #print(train_X.shape)
    #print(train_Y.shape)
   # print(test_X.shape)
    #print(test_Y.shape)
    return train_X,train_Y,test_X,test_Y
    


# In[50]:

def accuracy(predicted,output):
    l=len(predicted)
    return (1-np.sum(predicted == output)/l)


# In[77]:

def do_ensemble_methods(train_X,train_Y,test_X,test_Y):
    error_on_test=[]
    error_on_train=[]
    ix=[]
    for i in range(1,42,2):
        print(i)
        #dt = DecisionTreeClassifier() 
        #AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)[source]
        dt=DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=12, min_samples_split=2, min_samples_leaf=2,random_state=0)
        clf = AdaBoostClassifier(base_estimator=dt,n_estimators=i,learning_rate=1.0)
        clf=clf.fit(train_X,train_Y)
        out_test=clf.predict(test_X)
        out_train=clf.predict(train_X)
        #scores = cross_val_score(clf, train_X,train_Y, cv=5)
        train_ac=accuracy(out_train,train_Y)
        test_ac=accuracy(out_test,test_Y)
        error_on_train.append(train_ac)
        error_on_test.append(test_ac)
        ix.append(i)
        #print("score is ",score)
        
    return ix,error_on_train,error_on_test
 


# In[101]:

def do_cross_validation(train_X,train_Y,test_X,test_Y):
    max_depth=[8,10,12,15]
    no_of_trees=[1,10,15,25]
    trees=[]
    depth=[]
    cvscore=[]
    for i in range(len(max_depth)):
        for j in range(len(no_of_trees)):
            
            #dt = DecisionTreeClassifier() 
            #AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)[source]
            dt=DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=max_depth[i], min_samples_split=2, min_samples_leaf=2,random_state=0)
            clf = AdaBoostClassifier(base_estimator=dt,n_estimators=no_of_trees[j],learning_rate=1.0)
            clf=clf.fit(train_X,train_Y)
            out_test=clf.predict(test_X)
            out_train=clf.predict(train_X)
            scores = cross_val_score(clf, train_X,train_Y, cv=5)
            score=np.mean(scores)
            depth.append(max_depth[i])
            trees.append(no_of_trees[j])
            cvscore.append(score)
        #print("score is ",score)
    
    res=pd.DataFrame()
    depth=np.array(depth)
    trees=np.array(trees)
    cvscore=np.array(cvscore)
    print(len(cvscore))
    print(len(max_depth))
    print(len(cvscore))
    res['Maximum_depth_of_the_tree']=pd.Series(depth.flatten())
    res['Number of weak_learners']=pd.Series(trees.flatten())
    res['Cross_Validation _accuracy']=pd.Series(cvscore.flatten())
    res.to_csv("Cross_validated.csv",index=False)
 


# In[99]:

data=pd.read_csv("temporal.csv")
arr=list(data)
#print(arr)
a=['B365H_x', 'B365D_x', 'B365A_x', 'home_buildUpPlaySpeed', 'away_buildUpPlaySpeed', 'home_buildUpPlayDribbling', 'away_buildUpPlayDribbling', 'home_buildUpPlayPassing', 'away_buildUpPlayPassing', 'home_chanceCreationPassing', 'away_chanceCreationPassing', 'home_chanceCreationCrossing', 'away_chanceCreationCrossing', 'home_chanceCreationShooting', 'away_chanceCreationShooting', 'home_defencePressure', 'away_defencePressure', 'home_defenceAggression', 'away_defenceAggression', 'home_defenceTeamWidth', 'away_defenceTeamWidth', 
   'winner','win_by_home_team', 'win_by_away_team']
data=data[a]
data.to_csv("dataset_for_decision_tree.csv")
print(data.shape)
train_X,train_Y,test_X,test_Y=build_the_data(data)
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)
train_Y.shape=7000,
test_Y.shape=1528,
ix,error_on_train,error_on_test=do_ensemble_methods(train_X,train_Y,test_X,test_Y)
print(error)
ix=np.array(ix)
error_on_train=np.array(error_on_train)
error_on_test=np.array(error_on_test)



# In[108]:

def study_precision_vs_recall(train_X,train_Y,test_X,test_Y):
            #dt = DecisionTreeClassifier() 
            #AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)[source]
    confusion=np.zeros((3,3))
    print(confusion)
    dt=DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=15, min_samples_split=2, min_samples_leaf=2,random_state=0)
    clf = AdaBoostClassifier(base_estimator=dt,n_estimators=30,learning_rate=1.0)
    clf=clf.fit(train_X,train_Y)
    out_test=clf.predict(test_X)
    l=len(out_test)
  
    for i in range(l):
        confusion[test_Y[i]][out_test[i]]+=1
    return confusion
#     res['Maximum_depth_of_the_tree']=pd.Series(depth.flatten())
#     res['Number of weak_learners']=pd.Series(trees.flatten())
#     res['Cross_Validation _accuracy']=pd.Series(cvscore.flatten())
#     res.to_csv("Cross_validated.csv",index=False)
 


# In[94]:

plot_graph(ix,error_on_train,error_on_test)


# In[91]:

do_cross_validation(train_X,train_Y,test_X,test_Y)


# In[111]:

confusion=study_precision_vs_recall(train_X,train_Y,test_X,test_Y)
print(confusion)


# In[141]:

t1=confusion[0,:]
t2=confusion[1,:]
t3=confusion[2,:]

t1=pd.Series(t1)
print(t1)
df=pd.DataFrame(columns=['Predicted 0','Predicted 1','Predicted 2'], index=['Actual0','Actual1','Actual2'])
for i in range(3):
    row='Actual'+str(i)
    for j in range (3):
        
        col='Predicted '+str(j)
        df.loc[row,col]=confusion[i][j]
print(df)
df.to_csv("Confusion.csv")

