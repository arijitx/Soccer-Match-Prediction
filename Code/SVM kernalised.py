from  sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from  sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
def main():
    df = pd.read_csv("submit.csv")
    df=df.sample(frac=1)
    Y = df['win']
    print(df.shape)
    Y = Y[0:15000]
    X = df[0:15000]
    #X=df.copy(deep=True)
    del X['win']
    train = X.values
    train_out = Y.values
    print(X.shape, "   ", Y.shape)
    valid_out = df['win']
    #valid_out = valid_out[7000:10000]
    valid_X = df[0:15000]
    valid_out = valid_out[0:15000]
    #valid_X = df.copy(deep=True)

    del valid_X['win']
    valid_out = valid_out.values
    valid_X = valid_X.values
    #SVM RBF Kernel
    params={'C':[1,10],'gamma':[0.1]}
    classify = SVC(kernel='rbf',class_weight='balanced')
    classify = GridSearchCV(classify,params)
    classify.fit(train, train_out)
    predict = classify.predict(valid_X)
    values,counts=np.unique(predict,return_counts=True)
    print(values)
    print(counts)
    # print(predict.shape)
    # print(valid_out.shape)
    # np.save("final_y",predict)
    print("SVM RBF Accuracy: ",classify.score(valid_X, valid_out))
    print(confusion_matrix(valid_out,predict))
    #SGD Classifier
    classify = SGDClassifier(class_weight='balanced')
    classify.fit(train, train_out)
    predict = classify.predict(valid_X)
    values, counts = np.unique(predict, return_counts=True)
    print(values)
    print(counts)
    print("SGD Accuracy: ",classify.score(valid_X, valid_out))
    print(confusion_matrix(valid_out, predict))


if __name__ == '__main__':
    main()