from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
def main():
	df = pd.read_csv("submit.csv")
	df=df.sample(frac=1)
	y = df['win']
	X = df
	del X['win']
	X_train = X[0:15000]
	y_train = y[0:15000]
	X_test = X[15000:]
	y_test = y[15000:]
	svc = svm.SVC(kernel='linear')
	C_s = [1e-6, 1e-2, 1.0]

	scores = list()
	scores_std = list()
	for C in C_s:
		svc.C = C
		this_scores = cross_val_score(svc, X, y, n_jobs=1)
		scores.append(np.mean(this_scores)) 
		print(C, scores[-1])

	svc.C = 1.0
	svc.fit(X_train, y_train)
	predicted = svc.predict(X_test)
	print(confusion_matrix(y_test, predicted))


if __name__ == '__main__':
	main()