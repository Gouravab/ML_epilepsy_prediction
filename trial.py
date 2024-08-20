import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
ESR = pd.read_csv('data.csv')
#print(ESR.head())
cols = ESR.columns
tgt = ESR.y
tgt[tgt>1]=0
ax = sn.countplot(tgt,label="Count")
non_seizure, seizure = tgt.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)
# till here is for detecting which type of seizure class
X = ESR.iloc[:,1:179].values
y = ESR.iloc[:,179].values
#logistic regresssion
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round(clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' %')
from scipy.io import savemat
savemat('eeg_results_improved.mat', {'y_test': y_test, 'y_pred': y_pred_log_reg})
