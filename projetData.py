# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 23:14:17 2021

@author: Maison info
"""
# import biblio

import numpy as np
import pandas as pd
import random as rnd

# visualisation biblio
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set()
# machine learning biblio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# import data et suppression des données inutiles 
#df_train=pd.read_csv('OneDrive/Bureau/train1.csv')
df_train=pd.read_csv('C:/Users/Maison info/OneDrive/Bureau/DataMiningTitanic_Ons_Gadhoum/train1.csv')
#print (df_train.isnull().sum())
df_train=df_train.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

#df_test=pd.read_csv('OneDrive/Bureau/test.csv')
df_test=pd.read_csv('C:/Users/Maison info/OneDrive/Bureau/DataMiningTitanic_Ons_Gadhoum/test.csv')
df_test=df_test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

#gestion des valeurs  age, fare and embarked 

df_train['Age']=df_train['Age'].fillna(df_train['Age'].mean())
df_train['Fare']=df_train['Fare'].fillna(df_train['Fare'].mean())
df_train=df_train.dropna()
df_test['Age']=df_test['Age'].fillna(df_test['Age'].mean())
df_test['Fare']=df_test['Fare'].fillna(df_test['Fare'].mean())

common_value = 'S'
data = [df_train, df_test]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
    
    
# remplacer male par 0 et female par 1
df_train['Sex'].replace(['male', 'female'],[0,1],inplace=True)
df_test['Sex'].replace(['male', 'female'],[0,1],inplace=True)

# trasnformation d'Embarked
ports = {"S": 0, "C": 1, "Q": 2}
data = [df_train, df_test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
    
    
# remplacer fare par 0, 1, 2, 3, 4, 5
data = [df_train, df_test]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

#  transformation d'age
data = [df_train, df_test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

#declaration pour les algo
X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test
#y_class=pd.read_csv('OneDrive/Bureau/gender_submission.csv')
#y_test=y_class['Survived'].values

# KNN 
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train) 
Y_pred = knn.predict(X_test) 
#acc_knn2=round(knn.score(X_test,y_test)* 100, 2) 
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
knn.predict(X_test)
#arbre de décision
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)
 #SVM
linear_svc = SVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2) 

#quel est le meilleur Modele
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head()
# confusion matrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(decision_tree, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)

#calcul de précision

from sklearn.metrics import precision_score, recall_score
print("Precision_ArbreDeDecision:", precision_score(Y_train, predictions))


predictions2 = cross_val_predict(knn, X_train, Y_train, cv=3)
print("Precision_knn:", precision_score(Y_train, predictions2))

predictions3 = cross_val_predict(linear_svc, X_train, Y_train, cv=3)
print("Precision_SVM:", precision_score(Y_train, predictions3))

from sklearn.metrics import f1_score
f1_score(Y_train, predictions)
print(f1_score)


print(acc_knn)
print(acc_decision_tree)
print(acc_linear_svc)

from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = decision_tree.predict_proba(X_train)
y_scores = y_scores[:,1]
# precision et recall courbe 
precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()



