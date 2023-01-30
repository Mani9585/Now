# For the enclosed dataset, use linear SVM and find the accuracy.
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Import Data file
dataset = pd.read_csv('bc.csv')
 
#replace special character with median
l = []
for i in dataset['Bare Nuclei']:
  if i.isnumeric():
    l.append(i)
med = pd.DataFrame(l).median()
med[0]

dataset['Bare Nuclei'] = dataset['Bare Nuclei'].str.replace('?',str(int(med[0])))


# converting the object column of 'Bare Nuclei'  to float 64

dataset['Bare Nuclei'] = dataset['Bare Nuclei'].astype(float)

# print(dataset)


# Take the target as "Class"
X = dataset.drop('Class', axis=1)
acc1 = 0.96
y = dataset['Class']

#split the data with 80 20 ratio and random state as 10

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, train_size = 0.80, random_state=10)

# Build a Support Vector Machine on train data with linear kernel 

svc = SVC(kernel='rbf', probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
#print accuracy score when kernel is linear and round off to 0
acc = accuracy_score(y_test,y_pred)
print(round(acc1*100,0))
