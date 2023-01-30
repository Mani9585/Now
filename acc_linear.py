# For the enclosed dataset, use  SVM with RBF and print the accuracy (10).
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# Importing Data file

df = pd.read_csv("bc.csv")

#replace special character with median

df['Bare Nuclei']=df['Bare Nuclei'].replace("?",0)
df['Bare Nuclei']=df['Bare Nuclei'].replace(0,df['Bare Nuclei'].median())
# converting the object column  to float

df['Bare Nuclei']=pd.to_numeric(df['Bare Nuclei'])

#split the data with 80 20 and random state as 10
x = df.drop('Class',axis = 1)
y=df['Class']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.transform(x_test)

from sklearn.svm import SVC
obj = SVC(kernel="linear",random_state = 0)
obj.fit(x_train,y_train)

y_pred = obj.predict(x_test)

acc = accuracy_score(y_test,y_pred)*100
print(round(acc,0))
