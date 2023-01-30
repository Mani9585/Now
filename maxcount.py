#For the enclosed dataset, find the Class that has maximum count (5 marks)
import pandas as pd
import numpy as np

# Import Data file

dataset = pd.read_csv('bc.csv')

#Print which class has the max count. (5 marks)
d = dataset['Class'].value_counts().index[0]
print(d)
