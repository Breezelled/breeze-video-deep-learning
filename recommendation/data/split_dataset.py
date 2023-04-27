import csv
import pandas as pd
from sklearn.model_selection import train_test_split

my_matrix = pd.read_csv("dataset.csv", header=None, quoting=csv.QUOTE_ALL)
x, y = my_matrix.iloc[:, :-1], my_matrix.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=36)
train = pd.concat([x_train, y_train], axis=1)
train.to_csv('train.csv', index=False, header=None)
test = pd.concat([x_test, y_test], axis=1)
test.to_csv('test.csv', index=False, header=None)
