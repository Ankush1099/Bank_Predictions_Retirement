# Step1: Import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Step2: Import the dataset
bank_df = pd.read_csv('Bank_Customer_retirement.csv')
bank_df.head(5)
bank_df.tail(5)
bank_df.keys()

#Step3: Visualize Dataset
sns.pairplot(bank_df, hue = 'Retire', vars = ['Age','401K Savings'])
sns.countplot(bank_df['Retire'], label = 'Retirement')

#Step4: Model Training
bank_df = bank_df.drop(['Customer ID'], axis = 1) 
bank_df

X = bank_df.drop(['Retire'], axis = 1)
X
y = bank_df['Retire']
y

#Step5: Model Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)

#Step 6: Evaluation
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_predict))
