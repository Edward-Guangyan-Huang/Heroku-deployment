import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

election_data = pd.read_csv('2020_Election_Demographics.csv')

sex = pd.get_dummies(election_data['Sex'],drop_first=True)
marital_status = pd.get_dummies(election_data['Marital Status'],drop_first=True)
race = pd.get_dummies(election_data['Race'],drop_first=True)
age = pd.get_dummies(election_data['Age'],drop_first=True)
education = pd.get_dummies(election_data['Education'],drop_first=True)
income = pd.get_dummies(election_data['Income'],drop_first=True)

election_data2 = pd.concat([sex, race, age, education, income, marital_status, election_data['Candidate']], axis=1)

logmodel = LogisticRegression()
logmodel.fit(election_data2.drop('Candidate', axis=1), election_data2['Candidate'])

# Saving model to disk
pickle.dump(logmodel, open('logmodel.pkl','wb'))

# Loading model to compare the results
logmodel2 = pickle.load(open('logmodel.pkl','rb'))
print(logmodel2.predict([[1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]]))
