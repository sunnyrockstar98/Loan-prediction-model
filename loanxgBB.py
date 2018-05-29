


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
testset = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)
dataset['Loan_Status'] = dataset['Loan_Status'].map(lambda x: 1 if x=='Y' else 0)

dataset['Credit_History'].fillna(dataset['Loan_Status'], inplace = True)
#for dependents
dataset['Dependents'] = dataset['Dependents'].replace({'3+':3})
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)




testset['Gender'].fillna(testset['Gender'].mode()[0],inplace=True)
testset['Married'].fillna(testset['Married'].mode()[0],inplace=True)
testset['Self_Employed'].fillna(testset['Self_Employed'].mode()[0],inplace=True)
testset['Credit_History'].fillna(testset['Credit_History'].mode()[0],inplace=True)
testset['Dependents'] = testset['Dependents'].replace({'3+':3})
testset['Dependents'].fillna(testset['Dependents'].mode()[0],inplace=True)


X = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]].values
Xtest = testset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 8:9])
X[:, 8:9] = imputer.transform(X[:, 8:9])
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer1.fit(X[:, 7:8])
X[:, 7:8] = imputer1.transform(X[:, 7:8])

imputer2 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer2 = imputer2.fit(Xtest[:, 8:9])
Xtest[:, 8:9] = imputer2.transform(Xtest[:, 8:9])
imputer12 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer12 = imputer12.fit(Xtest[:, 7:8])
Xtest[:, 7:8] = imputer12.transform(Xtest[:, 7:8])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
#labelencoder_X111 = LabelEncoder()
#X[:, 2] = labelencoder_X111.fit_transform(X[:, 2])
labelencoder_X2 = LabelEncoder()
X[:, 3] = labelencoder_X2.fit_transform(X[:, 3])
labelencoder_X3 = LabelEncoder()
X[:, 4] = labelencoder_X3.fit_transform(X[:, 4])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()
labelencoder_X4 = LabelEncoder()
X[:, 10] = labelencoder_X4.fit_transform(X[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [10])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]


labelencoder_Xtest = LabelEncoder()
Xtest[:, 0] = labelencoder_Xtest.fit_transform(Xtest[:, 0])

labelencoder_Xtest1 = LabelEncoder()
Xtest[:, 1] = labelencoder_Xtest1.fit_transform(Xtest[:, 1])
#labelencoder_Xtest111 = LabelEncoder()
#Xtest[:, 2] = labelencoder_Xtest111.fit_transform(Xtest[:, 2])
labelencoder_Xtest2 = LabelEncoder()
Xtest[:, 3] = labelencoder_Xtest2.fit_transform(Xtest[:, 3])
labelencoder_Xtest3 = LabelEncoder()
Xtest[:, 4] = labelencoder_Xtest3.fit_transform(Xtest[:, 4])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#Xtest = onehotencoder.fit_transform(Xtest).toarray()
labelencoder_Xtest4 = LabelEncoder()
Xtest[:, 10] = labelencoder_Xtest4.fit_transform(Xtest[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [10])
Xtest = onehotencoder.fit_transform(Xtest).toarray()

Xtest = Xtest[:, 1:]
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, y)

y_pred = classifier.predict(Xtest)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
accuracies.mean()
accuracies.std()



y_pred=list(y_pred)
for j in range(len(y_pred)):
    if y_pred[j]==1:
        y_pred[j]='Y'
    else:
        y_pred[j]='N'