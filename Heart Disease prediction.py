# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score





# Data Collection and Processing
  # loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('D:\Project\heart_disease_data.csv')

# print first 5 rows of the dataset
heart_data.head()
# print(heart_data.head())

# print last 5 rows of the dataset
heart_data.tail()
#print(heart_data.tail())

# The number of rows and columns in the dataset
heart_data.shape
#print(heart_data.shape)

# getting some info about the data
heart_data.info()
#print(heart_data.info())

#Checking for missing value
heart_data.isnull().sum()
#print(heart_data.isnull().sum())

# Statistical measure about the data
heart_data.describe()
#print(heart_data.describe())

#Checking the distribution of Target variable
heart_data['target'].value_counts()
#print(heart_data['target'].value_counts())


#--> Defective Heart 
# 0--> Defective Heart 
# 1--> Healthy Heart

#Splitting the Features and Target 
X = heart_data.drop(columns='target',axis=1)
#print(X)
Y = heart_data['target']
#print(Y)

# Splitting the Data into Training Data & Test Data
X_train,X_test, Y_train, Y_test =train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

# Model Training
 #Logistics Regression
model = LogisticRegression()
#training the LogisticsRegression model with Training  Data
model.fit(X_train, Y_train)

#Model Evaluation

#Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)
#print('Accuracy on Training data :', training_data_accuracy)

# accuracy on test  data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)
#print('Accuracy on Test data :', test_data_accuracy)

#Building  a Predective System 
input_data =(51,1,0,140,298,0,1,122,1,4.2,1,3,3)

# Change the input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediciton= model.predict(input_data_reshaped)
print(prediciton)

if (prediciton[0]==0):
    print('The Person does not have a Heart Disease')
else:
    print('The Person has heart Disease')