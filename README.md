# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.Shanmathi
RegisterNumber: 212222100049
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

df.tail()

#Array value of X
X=df.iloc[:,:-1].values
X

#Array value of Y
Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours Vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
### df.head()
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/b8ce1d61-b828-40e3-8b77-bd6d8cf8a067)

### df.tail()
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/c3c0295c-50fe-4d92-a0fc-f665b93f4bc5)

### Array value of X
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/a5ee3ba7-15af-40ba-aee8-c5bdda7fe31b)

### Array value of Y
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/b49f7299-b828-45d1-84cc-d79672f3236e)

### Values of Y prediction
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/9b9b6375-dc17-4bb0-8acd-5bd98153612c)

### Array values of Y test
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/027a1a39-6de4-455b-9c32-95eabe2541fa)

### Training Set Graph

![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/1a1898d6-4be2-459a-9d69-8d4a2524bf8e)

### Test Set Graph

![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/af17ef9e-4a03-4f64-874e-953e230fff50)

### Values of MSE, MAE and RMSE

![image](https://github.com/ShanmathiShanmugam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121243595/dae19f26-1dc7-4481-b702-aa7b19126ccf)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
